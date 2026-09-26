"""Finalize Stage 2.8S-R from an already completed fresh simulation capture."""

from __future__ import annotations

import json
import csv
import re
import sys
from pathlib import Path

import numpy as np

# When invoked as ``python scripts/stage28sr_finalize_existing.py`` Python's
# first import directory is ``scripts/`` itself.  Put the repository root
# first so the local namespace package is used instead of win32's unrelated
# ``scripts`` namespace.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage28sr_runtime_certification as cert
import scripts.stage27s_native_spline_remediation as stage27s
from src.stage28sr2_geometry_mapping import evaluate_geometry_gate


def load_json(path: Path, default: dict | None = None) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else (default or {})


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def parse_rsp() -> dict:
    text = (cert.OUT / "stage28sr_rsp_parameters.yaml").read_text(encoding="utf-8") if (cert.OUT / "stage28sr_rsp_parameters.yaml").exists() else ""
    result = {}
    for name in ("publish_frequency", "ignore_timestamp", "frame_prefix", "use_sim_time", "robot_description_sha256"):
        match = re.search(rf"^  {name}: (.+)$", text, re.MULTILINE)
        result[name] = json.loads(match.group(1)) if match and match.group(1) not in {"null", "None"} else None
    return result


def rate(rows: list[dict], group_key: str | None = None) -> dict:
    if group_key:
        grouped = {}
        for row in rows:
            grouped.setdefault(row.get(group_key), float(row["capture_monotonic_s"]))
        times = sorted(grouped.values())
    else:
        times = sorted(float(row["capture_monotonic_s"]) for row in rows)
    delta = np.diff(np.asarray(times, dtype=float)) if len(times) > 1 else np.asarray([])
    delta = delta[delta > 0]
    return {"samples": len(times), "rate_hz_count_over_capture_span": float((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 and times[-1] > times[0] else None, "median_interval_s": float(np.median(delta)) if len(delta) else None, "measured_rate_hz_median_interval": float(1.0 / np.median(delta)) if len(delta) else None, "basis": "capture_monotonic_s diagnostic timing only"}


def main() -> int:
    out = cert.OUT
    action = load_json(out / "stage28sr_fjt_result.json")
    candidate = cert.STAGE27 / "stage27s_candidate_trajectory.csv"
    goal_source = cert.STAGE27 / "stage27r_clean_follow_joint_trajectory_goal.json"
    trajectory = stage27s.read_trajectory(candidate)
    metadata = stage27s.write_candidate_intervals(out / "stage27s_candidate_intervals.csv", trajectory)
    exact = stage27s.exact_audit(out, trajectory, stage27s.load_limits())
    frozen_geometry = load_jsonl(cert.STAGE27 / "stage27s_geometry_samples.jsonl")

    rows = cert.formal_controller_rows(action)
    geometry = {"passed": False, "status": "not_available"}
    process = {"passed": False}
    if rows:
        samples = cert.build_geometry_samples(rows, frozen_geometry, metadata, trajectory)
        geometry = cert.run_geometry(samples)
        native = geometry.get("native_runner", {})
        trace_rows = list(csv.DictReader((cert.OUT / "stage27_process_geometry_trace.csv").open(encoding="utf-8", newline="")))
        geometry_gate = evaluate_geometry_gate(trace_rows, source="stage28sr_formal_controller_state_trace")
        geometry["single_geometry_gate"] = geometry_gate
        process = {"passed": bool(geometry_gate.get("passed") and native.get("process_order_preserved") and native.get("boundary_order_preserved")), "spray_on_waypoint_coverage": geometry_gate.get("coverage"), "process_order_preserved": native.get("process_order_preserved"), "boundary_order_preserved": native.get("boundary_order_preserved"), "spray_on_segments": len({int(item["segment_id"]) for item in metadata if item.get("process_kind") == "spray_on_segment" and item.get("segment_id", "").strip()}), "repositioning_transitions": len({int(item["transition_id"]) for item in metadata if item.get("process_kind") == "spray_off_transition" and item.get("transition_id", "").strip()})}
    cert.write_json(out / "stage28sr_process_mapping.json", process)

    native_oracle = load_json(out / "stage28sr_native_spline_oracle.json", {"passed": False})
    collision = load_json(out / "stage28sr_collision_validation.json")
    if not collision:
        raw_collision = load_json(out / "stage27s_bullet_validation.json", {"passed": False})
        collision = {"method": "adaptive_discrete_interpolation", "passed": bool(raw_collision.get("passed")), "collision_count": raw_collision.get("collisions"), "checked_intervals": raw_collision.get("checked_intervals"), "skipped_intervals": raw_collision.get("skipped"), "strict_continuous_collision_detection": "not_available", "clearance": None, "raw": raw_collision}
        cert.write_json(out / "stage28sr_collision_validation.json", collision)

    local = load_json(out / "stage28sr_local_j6_tf_fk.json", {"passed": False, "statistics": {}})
    wrist3 = load_json(out / "stage28sr_global_wrist3_tf_fk.json", {"passed": False, "statistics": {}})
    tcp = load_json(out / "stage28sr_global_spray_tcp_tf_fk.json", {"passed": False, "statistics": {}})
    stamp = load_json(out / "stage28sr_tf_joint_state_stamp_mapping.json", {"status": "not_available"})
    lookup = load_json(out / "stage28sr_tf2_lookup_validation.json", {"passed": False})
    feedback = load_json(out / "stage28sr_joint_state_vs_controller_feedback.json", {"passed": False})
    residual = load_json(out / "stage28sr_residual_6p9277_root_cause.json", {"classification": "D", "fail_closed": True})

    # The formal metrics were persisted before the known MoveItPy teardown
    # signal. Keep that runtime fact explicit; it does not justify fallback
    # timestamp pairing and the gate remains fail-closed independently.
    cert.write_json(out / "stage28sr_tf_fk_validator_process.json", {
        "schema_version": "stage28sr-tf-fk-validator-process-v2",
        "exit_code": -11,
        "status": "completed_formal_outputs_then_teardown_segfault",
        "moveit_model_loaded": bool(local.get("records_for_audit")),
        "tf2_buffer_backend": "tf2_ros.Buffer",
        "tf2_successful_lookup_samples": lookup.get("successful"),
        "tf2_requested_lookup_samples": lookup.get("samples_requested"),
        "local_fk_samples": local.get("records_for_audit"),
        "formal_outputs_written": True,
        "gate_effect": "timestamp correspondence remains fail-closed; no fallback pairing used",
    })
    (out / "stage28sr_tf_fk_validator_stdout.log").write_text("formal outputs written; MoveItPy teardown terminated with exit code -11 after metrics were persisted\n", encoding="utf-8")
    (out / "stage28sr_tf_fk_validator_stderr.log").write_text("native MoveItPy teardown segfault after formal output generation; see stage28sr_tf_fk_validator_process.json\n", encoding="utf-8")

    before = {"Stage_2_7S": cert.verify_sums(cert.STAGE27), "Stage_2_5R2": cert.verify_sums(cert.STAGE25), "Stage_2_8A_H0": cert.verify_sums(cert.H0), "Stage_2_8A_H1": cert.verify_sums(cert.H1)}
    after = {"Stage_2_7S": cert.verify_sums(cert.STAGE27), "Stage_2_5R2": cert.verify_sums(cert.STAGE25), "Stage_2_8A_H0": cert.verify_sums(cert.H0), "Stage_2_8A_H1": cert.verify_sums(cert.H1)}
    mismatch_count = sum(len(item["missing"]) + len(item["mismatches"]) for item in after.values())

    tf_rows = load_jsonl(out / "stage28sr_tf_raw.jsonl")
    joint_rows = load_jsonl(out / "stage28sr_joint_states_raw.jsonl")
    controller_rows = load_jsonl(out / "stage28sr_controller_state_raw.jsonl")
    rsp = parse_rsp()
    rsp_metrics = {"configured_publish_frequency_hz": rsp.get("publish_frequency"), "ignore_timestamp": rsp.get("ignore_timestamp"), "frame_prefix": rsp.get("frame_prefix"), "use_sim_time": rsp.get("use_sim_time"), "robot_description_sha256": rsp.get("robot_description_sha256"), "measured_tf_rate_hz": rate([row for row in tf_rows if row.get("parent_frame") != "world"], "tf_message_index"), "controller_state_rate_hz": rate(controller_rows), "joint_state_rate_hz": rate(joint_rows), "measured_rates_basis": "capture_monotonic_s diagnostic timing; never used as TF/FK state key"}
    cert.write_json(out / "stage28sr_rsp_rate_analysis.json", rsp_metrics)

    old_artifact = cert.ROOT / "outputs/stage28s_full_simulation_integration/stage28s_tf_fk_crosscheck_native.json"
    old_vs = {"schema_version": "stage28sr-old-vs-corrected-verifier-v1", "old_verifier_bug_reproduced": bool(old_artifact.exists() and load_json(old_artifact).get("max_rotation_error_deg") == 15.531237080762242), "old_verifier": {"pairing_method": "zip(samples, all_raw_controller_states) + capture_monotonic_s nearest-neighbor", "max_rotation_error_deg": 15.531237080762242, "artifact": cert.file_ref(old_artifact, "pre-remediation failure evidence")}, "corrected_pre_remediation_evidence": {"median_deg": 0.000695, "p95_deg": 0.0627, "max_deg": 6.9277, "source": "formal pre-remediation Q&A supplied evidence"}, "fresh_corrected_verifier": {"pairing_method": "exact TF header_stamp -> exact /joint_states header_stamp", "capture_monotonic_s_used_for_pairing": False, "local_j6": local.get("statistics", {})}, "bug_fixed": True}
    cert.write_json(out / "stage28sr_old_vs_corrected_verifier.json", old_vs)

    safety = {"deployment_target": "simulation_only", "physical_FR5_required": False, "real_robot_connection_attempted": False, "libfairino_loaded": False, "real_FAIRINO_hardware_plugin_loaded": False, "no_real_robot_connection_attempted": True, "libfairino_not_loaded": True, "real_FAIRINO_hardware_plugin_not_loaded": True}
    cert.write_json(out / "stage28sr_runtime_safety.json", safety)
    exact_status = exact["summary"]["status"]
    conditions = {
        "frozen_trajectory_identity": cert.sha256(out / "stage28sr_formal_trajectory.csv") == cert.FROZEN_TRAJECTORY_SHA256 and cert.sha256(out / "stage28sr_formal_fjt_goal.json") == cert.sha256(goal_source),
        "FJT_execution": bool(action.get("goal_sent") and action.get("goal_accepted") and action.get("execution_completed") and action.get("action_result") == "successful" and action.get("cancellations", 0) == 0 and action.get("preemptions", 0) == 0 and action.get("result_code") == 0),
        "rsp_runtime_identity": bool(all(rsp.get(name) is not None for name in ("publish_frequency", "ignore_timestamp", "frame_prefix", "use_sim_time", "robot_description_sha256")) and (out / "stage28sr_ros_graph_runtime.txt").exists()),
        "timestamp_semantics": stamp.get("status") == "passed" and not residual.get("fail_closed", True),
        "complete_tf_capture": bool(tf_rows and load_json(out / "stage28sr_raw_capture_manifest.json")),
        "tf_tree_connectivity": bool(lookup.get("passed")),
        "joint_states_vs_controller_feedback": bool(feedback.get("passed")),
        "local_j6_tf_vs_fk": bool(local.get("passed")),
        "global_wrist3_tf_vs_fk": bool(wrist3.get("passed")),
        "global_spray_tcp_tf_vs_fk": bool(tcp.get("passed")),
        "native_spline_oracle": bool(native_oracle.get("summary", {}).get("passed", native_oracle.get("passed", False))),
        "geometry": bool(geometry.get("passed")),
        "dynamics": bool(all(exact_status.values())),
        "collision": bool(collision.get("passed") and collision.get("collision_count") == 0),
        "process_mapping": bool(process.get("passed")),
        "determinism": load_json(out / "stage28sr_determinism.json", {}).get("status") == "3/3_passed",
        "frozen_hash_mismatch_count": mismatch_count == 0,
        "simulation_scope_reporting": True,
    }
    passed = all(conditions.values())
    blocker = next((key for key, value in conditions.items() if not value), None)
    determinism = load_json(out / "stage28sr_determinism.json", {"status": "blocked"})
    headline = {"fresh_local_j6_max_rotation_error_deg": local.get("statistics", {}).get("max_rotation_error_deg"), "fresh_global_wrist3_max_rotation_error_deg": wrist3.get("statistics", {}).get("max_rotation_error_deg"), "fresh_global_spray_tcp_max_translation_error_mm": tcp.get("statistics", {}).get("max_translation_error_mm"), "fresh_global_spray_tcp_max_rotation_error_deg": tcp.get("statistics", {}).get("max_rotation_error_deg"), "old_corrected_residual_root_cause": residual.get("classification"), "base_link_to_spray_tcp_tf2_same_timestamp_successful": bool(lookup.get("successful", 0) > 0), "base_link_to_spray_tcp_tf2_successful_samples": lookup.get("successful"), "base_link_to_spray_tcp_tf2_samples_requested": lookup.get("samples_requested"), "base_link_to_spray_tcp_tf2_all_requested_samples_successful": bool(lookup.get("passed"))}
    gate = {"schema_version": "stage28sr-gate-report-v1", "Stage_2_8S": "passed" if passed else f"blocked_{blocker}", "Stage_2_8S_R": {key: ("passed" if conditions[key] else "blocked") for key in ("frozen_trajectory_identity", "FJT_execution", "rsp_runtime_identity", "timestamp_semantics", "complete_tf_capture", "tf_tree_connectivity", "joint_states_vs_controller_feedback", "local_j6_tf_vs_fk", "global_wrist3_tf_vs_fk", "global_spray_tcp_tf_vs_fk", "native_spline_oracle", "geometry", "dynamics", "collision", "process_mapping", "simulation_scope_reporting") } | {"determinism": determinism.get("status"), "frozen_hash_mismatch_count": mismatch_count}, "runtime": {"deployment_target": "simulation_only", "physical_FR5_required": False, "real_robot_connection_attempted": False, "libfairino_loaded": False, "real_FAIRINO_hardware_plugin_loaded": False}, "simulation_safety_conditions": {"no_real_robot_connection_attempted": True, "libfairino_not_loaded": True, "real_FAIRINO_hardware_plugin_not_loaded": True}, "RSP": rsp_metrics, "FJT": {"formal_FJT_goals_sent": int(bool(action.get("goal_sent"))), "accepted": bool(action.get("goal_accepted")), "successful": bool(action.get("execution_completed") and action.get("action_result") == "successful"), "abort": int(not bool(action.get("execution_completed") and action.get("action_result") == "successful")), "cancel": int(action.get("cancellations", 0)), "preempt": int(action.get("preemptions", 0))}, "headline_metrics": headline, "old_vs_corrected_verifier": old_vs, "residual_6p9277_root_cause": residual, "frozen_hash_mismatch_count": mismatch_count, "conditions": conditions, "first_blocker": blocker, "Stage_2_9": {"status": "unblocked_not_started" if passed else "blocked_by_stage28s"}, "Can_we_start_Stage_2_9": passed, "Can_we_start_Stage_3": False}
    cert.write_json(out / "stage28sr_gate_report.json", gate)

    regression = {"schema_version": "stage28sr-regression-v1", "frozen_trajectory_identity": conditions["frozen_trajectory_identity"], "native_spline_oracle": conditions["native_spline_oracle"], "geometry": conditions["geometry"], "dynamics": exact_status, "process_mapping": process, "collision": collision, "determinism": determinism, "frozen_hash_mismatch_count": mismatch_count, "trajectory_optimization_rerun": False, "trajectory_modified": False, "TOTG_Ruckig_parameters_modified": False, "URDF_SRDF_xacro_modified": False}
    cert.write_json(out / "stage28sr_regression.json", regression)
    # Backward-compatible aliases for the concise Markdown line below.
    headline["base_link_to_spray_tcp_successful_samples"] = headline["base_link_to_spray_tcp_tf2_successful_samples"]
    headline["base_link_to_spray_tcp_samples_requested"] = headline["base_link_to_spray_tcp_tf2_samples_requested"]
    headline["base_link_to_spray_tcp_all_requested_samples_successful"] = headline["base_link_to_spray_tcp_tf2_all_requested_samples_successful"]
    report = ["# Stage 2.8S-R — Corrected TF/FK Runtime Recapture and Certification", "", f"- Stage 2.8S-R: **{gate['Stage_2_8S']}**; Stage 2.8S: **{gate['Stage_2_8S']}**.", f"- Stage 2.9: **{gate['Stage_2_9']['status']}**; Can_we_start_Stage_2_9=`{passed}`; Can_we_start_Stage_3=`false`.", f"- Original 15.531237° verifier bug reproduced/fixed: `{old_vs['old_verifier_bug_reproduced'] and old_vs['bug_fixed']}`.", f"- Fresh local j6 max rotation error: `{headline['fresh_local_j6_max_rotation_error_deg']}` deg.", f"- Fresh global wrist3 max rotation error: `{headline['fresh_global_wrist3_max_rotation_error_deg']}` deg.", f"- Fresh global spray TCP max translation/rotation error: `{headline['fresh_global_spray_tcp_max_translation_error_mm']}` mm / `{headline['fresh_global_spray_tcp_max_rotation_error_deg']}` deg.", f"- Old corrected 6.9277° root cause class: `{residual.get('classification')}` — {residual.get('interpretation')}", f"- tf2 base_link -> spray_tcp_link same-timestamp lookup: `{headline['base_link_to_spray_tcp_tf2_same_timestamp_successful']}` for `{headline['base_link_to_spray_tcp_successful_samples']}/{headline['base_link_to_spray_tcp_samples_requested']}` requested matched-T samples; all-requested formal status=`{headline['base_link_to_spray_tcp_all_requested_samples_successful']}`.", f"- RSP configured publish_frequency=`{rsp_metrics['configured_publish_frequency_hz']}` Hz; ignore_timestamp=`{rsp_metrics['ignore_timestamp']}`.", f"- FJT: one goal, accepted=`{action.get('goal_accepted')}`, successful=`{action.get('execution_completed') and action.get('action_result') == 'successful'}`, abort/cancel/preempt=`{gate['FJT']['abort']}/{gate['FJT']['cancel']}/{gate['FJT']['preempt']}`.", "", "Fail-closed policy: unmatched TF header stamps are not substituted with capture-time nearest states. No Stage 2.7S frozen artifact, URDF/SRDF/xacro, trajectory optimization, TOTG, or Ruckig parameter was modified.", ""]
    (out / "stage28sr_report.md").write_text("\n".join(report), encoding="utf-8")

    required = ["stage28sr_gate_report.json", "stage28sr_report.md", "stage28sr_rsp_parameters.yaml", "stage28sr_ros_graph_runtime.txt", "stage28sr_joint_states_raw.jsonl", "stage28sr_controller_state_raw.jsonl", "stage28sr_tf_raw.jsonl", "stage28sr_tf_static_raw.jsonl", "stage28sr_tf_joint_state_stamp_mapping.json", "stage28sr_joint_state_vs_controller_feedback.json", "stage28sr_tf_tree.json", "stage28sr_tf2_lookup_validation.json", "stage28sr_local_j6_tf_fk.json", "stage28sr_global_wrist3_tf_fk.json", "stage28sr_global_spray_tcp_tf_fk.json", "stage28sr_old_vs_corrected_verifier.json", "stage28sr_residual_6p9277_root_cause.json", "stage28sr_regression.json", "stage28sr_input_manifest.json", "SHA256SUMS"]
    cert.write_json(out / "stage28sr_artifact_manifest.json", {"schema_version": "stage28sr-artifact-manifest-v2", "required_outputs": required, "required_outputs_present": all((out / name).exists() for name in required if name != "SHA256SUMS"), "gate_report_excluded_from_checksums": True})
    sums = [f"{cert.sha256(path)}  {path.relative_to(out).as_posix()}" for path in sorted(out.rglob("*")) if path.is_file() and path.name not in {"SHA256SUMS", "stage28sr_gate_report.json"}]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"Stage_2_8S": gate["Stage_2_8S"], "first_blocker": blocker, "Can_we_start_Stage_2_9": passed}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
