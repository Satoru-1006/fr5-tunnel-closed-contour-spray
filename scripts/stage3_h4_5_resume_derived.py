#!/usr/bin/env python3
"""Finish the H4.5 derived-model replay after a wrapper-only repair."""

from __future__ import annotations

import json
import re
from pathlib import Path

import stage3_h4_5 as h45
import stage3_h4_2 as h42


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z"


def main() -> int:
    inputs = h45.h44.locate_inputs()
    paths = h45.h44.base_paths(inputs)
    selected = h45.load_json(OUTPUT / "stage3_h4_5_selected_candidate.json")["selected_candidate"]
    metrics = h45.load_jsonl(OUTPUT / "stage3_h4_5_candidate_metrics.jsonl")
    selected_row = next(row for row in metrics if row.get("candidate_id") == selected["candidate_id"] and row.get("status") == "EVALUATED")
    baseline_record = selected_row["fresh_process_record"]
    baseline_result = h45.load_json(ROOT / str(baseline_record["worker_dir"]) / "worker_result.json")
    fields = ("raw_ik_candidate_semantic_hash", "fk_hash", "fcl_hash", "classification_hash", "aggregate_metrics_hash", "semantic_hash")
    base_hashes = h42.replay_hashes(baseline_result)
    diff_path = OUTPUT / "stage3_h4_5_derived_configuration_diff.json"
    diff = h45.load_json(diff_path)
    derived_urdf = OUTPUT / "derived_robot_model.urdf"
    replay = None
    if diff.get("replay_validation", {}).get("status") == "PASSED":
        comparison = diff["replay_validation"]
    else:
        replay = h45.run_derived_replay(OUTPUT, selected, 9004, derived_urdf, paths["srdf"], paths["h2_manifest"], paths["h3_targets"], paths["kinematics"])
        derived_hashes = h42.replay_hashes(replay["result"])
        comparison = {
            "required": True,
            "status": "PASSED" if replay["result"].get("target_count") == 192 and all(base_hashes[field] == derived_hashes[field] for field in fields) else "BLOCKED",
            "baseline_hashes": base_hashes,
            "derived_hashes": derived_hashes,
            "hash_equality": {field: base_hashes[field] == derived_hashes[field] for field in fields},
            "derived_target_count": replay["result"].get("target_count"),
            "derived_per_geometry": replay["result"].get("metrics", {}).get("per_geometry"),
            "fresh_process_record": replay["record"],
        }
    diff["replay_validation"] = comparison
    h45.dump_json(diff_path, diff)
    h45.dump_json(OUTPUT / "stage3_h4_5_derived_replay_summary.json", {"derived_configuration": diff, "replay": replay["record"] if replay else comparison.get("fresh_process_record")})

    fk_state = h45.load_json(OUTPUT / "fk_audit_input.json")
    fk_audit = h45.run_fk_audit(OUTPUT, fk_state)

    gate_path = OUTPUT / "stage3_h4_5_gate_report.json"
    gate = h45.load_json(gate_path)
    mandatory = gate["mandatory_gates"]
    mandatory["derived_model_ambiguity_resolved"] = comparison["status"] == "PASSED"
    mandatory["native_fk_audit_passed"] = fk_audit.get("status") == "PASSED"
    gate["mandatory_gates"] = mandatory
    gate["derived_configuration"] = comparison
    if all(mandatory.values()):
        gate["STAGE_3_H4_5"] = "PASSED"
        gate["FIRST_BLOCKER"] = None
        gate["READY_FOR_STAGE_3_H5"] = "YES"
    else:
        gate["STAGE_3_H4_5"] = "BLOCKED"
        gate["FIRST_BLOCKER"] = next(key for key, value in mandatory.items() if not value)
        gate["READY_FOR_STAGE_3_H5"] = "NO"
    h45.dump_json(gate_path, gate)

    terminal_path = OUTPUT / "stage3_h4_5_terminal_certificate.json"
    terminal = h45.load_json(terminal_path)
    terminal["STAGE_3_H4_5"] = gate["STAGE_3_H4_5"]
    terminal["FIRST_BLOCKER"] = gate["FIRST_BLOCKER"]
    terminal["READY_FOR_STAGE_3_H5"] = gate["READY_FOR_STAGE_3_H5"]
    terminal["DERIVED_REPLAY"] = comparison["status"]
    h45.dump_json(terminal_path, terminal)

    config_path = OUTPUT / "stage3_h4_5_configuration_manifest.json"
    config = h45.load_json(config_path)
    config["derived_replay_status"] = comparison["status"]
    h45.dump_json(config_path, config)

    report_path = OUTPUT / "FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("STAGE_3_H4_5: BLOCKED", f"STAGE_3_H4_5: {gate['STAGE_3_H4_5']}")
    report = report.replace("FIRST_BLOCKER: derived_model_ambiguity_resolved", f"FIRST_BLOCKER: {gate['FIRST_BLOCKER'] or 'none'}")
    report = report.replace("READY_FOR_STAGE_3_H5: NO", f"READY_FOR_STAGE_3_H5: {gate['READY_FOR_STAGE_3_H5']}")
    report = report.replace("Derived replay status: `BLOCKED`", f"Derived replay status: `{comparison['status']}`")
    report = report.replace("Model ambiguity resolved: `False`", f"Model ambiguity resolved: `{comparison['status'] == 'PASSED'}`")
    report = re.sub(r"Q13\. Native FK audit: `[^`]+`", f"Q13. Native FK audit: `{fk_audit.get('status')}`", report)
    report = report.replace("current result is `NO`", f"current result is `{gate['READY_FOR_STAGE_3_H5']}`")
    report_path.write_text(report, encoding="utf-8", newline="\n")
    h45.write_final_hash_manifest(OUTPUT)
    print(json.dumps({"STAGE_3_H4_5": gate["STAGE_3_H4_5"], "FIRST_BLOCKER": gate["FIRST_BLOCKER"], "READY_FOR_STAGE_3_H5": gate["READY_FOR_STAGE_3_H5"], "DERIVED_REPLAY": comparison["status"]}, sort_keys=True))
    return 0 if gate["STAGE_3_H4_5"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
