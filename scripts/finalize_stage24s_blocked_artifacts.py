#!/usr/bin/env python3
"""Finalize a completed Stage 2.4S blocked search without rerunning it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage24s_segmented_process_rebaseline import aggregate_off_attempt_artifacts, sha256, write_json, write_sha256sums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--light", action="store_true", help="Use already flattened attempt JSONL; do not reread child trajectory JSON")
    args = parser.parse_args()
    root = args.root.resolve()
    attempts_path = root / "stage24s_off_planning_attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    first_failed = None
    if not args.light:
        first_failed = aggregate_off_attempt_artifacts(root, attempts)
    else:
        flat = root / "stage24s_retract_attempts.jsonl"
        if flat.exists():
            first = json.loads(next((line for line in flat.read_text(encoding="utf-8").splitlines() if line.strip()), "{}"))
            first_failed = {"from_segment": int(first.get("transition_id", -1)), "to_segment": int(first.get("transition_id", -1)) + 1, "segment_count": int(first.get("segment_count", -1)), "candidate_index": int(first.get("candidate_index", -1)), "endpoint_candidates_attempted": [f"{first.get('endpoint_candidate_a')}->{first.get('endpoint_candidate_b')}"], "retract_attempts": 1, "planner_attempts": 1 if first.get("failure_reason") == "planning_no_solution" else 0, "best_failure_reason": first.get("failure_reason", "not_recorded")}
    gate_path = root / "stage24s_gate_report.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["minimum_ON_segment_count_from_existing_graph"] = gate.get("minimum_graph_only_ON_segments")
    gate["OFF_transitions_required"] = int(gate.get("minimum_graph_only_ON_segments") or 0) - 1
    gate["OFF_transitions_passed"] = 0
    gate["best_executable_segment_count_found"] = None
    gate["first_failed_transition"] = first_failed or gate.get("first_failed_transition")
    gate["formal_graph_waypoints_covered"] = "720/720"
    gate["all_required_waypoints_covered"] = False
    gate["OFF_transition_search_budget"] = {"process_candidates_total": len(attempts), "process_candidates_by_segment_count": {str(k): sum(int(row.get("segment_count", -1)) == k for row in attempts) for k in sorted({int(row.get("segment_count", -1)) for row in attempts})}, "retract_distance_ladder_mm": [10, 20, 30, 40, 60, 80], "planner_attempts_per_endpoint_pair": 1, "infinite_retries": False}
    write_json(gate_path, gate)
    (root / "stage24s_report.md").write_text((root / "stage24s_report.md").read_text(encoding="utf-8") + "\nThe finite OFF search completed without a complete executable segmented process. Native FCL/Bullet OFF validation was not reached because no candidate had all MoveIt2 OFF transitions planned and PlanningScene-validated.\n", encoding="utf-8")
    write_sha256sums(root)
    print(json.dumps({"root": str(root), "status": gate.get("Stage_2_4S"), "first_failed_transition": first_failed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
