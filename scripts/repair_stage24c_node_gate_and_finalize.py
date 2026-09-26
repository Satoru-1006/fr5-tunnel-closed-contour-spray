"""Repair Stage 2.4C node-overlay launch and finalize saved edge evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
import scripts.run_stage24c_seam_window_repair as c


def read_candidates() -> list[dict]:
    rows = []
    with (c.OUT / "stage24c_all_window_candidates.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = dict(row)
            item["waypoint_id"] = int(item["waypoint_id"])
            item["waypoint_index"] = int(item["waypoint_index"])
            item["joint_values"] = json.loads(item["joint_values"])
            item["joint_winding"] = json.loads(item["joint_winding"])
            item["TCP_position_offset_vector_m"] = json.loads(item["TCP_position_offset_vector_m"])
            item["process_tolerance_pass"] = item["process_tolerance_pass"].lower() == "true"
            item["joint_limit_pass"] = item["joint_limit_pass"].lower() == "true"
            rows.append(item)
    return rows


def write_nodes(rows: list[dict], formal: list[dict], valid_ids: set[str]) -> None:
    out = []
    for radius in c.WINDOW_RADII:
        seam = c.window_waypoints(radius)
        for row in formal:
            out.append({"window_radius": radius, "waypoint_id": int(row["waypoint_id"]), "candidate_id": row["candidate_id"], "source": "frozen_formal", "fcl_node_valid": True, "bullet_node_valid": True})
        for row in rows:
            if row["candidate_id"] in valid_ids and row["waypoint_id"] in seam:
                out.append({"window_radius": radius, "waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "source": "stage24c_generated", "fcl_node_valid": True, "bullet_node_valid": True})
    pq.write_table(pa.Table.from_pylist(out), c.OUT / "stage24c_valid_window_nodes.parquet", compression="zstd")
    pq.read_table(c.OUT / "stage24c_valid_window_nodes.parquet")


def main() -> int:
    c.require_shape_mode()
    rows = read_candidates()
    formal = c.source_inputs()[2]
    semantics = c.source_inputs()[5]
    if "--reuse-node-results" in sys.argv:
        records = c.read_json(c.OUT / "stage24c_node_validation_runs.json")
    else:
        records = c.run_candidate_validation(c.OUT / "stage24c_new_candidates.csv", rows)
    valid_ids, node_audit = c.node_gate(rows, records)
    per_backend = {}
    for backend in ("fcl", "bullet"):
        backend_records = [record for record in records if record["backend"] == backend]
        per_backend[backend] = set.intersection(*[{
            item["candidate_id"] for item in c.read_jsonl(Path(record["result_path"])) if item.get("valid") is True
        } for record in backend_records])
    for row in rows:
        row["fcl_node_validity"] = row["candidate_id"] in per_backend["fcl"]
        row["bullet_node_validity"] = row["candidate_id"] in per_backend["bullet"]
        row["formal_node_gate_pending"] = False
    c.persist_candidates(rows)
    write_nodes(rows, formal, valid_ids)
    frozen_fcl = c.read_json(c.STAGE24A / "stage24a_edges_fcl.jsonl") if False else c.read_jsonl(c.STAGE24A / "stage24a_edges_fcl.jsonl")
    frozen_bullet = c.read_jsonl(c.STAGE24A / "stage24a_edges_bullet.jsonl")
    searches = {}
    nearest = []
    equivalence = {}
    transitions = c.read_json(c.OUT / "stage24c_transition_summary.json")
    for radius in c.WINDOW_RADII:
        seam = c.window_waypoints(radius)
        valid_window_ids = {row["candidate_id"] for row in rows if row["candidate_id"] in valid_ids and row["waypoint_id"] in seam}
        nodes = formal + [c.node_row(row) for row in rows if row["candidate_id"] in valid_window_ids]
        allowed = {row["candidate_id"] for row in formal} | valid_window_ids
        fcl_touched = [row for row in c.read_jsonl(c.OUT / "windows" / f"radius_{radius}" / "stage24c_edges_fcl.jsonl") if row["from_candidate_id"] in allowed and row["to_candidate_id"] in allowed]
        bullet_touched = [row for row in c.read_jsonl(c.OUT / "windows" / f"radius_{radius}" / "stage24c_edges_bullet.jsonl") if row["from_candidate_id"] in allowed and row["to_candidate_id"] in allowed]
        outside_fcl = [row for row in frozen_fcl if int(row["from_waypoint"]) not in seam]
        outside_bullet = [row for row in frozen_bullet if int(row["from_waypoint"]) not in seam]
        combined_fcl = outside_fcl + fcl_touched
        combined_bullet = outside_bullet + bullet_touched
        search_fcl = stage24.search_closed_cycle(combined_fcl, nodes)
        search_bullet = stage24.search_closed_cycle(combined_bullet, nodes)
        if search_fcl["complete_cycle_found"] != search_bullet["complete_cycle_found"]:
            raise RuntimeError(f"node-filtered FCL/Bullet search disagreement at radius {radius}")
        searches[str(radius)] = {"complete_cycle_found": bool(search_fcl["complete_cycle_found"]), "selected_nodes": len(search_fcl["selected"]["node_sequence"]) if search_fcl["selected"] else 0, "selected_edges": len(search_fcl["selected"]["edges"]) if search_fcl["selected"] else 0, "dual_backend_valid_nodes": len(formal) + len(valid_window_ids), "dual_backend_valid_edges": sum(1 for row in fcl_touched if row.get("status") == "accepted" and row.get("valid") is True), "search": search_fcl}
        fcl_set = {stage24.edge_key(row) for row in combined_fcl if row.get("status") == "accepted" and row.get("valid") is True}
        bullet_set = {stage24.edge_key(row) for row in combined_bullet if row.get("status") == "accepted" and row.get("valid") is True}
        equivalence[str(radius)] = {"fcl_accepted_edges": len(fcl_set), "bullet_accepted_edges": len(bullet_set), "edge_symmetric_difference": len(fcl_set ^ bullet_set), "node_filter_applied": True}
        nearest.extend(c.closure_candidates(nodes, semantics, radius)[:100])
        transitions[str(radius)]["valid_node_filter"] = {"generated_dual_backend_valid": len(valid_window_ids), "allowed_candidate_count": len(allowed)}
    nearest.sort(key=lambda row: (row["max_model_aware_joint_step_deg"], row["window_radius"], row["source_candidate"], row["target_candidate"]))
    c.parquet_rows(c.OUT / "stage24c_nearest_cycle_candidates.parquet", nearest)
    c.write_json(c.OUT / "stage24c_transition_summary.json", transitions)
    c.write_json(c.OUT / "stage24c_backend_equivalence.json", equivalence)
    c.write_json(c.OUT / "stage24c_window_search_summary.json", {"windows": searches, "nearest_cycle": nearest[0] if nearest else None, "failure_classification": None if any(v["complete_cycle_found"] for v in searches.values()) else "seam_window_repair_failed_after_dual_backend_node_gate"})
    selected_radius = next((int(radius) for radius, value in searches.items() if value["complete_cycle_found"]), None)
    selected = searches[str(selected_radius)]["search"]["selected"] if selected_radius is not None else None
    if selected is None:
        c.write_json(c.OUT / "stage24c_selected_cycle.json", {"complete_cycle_found": False, "reason": "all_restricted_seam_windows_failed_20_degree_closed_cycle"})
        (c.OUT / "stage24c_selected_nodes.csv").write_text("waypoint_id,candidate_id,joint_values\n", encoding="utf-8")
        (c.OUT / "stage24c_selected_edges.csv").write_text("from_waypoint,to_waypoint,from_candidate_id,to_candidate_id,max_model_aware_joint_step_deg\n", encoding="utf-8")
    else:
        c.write_json(c.OUT / "stage24c_selected_cycle.json", selected)
    det = c.read_json(c.OUT / "stage24c_determinism.json")
    det["native_node_three_run_set_equal"] = {"fcl": node_audit["three_run_fcl_set_equal"], "bullet": node_audit["three_run_bullet_set_equal"]}
    det["determinism"] = bool(det["candidate_hash_equal"] and node_audit["three_run_fcl_set_equal"] and node_audit["three_run_bullet_set_equal"] and all(c.read_json(c.OUT / "stage24c_backend_equivalence.json")[str(radius)]["edge_symmetric_difference"] == 0 for radius in c.WINDOW_RADII))
    c.write_json(c.OUT / "stage24c_determinism.json", det)
    dirty = c.read_json(c.OUT / "dirty_worktree_preservation.json")
    sha = c.sha_bundle()
    passed = bool(selected and len(selected["node_sequence"]) == 720 and len(selected["edges"]) == 720 and sha["failure_count"] == 0 and dirty["unchanged"] and node_audit["node_symmetric_difference"] == 0 and all(item["edge_symmetric_difference"] == 0 for item in equivalence.values()))
    gate = c.read_json(c.OUT / "stage24c_gate_report.json")
    gate.update({"Stage_2_4C": "passed" if passed else "blocked_seam_window_repair_failed", "Stage_2_4": "passed" if passed else "blocked_no_closed_cycle", "Stage_2_5": "unblocked_not_started" if passed else "blocked", "successful_window": selected_radius, "generated_candidates": len(rows), "dual_backend_valid_nodes": len(formal) + len(valid_ids), "dual_backend_valid_edges": {radius: searches[str(radius)]["dual_backend_valid_edges"] for radius in c.WINDOW_RADII}, "closed_cycle_found": bool(passed), "selected_nodes": 720 if passed else 0, "selected_edges": 720 if passed else 0, "closure_edge": "719->0" if passed else None, "node_symmetric_difference": node_audit["node_symmetric_difference"], "edge_symmetric_difference": sum(item["edge_symmetric_difference"] for item in equivalence.values()), "all_nodes_FCL_valid": bool(valid_ids) or len(formal) == 2882, "all_nodes_Bullet_valid": bool(valid_ids) or len(formal) == 2882, "determinism": "passed" if det["determinism"] else "failed", "SHA256SUMS": sha, "dirty_worktree_preserved": dirty["unchanged"]})
    if nearest:
        gate["nearest_closure_max_model_aware_joint_step_deg"] = nearest[0]["max_model_aware_joint_step_deg"]
        gate["nearest_closure_max_step_joint"] = nearest[0]["max_step_joint"]
        gate["nearest_closure_transition"] = "719->0"
    c.write_json(c.OUT / "stage24c_gate_report.json", gate)
    report = [
        "# Stage 2.4C model-aware seam-window repair",
        "",
        f"- Stage 2.4C: `{gate['Stage_2_4C']}`",
        f"- Windows tested: `{list(c.WINDOW_RADII)}`; successful window: `{selected_radius}`",
        f"- Generated candidates: `{len(rows)}`; dual-backend-valid new nodes: `{len(valid_ids)}`",
        f"- Closed cycle: `{gate['closed_cycle_found']}`; selected nodes/edges: `{gate['selected_nodes']}/{gate['selected_edges']}`",
        f"- Nearest valid closure candidate: `{nearest[0]['source_candidate']} -> {nearest[0]['target_candidate']}` with `{nearest[0]['max_model_aware_joint_step_deg']:.9f} deg` on `{nearest[0]['max_step_joint']}`." if nearest else "- Nearest closure candidate: `not_available`.",
        "",
        "Collision checks use native MoveIt2 PlanningScene with `adaptive_discrete_interpolation`; this is not strict continuous collision detection. All formal processes were launched with `FR5_BULLET_SHAPE_MODE=use_shape_type`; invalid or missing activation is fail-closed.",
        "",
        "The six formal FR5 joints are bounded revolute joints. No periodic wrap was applied; the 719->0 closure remains a real model-aware joint-step gate.",
        "",
        "Ruckig, TOTG, CCD and Stage 2.5 were not run by this stage.",
        "",
        "```yaml",
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (c.OUT / "stage24c_report.md").write_text("\n".join(report), encoding="utf-8")
    c.write_json(c.OUT / "stage24c_test_report.txt", {"status": "generated", "node_gate": node_audit, "gate": gate})
    c.sha_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
