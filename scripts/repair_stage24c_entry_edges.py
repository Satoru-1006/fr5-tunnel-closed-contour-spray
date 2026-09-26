"""Add and formally validate the frozen-to-window entry transitions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

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
            item["joint_values"] = json.loads(item["joint_values"])
            item["joint_winding"] = json.loads(item["joint_winding"])
            item["TCP_position_offset_vector_m"] = json.loads(item["TCP_position_offset_vector_m"])
            rows.append(item)
    return rows


def main() -> int:
    c.require_shape_mode()
    rows = read_candidates()
    formal = c.source_inputs()[2]
    semantics = c.source_inputs()[5]
    node_records = c.read_json(c.OUT / "stage24c_node_validation_runs.json")
    per_backend = {}
    for backend in ("fcl", "bullet"):
        per_backend[backend] = set.intersection(*[{
            item["candidate_id"] for item in c.read_jsonl(Path(record["result_path"])) if item.get("valid") is True
        } for record in node_records if record["backend"] == backend])
    valid_ids = per_backend["fcl"] & per_backend["bullet"]
    build = c.read_json(c.OUT / "stage24c_edge_build_report.json")
    all_edges = pq.read_table(c.OUT / "stage24c_all_window_edges.parquet").to_pylist()
    transition_summary = c.read_json(c.OUT / "stage24c_transition_summary.json")
    backend_equiv = c.read_json(c.OUT / "stage24c_backend_equivalence.json")
    for radius in c.WINDOW_RADII:
        seam = c.window_waypoints(radius)
        entry_from = 720 - radius - 1
        entry_to = 720 - radius
        wrong_from = 719 - radius - 1
        all_edges = [row for row in all_edges if not (int(row.get("window_radius", -1)) == radius and int(row.get("from_waypoint", -1)) == wrong_from)]
        nodes = formal + [c.node_row(row) for row in rows if row["candidate_id"] in valid_ids and row["waypoint_id"] in seam]
        accepted, rejected, _ = c.build_transition_prefilter(nodes, entry_from, semantics)
        edge_csv = c.OUT / "windows" / f"radius_{radius}" / "stage24c_entry_edge_requests.csv"
        stage24.write_edge_requests(accepted, edge_csv)
        c.write_jsonl(edge_csv.with_name("stage24c_entry_joint_gate_rejected.jsonl"), rejected)
        records = [c.run_edge_native(radius, edge_csv, build, backend, run_index, subdir="entry_edge_runs") for backend in ("fcl", "bullet") for run_index in (1, 2, 3)]
        if any(record["exit_code"] != 0 for record in records):
            raise RuntimeError(f"entry edge validation failed for radius {radius}")
        manifests = {}
        for backend in ("fcl", "bullet"):
            selected = [record for record in records if record["backend"] == backend]
            assert len({record["result_sha256"] for record in selected}) == 1
            native = stage24.merge_native_edges(Path(selected[0]["result_path"]), backend)
            manifest = stage24.edge_manifest(accepted, rejected, native, backend)
            manifests[backend] = manifest
            c.write_jsonl(c.OUT / "windows" / f"radius_{radius}" / f"stage24c_entry_edges_{backend}.jsonl", manifest)
            existing_path = c.OUT / "windows" / f"radius_{radius}" / f"stage24c_edges_{backend}.jsonl"
            existing = [row for row in c.read_jsonl(existing_path) if int(row["from_waypoint"]) != 719 - radius - 1]
            merged = sorted(existing + manifest, key=stage24.edge_key)
            c.write_jsonl(existing_path, merged)
            all_edges.extend([{**row, "window_radius": radius} for row in manifest])
        fcl_set = {stage24.edge_key(row) for row in manifests["fcl"] if row.get("status") == "accepted" and row.get("valid") is True}
        bullet_set = {stage24.edge_key(row) for row in manifests["bullet"] if row.get("status") == "accepted" and row.get("valid") is True}
        backend_equiv[str(radius)]["entry_edge_fcl_accepted"] = len(fcl_set)
        backend_equiv[str(radius)]["entry_edge_bullet_accepted"] = len(bullet_set)
        backend_equiv[str(radius)]["entry_edge_symmetric_difference"] = len(fcl_set ^ bullet_set)
        transition_summary[str(radius)]["transitions"].pop(f"{719 - radius - 1}->{720 - radius - 1}", None)
        transition_summary[str(radius)]["transitions"].pop(f"{entry_from}->{entry_from}", None)
        transition_summary[str(radius)]["transitions"][f"{entry_from}->{entry_to}"] = {
            "fcl_accepted": sum(1 for row in manifests["fcl"] if row.get("status") == "accepted" and row.get("valid") is True),
            "bullet_accepted": sum(1 for row in manifests["bullet"] if row.get("status") == "accepted" and row.get("valid") is True),
            "minimum_model_aware_step_deg": min([float(row["max_model_aware_joint_step_deg"]) for row in accepted] or [None]),
        }
    c.parquet_rows(c.OUT / "stage24c_all_window_edges.parquet", all_edges)
    c.write_json(c.OUT / "stage24c_transition_summary.json", transition_summary)
    c.write_json(c.OUT / "stage24c_backend_equivalence.json", backend_equiv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
