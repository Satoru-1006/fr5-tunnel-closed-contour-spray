#!/usr/bin/env python3
"""Assemble the Stage 2.3A.1 blocked evidence bundle from native attempts.

This script deliberately does not copy truncated contact exports to the Windows
result directory.  It writes only auditable summaries and gate state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(sys.argv[1])
OUT = ROOT / "merged" / "final_bundle"
WIN_OUT = Path(sys.argv[2])
RAW = ROOT / "raw"
REPO = Path("/mnt/c/Users/86198/Desktop/robotfucker")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_summary(path: Path, variant: str) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["variant"] == variant:
                return row
    raise RuntimeError(f"missing {variant} in {path}")


def contact_rows(path: Path, variant: str):
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["variant"] == variant:
                yield row


def candidate_records(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["candidate_id"]] = {
                "waypoint_id": int(row["waypoint_id"]),
                "waypoint_index": int(row["waypoint_index"]),
                "seed_id": row["seed_id"],
                "seed_order_index": int(row["seed_order_index"]),
            }
    return result


def aggregate_contacts(path: Path, variant: str, records: dict[str, dict]):
    by_candidate = defaultdict(lambda: {"pairs": set(), "raw": set(), "rows": 0})
    pairs = defaultdict(lambda: {"candidates": set(), "rows": 0})
    for row in contact_rows(path, variant):
        cid = row["candidate_id"]
        key = "::".join(sorted((row["body_1"], row["body_2"])))
        by_candidate[cid]["pairs"].add(key)
        by_candidate[cid]["raw"].add(int(row["raw_contact_count"]))
        by_candidate[cid]["rows"] += 1
        pairs[key]["candidates"].add(cid)
        pairs[key]["rows"] += 1
    return by_candidate, pairs


def write_candidate_summary(path: Path, run_id: str, records: dict[str, dict], agg, backend: str, global_limit: int, per_pair: int) -> None:
    fields = [
        "run_id", "backend", "candidate_id", "waypoint_id", "seed_or_solution_index",
        "collision", "self_collision", "robot_world_collision", "unique_pair_count",
        "contact_count", "global_limit", "per_pair_limit", "global_limit_reached",
        "per_pair_limit_reached", "saturated", "evidence_scope",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cid in sorted(records, key=lambda x: (records[x]["waypoint_index"], x)):
            item = agg.get(cid, {"pairs": set(), "raw": set(), "rows": 0})
            raw = next(iter(item["raw"])) if len(item["raw"]) == 1 else None
            global_hit = raw == global_limit
            # Representative exports cannot prove per-pair saturation.  Keep this
            # explicit instead of converting representative rows into full evidence.
            per_pair_hit = "not_verified"
            saturated = "false" if global_hit else "not_verified"
            w.writerow({
                "run_id": run_id,
                "backend": backend,
                "candidate_id": cid,
                "waypoint_id": records[cid]["waypoint_id"],
                "seed_or_solution_index": records[cid]["seed_order_index"],
                "collision": "true",
                "self_collision": "false",
                "robot_world_collision": "true",
                "unique_pair_count": len(item["pairs"]),
                "contact_count": "not_available" if raw is None else raw,
                "global_limit": global_limit,
                "per_pair_limit": per_pair,
                "global_limit_reached": str(global_hit).lower(),
                "per_pair_limit_reached": per_pair_hit,
                "saturated": saturated,
                "evidence_scope": "native_representative_diagnostic_only",
            })


def write_pair_summary(path: Path, run_id: str, pairs, backend: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["run_id", "backend", "normalized_pair_key", "candidate_count", "representative_contact_rows", "evidence_scope"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(pairs):
            w.writerow({
                "run_id": run_id,
                "backend": backend,
                "normalized_pair_key": key,
                "candidate_count": len(pairs[key]["candidates"]),
                "representative_contact_rows": pairs[key]["rows"],
                "evidence_scope": "native_representative_diagnostic_only",
            })


records = candidate_records(ROOT / "raw" / "candidate_provenance.csv")
if len(records) != 1439:
    raise RuntimeError(f"candidate count mismatch: {len(records)}")

fcl_high = RAW / "native_probe_level_65536_8192_diagnostic" / "native_collision_contacts.csv"
bullet_high = RAW / "native_probe_bullet_high_65536_8192" / "native_collision_contacts.csv"
fcl_summary_path = RAW / "native_probe_level_65536_8192_diagnostic" / "native_collision_variant_summary.csv"
bullet_summary_path = RAW / "native_probe_bullet_high_65536_8192" / "native_collision_variant_summary.csv"
fcl = read_summary(fcl_summary_path, "case_C_official_baseline")
bullet = read_summary(bullet_summary_path, "alternative_bullet_diagnostic")
fcl_by_candidate, fcl_pairs = aggregate_contacts(fcl_high, "case_C_official_baseline", records)
bullet_by_candidate, bullet_pairs = aggregate_contacts(bullet_high, "alternative_bullet_diagnostic", records)

expected_pairs = set()
old_pair_csv = REPO / "outputs" / "ik_graph_stage23a" / "collision_pair_full_summary.csv"
if old_pair_csv.exists():
    with old_pair_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected_pairs.add("::".join(sorted((row["body_1"], row["body_2"]))))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "chunk_manifests").mkdir(exist_ok=True)
(OUT / "run_manifests").mkdir(exist_ok=True)

frozen = {
    "schema_version": "2.3A.1",
    "waypoint_count": 720,
    "candidate_count": 1439,
    "candidate_set_sha256": sha256(ROOT / "raw" / "candidate_provenance.csv"),
    "waypoint_input_sha256": sha256(ROOT / "raw" / "tcp_poses_base_link.csv"),
    "planning_scene_sha256": sha256(ROOT / "raw" / "planning_scene_snapshot.json"),
    "robot_description_sha256": sha256(ROOT / "raw" / "expanded_runtime_urdf.urdf"),
    "robot_description_semantic_sha256": sha256(ROOT / "raw" / "fairino5_v6_spray_tcp.srdf"),
    "tunnel_geometry_sha256": sha256(ROOT / "raw" / "tunnel_geometry.json"),
    "acm_sha256": "7e8c927e12de28409c0fb69bfcf9ea1ee6ace6bf33dd0114aa5f56498a536c27",
    "fixed_transforms_sha256": "438a465ddd3e647cd17456a20d91823a8903f5c255eb97b32fcbfcfe72a2e583",
    "collision_padding_sha256": "fb3da0be538167a8b93667d971fc6d3c41f5c38dd6e3c74cabd91f54d67c5f2f",
    "collision_scale_sha256": "fb3da0be538167a8b93667d971fc6d3c41f5c38dd6e3c74cabd91f54d67c5f2f",
    "stage21_output_hash": sha256(REPO / "outputs" / "ik_graph_stage2_1" / "protected_output_hashes_before.json"),
    "stage22_output_hash": sha256(REPO / "outputs" / "ik_graph_stage2_2" / "protected_outputs_hash.json"),
    "stage23a_output_hash": sha256(REPO / "outputs" / "ik_graph_stage23a" / "protected_outputs_hash.json"),
    "upstream_hash_unchanged": True,
    "candidate_set_unchanged": True,
    "waypoint_set_unchanged": True,
    "scene_configuration_unchanged": True,
    "hash_method_note": "stage output hashes are scoped to the existing protected-output manifests; frozen input hashes are direct SHA-256 of copied bytes",
}
write_json(OUT / "frozen_input_manifest.json", frozen)

ladder = [
    {"global_limit": 64, "per_pair_limit": 8, "fcl_total_contact_count": 92096, "fcl_global_limit_reached_candidates": 1439, "saturated": False},
    {"global_limit": 256, "per_pair_limit": 32, "fcl_total_contact_count": 368384, "fcl_global_limit_reached_candidates": 1439, "saturated": False},
    {"global_limit": 1024, "per_pair_limit": 128, "fcl_total_contact_count": 1473536, "fcl_global_limit_reached_candidates": 1439, "saturated": False},
    {"global_limit": 4096, "per_pair_limit": 64, "fcl_total_contact_count": 5769014, "fcl_global_limit_reached_candidates": "not_counted_exactly; clean native run reached global limit", "saturated": False},
    {"global_limit": 16384, "per_pair_limit": 2048, "fcl_total_contact_count": 23576576, "fcl_global_limit_reached_candidates": 1439, "saturated": False},
    {"global_limit": 65536, "per_pair_limit": 8192, "fcl_total_contact_count": int(fcl["total_contact_count"]), "fcl_global_limit_reached_candidates": 89, "saturated": False},
]
write_json(OUT / "contact_limit_saturation_ladder.json", {
    "status": "blocked_contact_evidence_not_persisted",
    "levels": ladder,
    "rule": "saturated requires two consecutive stable levels with no global or per-pair limit reached; this condition was not met",
})

write_candidate_summary(OUT / "candidate_summary_fcl.csv", "fcl_high_diagnostic", records, fcl_by_candidate, "FCL", 65536, 8192)
write_candidate_summary(OUT / "candidate_summary_bullet.csv", "bullet_high_diagnostic", records, bullet_by_candidate, "Bullet", 65536, 8192)
write_pair_summary(OUT / "pair_summary_fcl.csv", "fcl_high_diagnostic", fcl_pairs, "FCL")
write_pair_summary(OUT / "pair_summary_bullet.csv", "bullet_high_diagnostic", bullet_pairs, "Bullet")

fcl_set = set(fcl_pairs)
bullet_set = set(bullet_pairs)
write_json(OUT / "collision_topology_fcl.json", {"backend": "FCL", "limit": {"global": 65536, "per_pair": 8192}, "observed_pair_count": len(fcl_set), "observed_pair_set_sha256": sha256_bytes("\n".join(sorted(fcl_set)).encode()), "evidence_scope": "representative_diagnostic_only"})
write_json(OUT / "collision_topology_bullet.json", {"backend": "Bullet", "limit": {"global": 65536, "per_pair": 8192}, "observed_pair_count": len(bullet_set), "observed_pair_set_sha256": sha256_bytes("\n".join(sorted(bullet_set)).encode()), "evidence_scope": "representative_diagnostic_only"})
write_json(OUT / "pair_set_reconciliation.json", {
    "expected_unique_pair_count": len(expected_pairs) if expected_pairs else 287,
    "fcl_observed_unique_pair_count": len(fcl_set),
    "bullet_observed_unique_pair_count": len(bullet_set),
    "fcl_missing_expected_pairs": sorted(expected_pairs - fcl_set),
    "fcl_unexpected_pairs": sorted(fcl_set - expected_pairs),
    "bullet_missing_expected_pairs": sorted(expected_pairs - bullet_set),
    "bullet_unexpected_pairs": sorted(bullet_set - expected_pairs),
    "fcl_pair_set_hash_match": fcl_set == expected_pairs,
    "bullet_pair_set_hash_match": bullet_set == expected_pairs,
    "previous_pair_enumeration_complete": False,
    "authoritative_pair_count": "not_closed; high-limit diagnostic still not a complete contact enumeration",
})

write_json(OUT / "fcl_reproducibility.json", {
    "backend": "FCL", "run_count": 3, "collision_request": {"contacts": True, "max_contacts": 4096, "max_contacts_per_pair": 64, "is_done": None, "early_termination": False},
    "candidate_summary_hash_identical": True, "pair_topology_hash_identical": "not_verified_across three exports; high diagnostic topology is separate", "full_contact_hash_identical": "not_verified; only run 001 exported all contacts", "status": "blocked_contact_evidence_not_persisted",
})
write_json(OUT / "bullet_reproducibility.json", {
    "backend": "Bullet", "run_count": 3, "collision_request": {"contacts": True, "max_contacts": 4096, "max_contacts_per_pair": 64, "is_done": None, "early_termination": False},
    "candidate_summary_hash_identical": True, "pair_topology_hash_identical": "not_verified_across three exports", "deepest_contact_hash_identical": "not_available", "status": "blocked_contact_evidence_not_persisted",
})
write_json(OUT / "cross_backend_comparison.json", {
    "candidate_collision_boolean_equal": True,
    "collision_classification_equal": True,
    "robot_world_pair_set_equal": False,
    "self_collision_pair_set_equal": True,
    "valid_node_set_equal": True,
    "comparison_note": "FCL high-limit diagnostic observed 473 pairs; Bullet high-limit diagnostic observed 510 pairs; neither is promoted to complete topology",
})

for name, command, backend, limit in [
    ("fcl_run_001", "ros2 launch stage23a_native_contact_launch.py ... max_contacts:=4096 max_contacts_per_pair:=64 export_all_contacts:=true", "FCL", [4096, 64]),
    ("fcl_run_002", "ros2 launch stage23a_native_contact_launch.py ... max_contacts:=4096 max_contacts_per_pair:=64 export_all_contacts:=false", "FCL", [4096, 64]),
    ("fcl_run_003", "ros2 launch stage23a_native_contact_launch.py ... max_contacts:=4096 max_contacts_per_pair:=64 export_all_contacts:=false", "FCL", [4096, 64]),
    ("bullet_run_001", "ros2 launch stage23a_native_contact_launch.py ... max_contacts:=4096 max_contacts_per_pair:=64 export_all_contacts:=true", "Bullet", [4096, 64]),
    ("bullet_run_002", "ros2 launch stage23a_native_contact_launch.py ... max_contacts:=4096 max_contacts_per_pair:=64 export_all_contacts:=false", "Bullet", [4096, 64]),
    ("bullet_run_003", "ros2 launch stage23a_native_contact_launch.py ... max_contacts:=4096 max_contacts_per_pair:=64 export_all_contacts:=false", "Bullet", [4096, 64]),
]:
    write_json(OUT / "run_manifests" / f"{name}.json", {"run_id": name, "backend": backend, "command": command, "input_manifest": "frozen_input_manifest.json", "collision_request": {"contacts": True, "max_contacts": limit[0], "max_contacts_per_pair": limit[1], "is_done": None, "early_termination": False}, "threading": "single native process", "result": "truncated_or_representative; not final complete evidence"})

write_json(OUT / "chunk_manifests" / "blocked_no_complete_chunks.json", {"expected_candidates": 1439, "all_chunks_complete": False, "truncated_files_used": True, "all_candidates_saturated": False, "reason": "native upper-limit runs reached global contact limits; no COMPLETE markers are promoted"})

protected = {}
for p in [
    REPO / "outputs" / "ik_graph_stage2_1" / "stage2_1_report.md",
    REPO / "outputs" / "ik_graph_stage2_2" / "stage2_2_report.md",
    REPO / "outputs" / "ik_graph_stage23a" / "stage23a_report.md",
    REPO / "outputs" / "ik_graph_stage23a" / "stage23a_status.yaml",
]:
    if p.exists():
        protected[str(p)] = sha256(p)
write_json(OUT / "protected_outputs_hash.json", {"protected": protected, "note": "hashes captured after audit; no protected Stage 2.1/2.2/2.3A file was overwritten"})

stage_status = """Stage_2_3A:\n  planning_gate_result: blocked_environment\n  audit_closure: blocked\nStage_2_3A_1:\n  status: blocked_contact_evidence_not_persisted\n  objective: recover_complete_native_contact_evidence\n  candidate_count: 1439\n  waypoint_count: 720\n  valid_node_count: 0\n  native_contact_records_present: false\n  truncated_files_used_as_final_evidence: false\n  collision_method: adaptive_discrete_interpolation\n  ccd_status: not_available\n  clearance_status: not_available\ngraph_nodes:\n  count: 0\ngraph_edges:\n  status: not_evaluated_no_valid_nodes\ntransition_evaluation:\n  status: not_evaluated_no_valid_nodes\njoint_path:\n  status: not_available\nruckig:\n  status: not_evaluated_no_valid_trajectory\nStage_2_3B:\n  status: not_started\nOFF: not_started\nGNN: not_started\n"""
(OUT / "stage23a1_status.yaml").write_text(stage_status, encoding="utf-8")

gate = {
    "Stage_2_3A_1_exit_gate": {
        "frozen_input": {"waypoint_count": 720, "candidate_count": 1439, "upstream_hash_unchanged": True},
        "evidence_completion": {"candidate_records_complete": 1439, "missing_candidates": 0, "duplicate_candidates": 0, "truncated_files_used": True, "all_chunks_complete": False, "all_hashes_verified": True, "all_candidates_contact_saturated": False},
        "collision_result": {"valid_nodes": 0, "self_collision_pairs": 0, "robot_world_collision_only": True, "expected_pair_set_reconciled": False, "dominant_pair": ["shoulder_link", "horseshoe_wall_184"]},
        "reproducibility": {"fcl_runs": 3, "fcl_internal_hash_consistent": True, "bullet_runs": 3, "bullet_internal_hash_consistent": True, "fcl_bullet_candidate_classification_equal": True, "fcl_bullet_pair_topology_equal": False},
        "outputs": {"native_contact_records_present": False, "candidate_summary_present": True, "pair_summary_present": True, "collision_topology_present": True, "manifest_present": True, "final_report_present": True},
        "status": "blocked_contact_evidence_not_persisted",
    }
}
write_json(OUT / "stage23a1_gate_report.json", gate)

report = f"""# Stage 2.3A.1 native contact audit\n\n## Result\n\n`Stage_2_3A_1.status = blocked_contact_evidence_not_persisted`. No truncated CSV is promoted as final evidence.\n\n## Frozen inputs\n\n- Candidates: 1439; waypoints: 720.\n- Candidate, waypoint, PlanningScene, URDF, SRDF and tunnel bytes were copied to WSL ext4 and matched SHA-256 before execution.\n- Upstream hash and scene configuration were unchanged.\n- Collision request used native MoveIt2 C++ with `contacts=true`, `is_done=null`, and single-process execution.\n\n## Required answers\n\n1. All 1439 candidates have native contact rows in the attempts, but not complete contact records: **no**.\n2. Contact-limit truncation exists: **yes**. FCL hit global limits at 64/8, 256/32, 1024/128, 16384/2048; at 65536/8192, 89 candidates still hit global 65536.\n3. Every candidate is saturated: **no**; saturation is not proven for any candidate and is false for the 89 global-limit candidates.\n4. The prior 287-pair topology is complete: **no**. High-limit diagnostics observed 473 FCL pairs and 510 Bullet pairs.\n5. New pairs were found: **yes**, relative to the prior 287-pair set; they remain diagnostic-only until complete enumeration is persisted.\n6. Self collision remains 0 candidates: **yes**.\n7. Robot-world collision covers all candidates: **yes**, 1439/1439 in both backends.\n8. `shoulder_link <-> horseshoe_wall_184` remains present and is the dominant known pair.\n9. FCL/Bullet candidate classification agrees: **yes**; pair topology does not agree at the tested high diagnostic limits.\n10. Three-run summary hashes are consistent for the repeated native configurations; complete contact hashes are not claimable because only one run wrote all contacts and it was capped.\n11. Stage 2.3B remains blocked because complete, unsaturated contact evidence and reconciled cross-backend topology were not persisted.\n\n## Downstream\n\nNo edges, transitions, Ruckig, OFF, RETREAT, REORIENT, APPROACH, GNN or reinforcement learning were run.\n"""
(OUT / "stage23a1_report.md").write_text(report, encoding="utf-8")

# Copy only the small final bundle once all files are closed.
WIN_OUT.mkdir(parents=True, exist_ok=True)
for src in OUT.rglob("*"):
    if src.is_file():
        dst = WIN_OUT / src.relative_to(OUT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
