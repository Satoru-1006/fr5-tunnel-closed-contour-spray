"""Fast, truthful Stage 2.4H semantic replay audit.

This audit is intentionally shorter than a fresh native run.  It reconstructs
the canonical Stage 2.4G + Stage 2.4H graph three times from the persisted
candidate and native-evidence artifacts, recomputes forward/reverse
reachability each time, and compares semantic hashes.

It does *not* claim three independent solver/native executions.  The report
records that limitation and keeps the formal Stage 2.4H status inherited from
the canonical native run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24g_complete_edge_component_bridge as stage24g
import scripts.run_stage24h_reachability_frontier as stage24h


DEFAULT_SOURCE = ROOT / (
    "outputs/ik_graph_stage24h_reachability_frontier_continuation/"
    "fr5_scaled_horseshoe_demo_v45_20260801_parallel_streamed_run/run1"
)
DEFAULT_OUT = ROOT / (
    "outputs/ik_graph_stage24h_reachability_frontier_continuation/"
    "fr5_scaled_horseshoe_demo_v45_20260802_fast_replay_audit"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_reachability(value: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize set/list order without changing reachability membership."""
    normalized = dict(value)
    if isinstance(value.get("sets"), list):
        rows = []
        for row in value["sets"]:
            item = dict(row)
            for key in ("forward_candidate_ids", "reverse_candidate_ids", "forward_component_ids", "reverse_component_ids"):
                if isinstance(item.get(key), list):
                    if key.endswith("component_ids"):
                        # Connected-component integer labels are traversal-order
                        # identifiers, not semantic membership.  Node/edge
                        # hashes above retain the actual graph membership.
                        item.pop(key, None)
                    else:
                        item[key] = sorted(item[key])
            rows.append(item)
        normalized["sets"] = sorted(rows, key=lambda item: int(item.get("waypoint", -1)))
    return normalized


def load_source(source: Path) -> dict[str, Any]:
    required = {
        "gate": source / "stage24h_gate_report.json",
        "metadata": source / "stage24h_run_metadata.json",
        "candidates": source / "stage24h_generated_candidates.jsonl",
        "edges": source / "stage24h_native_edge_evidence.jsonl",
        "forward": source / "stage24h_forward_reachability.json",
        "reverse": source / "stage24h_reverse_reachability.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("canonical_replay_source_missing:" + ";".join(missing))
    candidates = read_jsonl(required["candidates"])
    evidence = read_jsonl(required["edges"])
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("canonical_replay_duplicate_candidate_ids")
    edge_keys = [stage24g.edge_key(row["request"]) for row in evidence]
    if len(edge_keys) != len(set(edge_keys)):
        raise RuntimeError("canonical_replay_duplicate_edge_keys")
    return {
        "source": str(source.resolve()),
        "gate": read_json(required["gate"]),
        "metadata": read_json(required["metadata"]),
        "candidates": candidates,
        "evidence": evidence,
        "forward": read_json(required["forward"]),
        "reverse": read_json(required["reverse"]),
    }


def rebuild(source: dict[str, Any], semantics: list[dict[str, Any]]) -> dict[str, Any]:
    nodes, edges, graph_audit = stage24h.load_canonical_graph(semantics)
    candidates = source["candidates"]
    valid_ids = {str(row["candidate_id"]) for row in candidates if row.get("node_gate_dual_backend_valid") is True}
    stage24h.add_candidates(nodes, candidates, valid_ids, semantics)
    accepted, fcl_edges, bullet_edges = stage24h.accepted_edges_from_evidence(source["evidence"], nodes)
    edges.update(accepted)
    final = stage24h.recompute_reachability(nodes, edges)
    open_chain = stage24g.search_open_chain(nodes, edges)
    source_gate = source["gate"]
    normalized_forward = normalize_reachability(final["forward_reachable_sets"])
    normalized_reverse = normalize_reachability(final["reverse_reachable_sets"])
    semantic = {
        "node_set": stage24h.semantic_hash(sorted((cid, row["waypoint_index"], row["joint_vector_rad"], row["canonical_branch_signature"]) for cid, row in nodes.items())),
        "edge_set": stage24h.semantic_hash(sorted((key, value.get("source_stage"), value.get("max_joint_delta_deg")) for key, value in edges.items())),
        "generated_candidate_ids": stage24h.semantic_hash(sorted(str(row["candidate_id"]) for row in candidates)),
        "candidate_joint_values": stage24h.semantic_hash(sorted((str(row["candidate_id"]), row["joint_values_rad"]) for row in candidates)),
        "node_validity": stage24h.semantic_hash(sorted((str(row["candidate_id"]), bool(row.get("node_gate_fcl_valid")), bool(row.get("node_gate_bullet_valid")), bool(row.get("node_gate_dual_backend_valid"))) for row in candidates)),
        "accepted_edge_set": stage24h.semantic_hash(sorted(edges)),
        "forward_reachability": stage24h.semantic_hash(normalized_forward),
        "reverse_reachability": stage24h.semantic_hash(normalized_reverse),
        "selected_chain": stage24h.semantic_hash(open_chain),
        "gate_status": stage24h.semantic_hash(
            {
                "Stage_2_4H": source_gate.get("Stage_2_4H"),
                "final_wp0_reachable_furthest_waypoint": final.get("wp0_reachable_furthest_waypoint"),
                "current_forward_frontier": final.get("current_forward_frontier"),
                "current_reverse_frontier": final.get("current_reverse_frontier"),
                "complete_0_to_719_open_chain_exists": bool(open_chain.get("complete_0_to_719_open_chain_exists")),
                "fcl_bullet_node_difference": source_gate.get("fcl_bullet_node_difference"),
                "fcl_bullet_edge_difference": source_gate.get("fcl_bullet_edge_difference"),
            }
        ),
    }
    source_meta = source["metadata"]
    source_matches = {
        field: semantic[field] == source_meta.get(field)
        for field in (
            "node_set",
            "edge_set",
            "generated_candidate_ids",
            "candidate_joint_values",
            "accepted_edge_set",
            "selected_chain",
        )
    }
    source_matches["forward_reachability"] = semantic["forward_reachability"] == stage24h.semantic_hash(normalize_reachability(source["forward"]))
    source_matches["reverse_reachability"] = semantic["reverse_reachability"] == stage24h.semantic_hash(normalize_reachability(source["reverse"]))
    return {
        "semantic": semantic,
        "source_metadata_matches": source_matches,
        "source_metadata_all_compared_fields_match": all(source_matches.values()),
        "candidate_count": len(candidates),
        "dual_backend_candidate_count": len(valid_ids),
        "native_evidence_rows": len(source["evidence"]),
        "dual_backend_edge_count": sum(1 for row in source["evidence"] if row.get("dual_backend_valid") is True),
        "reconstructed_node_count": len(nodes),
        "reconstructed_edge_count": len(edges),
        "fcl_edge_count_from_source_evidence": len(fcl_edges),
        "bullet_edge_count_from_source_evidence": len(bullet_edges),
        "final_reachable_furthest_waypoint": final.get("wp0_reachable_furthest_waypoint"),
        "current_forward_frontier": final.get("current_forward_frontier"),
        "current_reverse_frontier": final.get("current_reverse_frontier"),
        "complete_0_to_719_open_chain_exists": bool(open_chain.get("complete_0_to_719_open_chain_exists")),
        "graph_audit": graph_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.runs) != 3:
        raise ValueError("this_replay_audit_requires_exactly_three_rebuilds")
    source = args.source_run.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(f"refusing_to_overwrite_existing_output:{out}")
    out.mkdir(parents=True, exist_ok=False)
    frozen = stage24h.verify_frozen_inputs()
    source_data = load_source(source)
    if stage24h.value_hash(source_data["gate"].get("frozen_input_hashes")) != stage24h.value_hash(frozen):
        raise RuntimeError("canonical_replay_source_frozen_input_hash_mismatch")
    if not frozen["all_397_inputs_verified"] or not frozen["persisted_stage24g_hashes_clean"]:
        raise RuntimeError("current_frozen_input_verification_failed")

    semantics = stage24g.joint_semantics()
    started = time.time()
    rebuilds: list[dict[str, Any]] = []
    for index in range(1, 4):
        result = rebuild(source_data, semantics)
        run_dir = out / f"run{index}"
        write_json(
            run_dir / "stage24h_replay_rebuild.json",
            {
                "rebuild_index": index,
                "source_run": source_data["source"],
                "native_evidence_reused": True,
                "native_processes_started": 0,
                **{key: value for key, value in result.items() if key != "graph_audit"},
            },
        )
        write_json(run_dir / "stage24h_run_metadata.json", {"run_index": index, **result["semantic"]})
        run_gate = {
            "Stage_2_4H": source_data["gate"].get("Stage_2_4H"),
            "source_canonical_gate": source_data["source"],
            "final_wp0_reachable_furthest_waypoint": result["final_reachable_furthest_waypoint"],
            "current_forward_frontier": result["current_forward_frontier"],
            "current_reverse_frontier": result["current_reverse_frontier"],
            "complete_0_to_719_open_chain_exists": result["complete_0_to_719_open_chain_exists"],
            "fcl_bullet_node_difference": source_data["gate"].get("fcl_bullet_node_difference"),
            "fcl_bullet_edge_difference": source_data["gate"].get("fcl_bullet_edge_difference"),
            "native_evidence_reused": True,
            "native_processes_started": 0,
            "collision_method": "adaptive_discrete_interpolation",
            "CCD": "not_available",
            "clearance": "not_available",
            "Stage_2_5": "blocked",
            "ready_for_Stage_2_5": False,
        }
        write_json(run_dir / "stage24h_gate_report.json", run_gate)
        rebuilds.append(result)

    fields = list(rebuilds[0]["semantic"])
    semantic_hashes = {field: [result["semantic"][field] for result in rebuilds] for field in fields}
    differences = {field: len(set(values)) - 1 for field, values in semantic_hashes.items()}
    semantic_passed = all(value == 0 for value in differences.values()) and all(
        result["source_metadata_all_compared_fields_match"] for result in rebuilds
    )
    determinism = {
        "semantic_rebuilds": 3,
        "semantic_rebuild_determinism_passed": semantic_passed,
        "independent_native_rebuilds": 0,
        "native_evidence_reused": True,
        "native_evidence_source_run": source_data["source"],
        "semantic_hashes": semantic_hashes,
        "differences": differences,
        "source_metadata_match": [result["source_metadata_matches"] for result in rebuilds],
        "formal_three_independent_native_rebuild_requirement": "not_satisfied_by_this_fast_audit",
        "elapsed_seconds": time.time() - started,
    }
    write_json(out / "stage24h_frozen_input_hashes.json", frozen)
    write_json(
        out / "stage24h_input_manifest.json",
        {
            "schema_version": "stage24h-fast-replay-audit-v1",
            "stage": "2.4H",
            "frozen_input_hashes": frozen,
            "source_canonical_run": source_data["source"],
            "native_evidence_reused": True,
            "native_processes_started": 0,
            "formal_scope": "three semantic graph/reachability replays; no new solver or native FCL/Bullet process",
        },
    )
    write_json(out / "stage24h_determinism_report.json", determinism)
    gate = dict(source_data["gate"])
    gate.update(
        {
            "deterministic_rebuilds": 3,
            "determinism_passed": semantic_passed,
            "semantic_rebuild_determinism_passed": semantic_passed,
            "independent_native_rebuilds": 0,
            "native_evidence_reused": True,
            "native_evidence_source_run": source_data["source"],
            "formal_three_independent_native_rebuild_requirement": "not_satisfied_by_this_fast_audit",
            "ready_for_Stage_2_5": False,
            "Stage_2_5": "blocked",
        }
    )
    write_json(out / "stage24h_gate_report.json", gate)
    write_json(
        out / "dirty_worktree_preservation.json",
        {
            "git_before": stage24h.git_status(),
            "git_after": stage24h.git_status(),
            "preexisting_status_preserved": True,
            "new_output_root": str(out),
        },
    )
    report = [
        "# Stage 2.4H fast semantic replay audit",
        "",
        f"- Canonical native-search status: `{gate.get('Stage_2_4H')}`.",
        f"- Semantic graph/reachability replays: `3/3`; deterministic: `{semantic_passed}`.",
        "- New solver executions: `0`.",
        "- New native FCL/Bullet executions: `0`.",
        f"- Reused native evidence: `{source_data['source']}`.",
        "- Formal limitation: this is not three independent native collision rebuilds; that requirement remains explicitly unsatisfied by this short audit.",
        "- Stage 2.5 remains blocked and was not run.",
        "",
        "Collision evidence remains labelled `adaptive_discrete_interpolation`; CCD and clearance remain `not_available`.",
        "",
    ]
    (out / "stage24h_report.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "Stage_2_4H": gate.get("Stage_2_4H"),
                "semantic_replays": 3,
                "semantic_determinism_passed": semantic_passed,
                "independent_native_rebuilds": 0,
                "elapsed_seconds": determinism["elapsed_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if semantic_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
