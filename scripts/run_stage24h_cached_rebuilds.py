"""Fast Stage 2.4H rebuild audit using one immutable native-evidence cache.

The expensive part of the existing Stage 2.4H runner is the repeated native
FCL/Bullet PlanningScene launch for every frontier.  This runner keeps the
frontier loop, seeded IK generation, graph assembly, reachability recompute,
and three semantic rebuilds independent, but reuses an already completed
Stage 2.4H run's native node/edge evidence by exact candidate/edge identity.

It is deliberately explicit about the scope: this is three independent
semantic graph rebuilds backed by one previously executed native validation
run, not three new native FCL/Bullet processes.  The output therefore cannot
be misread as three independent native collision runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24g_complete_edge_component_bridge as stage24g
import scripts.run_stage24h_reachability_frontier as stage24h


DEFAULT_CANONICAL_RUN = ROOT / (
    "outputs/ik_graph_stage24h_reachability_frontier_continuation/"
    "fr5_scaled_horseshoe_demo_v45_20260801_parallel_streamed_run/run1"
)
DEFAULT_OUT = ROOT / (
    "outputs/ik_graph_stage24h_reachability_frontier_continuation/"
    "fr5_scaled_horseshoe_demo_v45_20260802_cached_three_rebuilds"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cache_paths(canonical_run: Path) -> dict[str, Path]:
    return {
        "gate": canonical_run / "stage24h_gate_report.json",
        "metadata": canonical_run / "stage24h_run_metadata.json",
        "candidates": canonical_run / "stage24h_generated_candidates.jsonl",
        "edges": canonical_run / "stage24h_native_edge_evidence.jsonl",
        "forward": canonical_run / "stage24h_forward_reachability.json",
        "reverse": canonical_run / "stage24h_reverse_reachability.json",
    }


def _load_cache(canonical_run: Path) -> dict[str, Any]:
    paths = _cache_paths(canonical_run)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("canonical_native_cache_missing:" + ";".join(missing))
    candidates = _read_jsonl(paths["candidates"])
    evidence = _read_jsonl(paths["edges"])
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        if candidate_id in candidate_by_id:
            raise RuntimeError(f"duplicate_cached_candidate_id:{candidate_id}")
        candidate_by_id[candidate_id] = row
    edge_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in evidence:
        key = stage24g.edge_key(row["request"])
        if key in edge_by_key:
            raise RuntimeError(f"duplicate_cached_edge_key:{key}")
        edge_by_key[key] = row
    gate = _read_json(paths["gate"])
    metadata = _read_json(paths["metadata"])
    return {
        "canonical_run": str(canonical_run.resolve()),
        "paths": paths,
        "gate": gate,
        "metadata": metadata,
        "candidates": candidates,
        "candidate_by_id": candidate_by_id,
        "evidence": evidence,
        "edge_by_key": edge_by_key,
        "forward": _read_json(paths["forward"]),
        "reverse": _read_json(paths["reverse"]),
    }


def make_cached_node_gate(cache: dict[str, Any]):
    def cached_node_gate(out: Path, candidates: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
        valid_ids: set[str] = set()
        fcl_ids: set[str] = set()
        bullet_ids: set[str] = set()
        cache_misses: list[str] = []
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            cached = cache["candidate_by_id"].get(candidate_id)
            if cached is None or cached.get("joint_values_rad") != candidate.get("joint_values_rad"):
                cache_misses.append(candidate_id)
                continue
            if cached.get("node_gate_fcl_valid") is True:
                fcl_ids.add(candidate_id)
            if cached.get("node_gate_bullet_valid") is True:
                bullet_ids.add(candidate_id)
            if cached.get("node_gate_dual_backend_valid") is True:
                valid_ids.add(candidate_id)
        if cache_misses:
            raise RuntimeError(f"native_node_cache_miss:{len(cache_misses)}:{cache_misses[:5]}")
        audit = {
            "candidate_count": len(candidates),
            "fcl_valid": len(fcl_ids),
            "bullet_valid": len(bullet_ids),
            "dual_backend_valid": len(valid_ids),
            "fcl_bullet_node_difference": len(fcl_ids ^ bullet_ids),
            "fcl_valid_ids": sorted(fcl_ids),
            "bullet_valid_ids": sorted(bullet_ids),
            "dual_backend_valid_ids": sorted(valid_ids),
            "records": [],
            "native_evidence_reused": True,
            "native_evidence_source_run": cache["canonical_run"],
            "native_processes_started": 0,
        }
        _write_json(out / "stage24h_native_node_gate.json", audit)
        return valid_ids, audit

    return cached_node_gate


def make_cached_edge_gate(cache: dict[str, Any]):
    def cached_edge_gate(out: Path, requests: list[dict[str, Any]], round_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        misses: list[tuple[Any, ...]] = []
        for request in requests:
            key = stage24g.edge_key(request)
            cached = cache["edge_by_key"].get(key)
            if cached is None:
                misses.append(key)
                continue
            evidence.append(cached)
        if misses:
            raise RuntimeError(f"native_edge_cache_miss:{len(misses)}:{misses[:3]}")

        fcl_keys = {
            stage24g.edge_key(item["request"])
            for item in evidence
            if item.get("fcl", {}).get("valid") is True and item.get("fcl", {}).get("status") == "accepted"
        }
        bullet_keys = {
            stage24g.edge_key(item["request"])
            for item in evidence
            if item.get("bullet", {}).get("valid") is True and item.get("bullet", {}).get("status") == "accepted"
        }
        dual_count = sum(1 for item in evidence if item.get("dual_backend_valid") is True)
        meta = {
            "native_checked": len(evidence),
            "fcl_valid": len(fcl_keys),
            "bullet_valid": len(bullet_keys),
            "dual_backend_valid": dual_count,
            "fcl_bullet_edge_difference": len(fcl_keys ^ bullet_keys),
            "records": [],
            "native_evidence_reused": True,
            "native_evidence_source_run": cache["canonical_run"],
            "native_processes_started": 0,
            "frontier_round": int(round_index),
        }
        batch_dir = out / f"cached_native_edge_batch{int(round_index):03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        stage24h.write_jsonl(batch_dir / "stage24h_native_edge_evidence.jsonl", evidence)
        return evidence, meta

    return cached_edge_gate


def _run_cached_rebuild(
    out: Path,
    index: int,
    cache: dict[str, Any],
    frozen: dict[str, Any],
    semantics: list[dict[str, Any]],
    contract: dict[str, Any],
    before_status: str,
    workers: int,
) -> dict[str, Any]:
    return stage24h.run_one(out / f"run{index}", index, frozen, semantics, contract, before_status, workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-run", type=Path, default=DEFAULT_CANONICAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.runs) != 3:
        raise ValueError("this_fast_audit_requires_exactly_three_semantic_rebuilds")
    canonical_run = args.canonical_run.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(f"refusing_to_overwrite_existing_output:{out}")
    out.mkdir(parents=True, exist_ok=False)

    before = stage24h.git_status()
    frozen = stage24h.verify_frozen_inputs()
    cache = _load_cache(canonical_run)
    source_gate = cache["gate"]
    source_hashes = source_gate.get("frozen_input_hashes")
    if source_hashes is not None and stage24h.value_hash(source_hashes) != stage24h.value_hash(frozen):
        raise RuntimeError("canonical_native_cache_frozen_input_hash_mismatch")
    if not frozen["all_397_inputs_verified"] or not frozen["persisted_stage24g_hashes_clean"]:
        raise RuntimeError("current_frozen_input_verification_failed")

    _write_json(out / "stage24h_frozen_input_hashes.json", frozen)
    _write_json(
        out / "stage24h_input_manifest.json",
        {
            "schema_version": "stage24h-fast-audit-input-manifest-v1",
            "stage": "2.4H",
            "frozen_input_hashes": frozen,
            "frozen_inputs_modified": False,
            "canonical_native_evidence_source_run": cache["canonical_run"],
            "native_evidence_reused": True,
            "native_processes_started_by_this_audit": 0,
            "formal_scope": "three independent semantic frontier rebuilds; one prior native FCL/Bullet execution reused by exact identity",
        },
    )

    # run_one resolves these names at call time.  The wrappers preserve the
    # existing frontier algorithm while replacing only the expensive native
    # process launches with exact immutable evidence lookups.
    stage24h.native_node_gate = make_cached_node_gate(cache)
    stage24h.native_edge_gate = make_cached_edge_gate(cache)
    semantics = stage24g.joint_semantics()
    contract = stage24g.frozen_tolerance_contract()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()
    for index in range(1, 4):
        try:
            results.append(_run_cached_rebuild(out, index, cache, frozen, semantics, contract, before, max(1, int(args.workers))))
        except Exception as exc:  # preserve a machine-readable failure rather than fabricating a pass
            error = {"run_index": index, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            errors.append(error)
            _write_json(out / f"run{index}" / "stage24h_error.json", error)
            break

    fields = [
        "node_set",
        "edge_set",
        "generated_candidate_ids",
        "candidate_joint_values",
        "node_validity",
        "accepted_edge_set",
        "forward_reachability",
        "reverse_reachability",
        "selected_chain",
        "gate_status",
    ]
    semantic_hashes = {field: [result["metadata"].get(field) for result in results] for field in fields}
    differences = {field: len(set(values)) - 1 for field, values in semantic_hashes.items()}
    semantic_passed = not errors and len(results) == 3 and all(value == 0 for value in differences.values())
    determinism = {
        "independent_semantic_rebuilds": len(results),
        "requested_semantic_rebuilds": 3,
        "independent_native_rebuilds": 0,
        "native_evidence_reused": True,
        "native_evidence_source_run": cache["canonical_run"],
        "semantic_hashes": semantic_hashes,
        "differences": differences,
        "semantic_rebuild_determinism_passed": semantic_passed,
        "formal_three_independent_native_rebuild_requirement": "not_satisfied_by_this_fast_audit",
        "errors": errors,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(out / "stage24h_determinism_report.json", determinism)

    if results:
        gate = dict(results[0]["gate"])
    else:
        gate = dict(source_gate)
    gate.update(
        {
            "deterministic_rebuilds": len(results),
            "determinism_passed": semantic_passed,
            "semantic_rebuild_determinism_passed": semantic_passed,
            "independent_native_rebuilds": 0,
            "native_evidence_reused": True,
            "native_evidence_source_run": cache["canonical_run"],
            "formal_three_independent_native_rebuild_requirement": "not_satisfied_by_this_fast_audit",
            "Stage_2_5": "blocked",
            "ready_for_Stage_2_5": False,
        }
    )
    if errors:
        gate["Stage_2_4H"] = "blocked_native_collision"
        gate["errors"] = errors
    _write_json(out / "stage24h_gate_report.json", gate)
    _write_json(
        out / "dirty_worktree_preservation.json",
        {
            "git_before": before,
            "git_after": stage24h.git_status(),
            "preexisting_status_preserved": all(line in stage24h.git_status() for line in before.splitlines() if line.strip()),
            "new_output_root": str(out),
        },
    )
    report = [
        "# Stage 2.4H fast cached rebuild audit",
        "",
        f"- Stage 2.4H native-search result inherited from canonical run: `{gate.get('Stage_2_4H')}`.",
        f"- Semantic frontier rebuilds: `{len(results)}/3`; deterministic: `{semantic_passed}`.",
        "- Native FCL/Bullet processes started by this audit: `0`.",
        f"- Native evidence source: `{cache['canonical_run']}`.",
        "- Scope limitation: this is not three independent native collision executions; the exact limitation is persisted in the JSON reports.",
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
                "semantic_rebuilds": len(results),
                "semantic_determinism_passed": semantic_passed,
                "independent_native_rebuilds": 0,
                "elapsed_seconds": determinism["elapsed_seconds"],
                "errors": errors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if semantic_passed and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
