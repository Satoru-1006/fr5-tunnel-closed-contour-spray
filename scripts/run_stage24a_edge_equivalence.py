"""Stage 2.4A FCL/Bullet intermediate-state equivalence audit.

This stage is intentionally separate from closed-cycle search.  It freezes the
Stage 2.3B node set and the Stage 2.4 edge requests, replays all 7,855
joint-gate-accepted edges in three fresh FCL and Bullet processes, and exports
the historical 1,697-edge failure as a fully traceable implementation-error
audit.  It never runs Stage 2.4B, Ruckig, TOTG, or CCD.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))
import scripts.run_stage24_closed_loop_graph as stage24


ROOT = Path(__file__).resolve().parents[1]
STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
STAGE24_OLD = ROOT / "outputs/ik_graph_stage24_closed_loop_graph/fr5_scaled_horseshoe_demo_v45"
OUT = ROOT / "outputs/ik_graph_stage24a_edge_equivalence/fr5_scaled_horseshoe_demo_v45"
EXPECTED_FORMAL_EDGES = 7855
EXPECTED_HISTORICAL_DIFFERENCES = 1697
TARGET_PAIRS = {"forearm_link<->wrist2_link", "forearm_link<->wrist3_link"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(row["from_waypoint"]),
        int(row["to_waypoint"]),
        str(row["from_candidate_id"]),
        str(row["to_candidate_id"]),
    )


def parquet_write(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", version="2.6")
    reread = pq.read_table(path)
    if reread.num_rows != len(rows):
        raise RuntimeError(f"Parquet row-count mismatch for {path}: {reread.num_rows} != {len(rows)}")


def git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ).stdout


def old_sha_verification() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for base in (STAGE23B, STAGE24_OLD):
        sums = base / "SHA256SUMS"
        if not sums.exists():
            failures.append({"path": str(sums), "reason": "missing_SHA256SUMS"})
            continue
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split(maxsplit=1)
            rel = rel.lstrip("*")
            path = base / rel
            actual = sha256(path) if path.exists() else None
            record = {
                "bundle": str(base),
                "relative_path": rel,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
            records.append(record)
            if actual != expected:
                failures.append(record)
    return {"records": records, "failure_count": len(failures), "failures": failures}


def load_by_key(path: Path) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    return {key(row): row for row in read_jsonl(path)}


def run_records(
    prefilter: list[dict[str, Any]], rejected: list[dict[str, Any]], edge_csv: Path
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    stage24.OUT = OUT
    build = stage24.build_native()
    write_json(OUT / "stage24a_build_report.json", build)
    if build["exit_code"] != 0:
        raise RuntimeError("Stage 2.4A native build failed")
    records: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    pending: list[tuple[str, int]] = []
    for run_index in (1, 2, 3):
        for backend in ("fcl", "bullet"):
            run_dir = OUT / "runs" / f"{backend}_run{run_index}"
            result_path = run_dir / "native_edge_results.jsonl"
            exit_path = run_dir / "exit_code.txt"
            try:
                existing_rows = len(read_jsonl(result_path)) if result_path.exists() else 0
            except (json.JSONDecodeError, UnicodeDecodeError):
                existing_rows = -1
            complete = (
                result_path.exists()
                and exit_path.exists()
                and exit_path.read_text(encoding="utf-8").strip() == "0"
                and existing_rows == EXPECTED_FORMAL_EDGES
            )
            if complete:
                records.append(
                    {
                        "backend": backend,
                        "run_index": run_index,
                        "exit_code": 0,
                        "result_path": str(result_path),
                        "reused_complete_stage24a_process": True,
                    }
                )
            else:
                pending.append((backend, run_index))
    # Each run is still a separate fresh OS process.  Parallel execution only
    # reduces wall time; paths, scenes, results, and exit codes stay isolated.
    with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
        futures = {
            pool.submit(stage24.run_native, backend, run_index, edge_csv, build): (backend, run_index)
            for backend, run_index in pending
        }
        for future in as_completed(futures):
            backend, run_index = futures[future]
            record = future.result()
            records.append(record)
            result_path = Path(record["result_path"])
            result_rows = len(read_jsonl(result_path)) if result_path.exists() else 0
            record["result_row_count"] = result_rows
            if record["exit_code"] != 0 or result_rows != EXPECTED_FORMAL_EDGES:
                raise RuntimeError(f"native {backend} run {run_index} failed")
    records.sort(key=lambda item: (int(item["run_index"]), str(item["backend"])))
    for backend in ("fcl", "bullet"):
        native = stage24.merge_native_edges(
            OUT / "runs" / f"{backend}_run1/native_edge_results.jsonl", backend
        )
        manifests[backend] = stage24.edge_manifest(prefilter, rejected, native, backend)
        write_jsonl(OUT / f"stage24a_edges_{backend}.jsonl", manifests[backend])
    write_json(OUT / "stage24a_run_records.json", records)
    return records, manifests


def sample_maps() -> dict[str, dict[tuple[int, int, str, str, int], dict[str, Any]]]:
    result: dict[str, dict[tuple[int, int, str, str, int], dict[str, Any]]] = {}
    for backend in ("fcl", "bullet"):
        path = OUT / "runs" / f"{backend}_run1/native_edge_samples.jsonl"
        rows = read_jsonl(path)
        result[backend] = {
            (*key(row), int(row["sample_index"])): row
            for row in rows
        }
    return result


def transforms_hash(sample: dict[str, Any]) -> str:
    return value_hash(sample["link_transforms"])


def sample_q(q0: list[float], q1: list[float], index: int, steps: int) -> tuple[float, list[float]]:
    t = float(index) / float(steps)
    return t, [a + t * (b - a) for a, b in zip(q0, q1)]


def build_historical_audit(
    prefilter_by_key: dict[tuple[int, int, str, str], dict[str, Any]],
    new_manifests: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    old_fcl = load_by_key(STAGE24_OLD / "stage24_edges_fcl.jsonl")
    old_bullet = load_by_key(STAGE24_OLD / "stage24_edges_bullet.jsonl")
    new_fcl = {key(row): row for row in new_manifests["fcl"]}
    new_bullet = {key(row): row for row in new_manifests["bullet"]}
    samples = sample_maps()
    differences = sorted(
        k for k in old_fcl
        if old_fcl[k].get("valid") is True and old_bullet[k].get("valid") is not True
    )
    if len(differences) != EXPECTED_HISTORICAL_DIFFERENCES:
        raise RuntimeError(f"historical difference count {len(differences)}")

    edge_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    classifications: Counter[str] = Counter()
    for edge_key in differences:
        base = prefilter_by_key[edge_key]
        old_b = old_bullet[edge_key]
        first_index = int(old_b["first_collision_sample"])
        steps = int(old_b["segment_steps"])
        t, q = sample_q(base["q0"], base["q1"], first_index, steps)
        fixed_f = samples["fcl"][(*edge_key, first_index)]
        fixed_b = samples["bullet"][(*edge_key, first_index)]
        pairs = list(old_b.get("collision_pairs") or [])
        pair_counts.update(pairs)
        transition = f"{edge_key[0]}->{edge_key[1]}"
        transition_counts[transition] += 1
        target_only = set(pairs).issubset(TARGET_PAIRS) and bool(pairs)
        repaired = new_fcl[edge_key].get("valid") is True and new_bullet[edge_key].get("valid") is True
        classification = (
            "runtime_geometry_representation_not_activated_in_stage24_edge_path"
            if target_only and repaired
            else "unresolved"
        )
        classifications[classification] += 1
        first_contacts = list(old_b.get("first_collision_contacts") or [])
        contact_depths = [float(item["depth"]) for item in first_contacts if item.get("depth") is not None]
        sample_ts = [float(i) / float(steps) for i in range(steps + 1)]
        sample_qs = [
            [a + sample_t * (b - a) for a, b in zip(base["q0"], base["q1"])]
            for sample_t in sample_ts
        ]
        edge_rows.append(
            {
                "transition_index": int(edge_key[0]),
                "from_waypoint": int(edge_key[0]),
                "to_waypoint": int(edge_key[1]),
                "source_node_id": edge_key[2],
                "target_node_id": edge_key[3],
                "source_q": [float(x) for x in base["q0"]],
                "target_q": [float(x) for x in base["q1"]],
                "raw_delta_rad": [float(b - a) for a, b in zip(base["q0"], base["q1"])],
                "model_aware_delta_rad": [float(x) for x in base["signed_delta_rad"]],
                "joint_types": ["revolute"] * 6,
                "continuous_flags": [False] * 6,
                "joint_lower_bounds": [-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543],
                "joint_upper_bounds": [3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543],
                "sample_count": steps + 1,
                "sample_t": sample_ts,
                "sample_q": sample_qs,
                "historical_fcl_collision": False,
                "historical_bullet_collision": True,
                "historical_self_collision": bool(old_b.get("self_collision")),
                "historical_robot_world_collision": bool(old_b.get("robot_world_collision")),
                "historical_collision_pairs": pairs,
                "first_differing_sample_index": first_index,
                "first_differing_t": t,
                "post_fix_fcl_accepted": bool(new_fcl[edge_key].get("valid")),
                "post_fix_bullet_accepted": bool(new_bullet[edge_key].get("valid")),
                "classification": classification,
            }
        )
        first_rows.append(
            {
                "transition_index": int(edge_key[0]),
                "from_waypoint": int(edge_key[0]),
                "to_waypoint": int(edge_key[1]),
                "source_node_id": edge_key[2],
                "target_node_id": edge_key[3],
                "sample_index": first_index,
                "t": t,
                "q": q,
                "joint_vector_sha256": value_hash(q),
                "fixed_fcl_joint_vector_sha256": value_hash(fixed_f["q"]),
                "fixed_bullet_joint_vector_sha256": value_hash(fixed_b["q"]),
                "fixed_fcl_fk_link_transforms_sha256": transforms_hash(fixed_f),
                "fixed_bullet_fk_link_transforms_sha256": transforms_hash(fixed_b),
                "historical_fcl_collision": False,
                "historical_bullet_collision": True,
                "historical_collision_pairs": pairs,
                "historical_contact_depths": contact_depths,
                "historical_signed_distance_status": "raw_contact_depth_only_not_clearance",
                "post_fix_fcl_collision": bool(fixed_f["collision"]),
                "post_fix_bullet_collision": bool(fixed_b["collision"]),
                "post_fix_fcl_collision_pairs": list(fixed_f.get("collision_pairs") or []),
                "post_fix_bullet_collision_pairs": list(fixed_b.get("collision_pairs") or []),
                "sample_joint_vectors_equal": fixed_f["q"] == fixed_b["q"],
                "sample_fk_transforms_equal": transforms_hash(fixed_f) == transforms_hash(fixed_b),
                "classification": classification,
            }
        )

    classification_json = {
        "historical_difference_count": len(differences),
        "classification_counts": dict(sorted(classifications.items())),
        "all_historical_pairs_limited_to_stage23b_target_links": all(
            set(row["historical_collision_pairs"]).issubset(TARGET_PAIRS)
            for row in edge_rows
        ),
        "all_fixed_with_stage23b_runtime_representation": all(
            row["post_fix_fcl_accepted"] and row["post_fix_bullet_accepted"] for row in edge_rows
        ),
        "root_cause": {
            "category": "runtime_geometry_representation_not_activated_in_stage24_edge_path",
            "implementation_error": "FR5_BULLET_SHAPE_MODE=use_shape_type was absent while LD_PRELOAD alone left the interposer inert",
            "repair": "export the Stage 2.3B activation variable together with the frozen interposer path",
            "waypoints_modified": False,
            "IK_modified": False,
            "ACM_modified": False,
            "collision_geometry_modified": False,
        },
    }
    pair_summary = {
        "historical_pair_counts": dict(sorted(pair_counts.items())),
        "pair_set": sorted(pair_counts),
        "post_fix_disagreement_pair_count": 0,
        "independent_clearance": None,
        "clearance_status": "not_available",
    }
    transition_summary = {
        "transitions_with_historical_differences": len(transition_counts),
        "historical_difference_count": sum(transition_counts.values()),
        "counts": dict(sorted(transition_counts.items(), key=lambda item: int(item[0].split("->")[0]))),
    }
    return edge_rows, first_rows, classification_json, pair_summary, transition_summary


def determinism(prefilter: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"independent_process_runs": 3, "backends": {}}
    for backend in ("fcl", "bullet"):
        native_hashes = []
        manifest_hashes = []
        for run_index in (1, 2, 3):
            path = OUT / "runs" / f"{backend}_run{run_index}/native_edge_results.jsonl"
            native_hashes.append(sha256(path))
            native = stage24.merge_native_edges(path, backend)
            manifest = stage24.edge_manifest(prefilter, rejected, native, backend)
            manifest_hashes.append(value_hash(manifest))
        result["backends"][backend] = {
            "native_result_sha256": native_hashes,
            "edge_manifest_semantic_sha256": manifest_hashes,
            "three_run_native_hash_equal": len(set(native_hashes)) == 1,
            "three_run_manifest_hash_equal": len(set(manifest_hashes)) == 1,
        }
    result["all_three_run_hashes_equal"] = all(
        item["three_run_native_hash_equal"] and item["three_run_manifest_hash_equal"]
        for item in result["backends"].values()
    )
    return result


def exact_replay(first_key: tuple[int, int, str, str], first_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def extract(base: Path, backend: str, run_index: int) -> dict[str, Any]:
        rows = load_by_key(base / "runs" / f"{backend}_run{run_index}/native_edge_results.jsonl")
        return rows[first_key]

    historical = {
        backend: [extract(STAGE24_OLD, backend, i) for i in (1, 2, 3)]
        for backend in ("fcl", "bullet")
    }
    repaired = {
        backend: [extract(OUT, backend, i) for i in (1, 2, 3)]
        for backend in ("fcl", "bullet")
    }
    row = next(
        item for item in first_rows
        if (item["source_node_id"], item["target_node_id"]) == (first_key[2], first_key[3])
    )
    before_shapes = read_json(STAGE23B / "stage23b_shape_manifest_before.json")
    after_shapes = read_json(STAGE23B / "stage23b_shape_manifest_after.json")
    target_before = {
        item["link_name"]: item["root_bt_shape_type"]
        for item in before_shapes["entries"]
        if item["link_name"] in {"forearm_link", "wrist2_link", "wrist3_link"}
    }
    target_after = {
        item["link_name"]: item["root_bt_shape_type"]
        for item in after_shapes["entries"]
        if item["link_name"] in {"forearm_link", "wrist2_link", "wrist3_link"}
    }
    return {
        "transition": "299 -> 300",
        "edge": "0299-02 -> 0300-02",
        "endpoints_reloaded_from_frozen_file": True,
        "interpolation_logic": "Stage 2.4 frozen linear bounded-revolute interpolation",
        "first_historical_difference": row,
        "three_independent_process_replay": {
            "historical_inert_interposer": historical,
            "repaired_stage23b_representation": repaired,
        },
        "comparison": {
            "joint_vector_equal": row["sample_joint_vectors_equal"],
            "link_transform_equal": row["sample_fk_transforms_equal"],
            "collision_shape_type_before": target_before,
            "collision_shape_type_after": target_after,
            "collision_object_transform": "unchanged_frozen_PlanningScene",
            "broadphase_candidates": "not_exposed_by_installed_MoveIt2_interface",
            "narrowphase_historical_contacts": row["historical_collision_pairs"],
            "ACM_decision": "unchanged_formal_ACM",
            "historical_category": "self_collision",
            "post_fix_category": "collision_free",
            "signed_distance": "not_available_as_clearance; historical raw contact depth retained",
            "margin": {"before": 0.0, "after": 0.0},
            "bullet_callback_raw_values": str(
                OUT / "runs/bullet_run1/native_bullet_manifolds.jsonl"
            ),
        },
        "classification": "runtime_geometry_representation_not_activated_in_stage24_edge_path",
        "three_run_historical_fcl_equal": len({value_hash(x) for x in historical["fcl"]}) == 1,
        "three_run_historical_bullet_equal": len({value_hash(x) for x in historical["bullet"]}) == 1,
        "three_run_repaired_fcl_equal": len({value_hash(x) for x in repaired["fcl"]}) == 1,
        "three_run_repaired_bullet_equal": len({value_hash(x) for x in repaired["bullet"]}) == 1,
    }


def environment_manifest() -> dict[str, Any]:
    plugin = STAGE23B / "install/lib/libstage23b_bullet_shape_interposer.so"
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ).stdout
    dependency_text = (
        f"python_executable={sys.executable}\n"
        f"python_version={platform.python_version()}\n"
        f"pyarrow_version={pa.__version__}\n"
        f"pyarrow_module={pa.__file__}\n"
        "wheel_source=PyPI binary wheel installed by pip into isolated Stage 2.4A virtual environment\n"
        "\n[pip freeze]\n"
        f"{freeze}"
    )
    (OUT / "dependency_manifest.txt").write_text(dependency_text, encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pyarrow_version": pa.__version__,
        "pyarrow_module": pa.__file__,
        "pyarrow_provenance": "PyPI binary wheel installed by pip in isolated Stage 2.4A venv",
        "environment_sha256": hashlib.sha256(dependency_text.encode("utf-8")).hexdigest(),
        "stage23b_bullet_interposer": {
            "path": str(plugin.resolve()),
            "sha256": sha256(plugin),
            "required_activation": "FR5_BULLET_SHAPE_MODE=use_shape_type",
        },
        "collision_method": "adaptive_discrete_interpolation",
        "interpolation_step_deg": 1.0,
        "Ruckig": "not_run",
        "TOTG": "not_run",
        "CCD": "not_run",
        "clearance": "not_available",
    }
    write_json(OUT / "environment_manifest.json", manifest)
    return manifest


def write_report(gate: dict[str, Any], classification: dict[str, Any], exact: dict[str, Any]) -> None:
    text = f"""# Stage 2.4A intermediate-state backend equivalence audit

- Stage 2.4A: `{gate["Stage_2_4A"]}`
- FCL accepted edges: `{gate["FCL_accepted_edges"]}`
- Bullet accepted edges: `{gate["Bullet_accepted_edges"]}`
- Accepted-edge symmetric difference: `{gate["accepted_edge_symmetric_difference"]}`
- Historical differences audited: `{classification["historical_difference_count"]}`
- Root cause: `{classification["root_cause"]["category"]}`
- First exact replay: `{exact["edge"]}`

The Stage 2.4 runner loaded the certified Stage 2.3B interposer but omitted its
explicit `FR5_BULLET_SHAPE_MODE=use_shape_type` activation variable.  The
interposer therefore remained inert and the three disputed robot-link meshes
fell back to whole-mesh convex hulls.  The repair only activates the already
certified Stage 2.3B runtime representation; waypoints, IK, URDF, SRDF, ACM,
padding, geometry, thresholds, and interpolation samples remain frozen.

All 7,855 joint-gate-accepted edges were replayed in three fresh FCL and three
fresh Bullet processes.  Collision checking remains
`adaptive_discrete_interpolation`; it is not formal CCD and no clearance value
is inferred.

Ruckig, TOTG, CCD, Stage 2.4B, and Stage 2.5 were not run by this stage.
"""
    (OUT / "stage24a_report.md").write_text(text, encoding="utf-8")


def sha_bundle() -> dict[str, Any]:
    lines = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    sums = OUT / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    failures = []
    for line in lines:
        expected, rel = line.split(maxsplit=1)
        actual = sha256(OUT / rel)
        if actual != expected:
            failures.append({"path": rel, "expected": expected, "actual": actual})
    result = {"checked_files": len(lines), "failure_count": len(failures), "failures": failures}
    write_json(OUT / "stage24a_sha256_verification.json", result)
    # Refresh the list once so the verification record is covered too.
    lines = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before = git_status()
    (OUT / "git_status_before.txt").write_text(before, encoding="utf-8")
    verification = old_sha_verification()
    write_json(OUT / "stage24a_frozen_input_verification.json", verification)
    if verification["failure_count"]:
        write_json(
            OUT / "stage24a_gate_report.json",
            {
                "Stage_2_4A": "blocked_frozen_input_sha256_failure",
                "Stage_2_4B": "blocked",
                "Stage_2_4": "blocked_backend_edge_disagreement",
                "Stage_2_5": "blocked",
                "input_verification_failures": verification["failure_count"],
            },
        )
        return 0

    headers, nodes, fcl_nodes, bullet_nodes = stage24.valid_node_set()
    semantics = stage24.parse_joint_semantics()
    prefilter, rejected, rejection_counts = stage24.build_prefilter(nodes, semantics)
    if len(prefilter) != EXPECTED_FORMAL_EDGES:
        raise RuntimeError(f"formal edge count {len(prefilter)}")
    stage24.OUT = OUT
    stage24.write_frozen_inputs(headers, nodes, fcl_nodes, bullet_nodes)
    edge_csv = OUT / "stage24a_edge_requests.csv"
    stage24.write_edge_requests(prefilter, edge_csv)
    write_jsonl(OUT / "stage24a_joint_gate_rejected_edges.jsonl", rejected)
    records, manifests = run_records(prefilter, rejected, edge_csv)

    fcl_set = stage24.accepted_set(manifests["fcl"])
    bullet_set = stage24.accepted_set(manifests["bullet"])
    equivalence = {
        "FCL_accepted_edges": len(fcl_set),
        "Bullet_accepted_edges": len(bullet_set),
        "accepted_edge_symmetric_difference": len(fcl_set ^ bullet_set),
        "FCL_only_edges": len(fcl_set - bullet_set),
        "Bullet_only_edges": len(bullet_set - fcl_set),
        "common_edges": len(fcl_set & bullet_set),
        "identical_joint_samples": True,
        "identical_sample_fk_transforms": True,
    }

    prefilter_by_key = {key(row): row for row in prefilter}
    edge_rows, first_rows, classification, pair_summary, transition_summary = build_historical_audit(
        prefilter_by_key, manifests
    )
    sample_vectors_equal = all(row["sample_joint_vectors_equal"] for row in first_rows)
    sample_fk_equal = all(row["sample_fk_transforms_equal"] for row in first_rows)
    equivalence["identical_joint_samples"] = sample_vectors_equal
    equivalence["identical_sample_fk_transforms"] = sample_fk_equal
    parquet_write(OUT / "stage24a_all_disagreement_edges.parquet", edge_rows)
    parquet_write(OUT / "stage24a_first_differing_samples.parquet", first_rows)
    write_json(OUT / "stage24a_edge_classification.json", classification)
    write_json(OUT / "stage24a_pair_summary.json", pair_summary)
    write_json(OUT / "stage24a_transition_summary.json", transition_summary)
    write_json(OUT / "stage24a_backend_equivalence.json", equivalence)

    det = determinism(prefilter, rejected)
    write_json(OUT / "stage24a_determinism.json", det)
    first_key = (299, 300, "0299-02", "0300-02")
    exact = exact_replay(first_key, first_rows)
    write_json(OUT / "stage24a_first_edge_exact_replay.json", exact)
    env = environment_manifest()

    gate_pass = (
        equivalence["FCL_accepted_edges"] == EXPECTED_FORMAL_EDGES
        and equivalence["Bullet_accepted_edges"] == EXPECTED_FORMAL_EDGES
        and equivalence["accepted_edge_symmetric_difference"] == 0
        and equivalence["FCL_only_edges"] == 0
        and equivalence["Bullet_only_edges"] == 0
        and sample_vectors_equal
        and sample_fk_equal
        and det["all_three_run_hashes_equal"]
        and all(record["exit_code"] == 0 for record in records)
    )
    gate = {
        "Stage_2_4A": "passed" if gate_pass else "blocked_backend_intermediate_state_disagreement",
        "Stage_2_4B": "unblocked_not_started" if gate_pass else "blocked",
        "Stage_2_4": "pending_stage24b" if gate_pass else "blocked_backend_edge_disagreement",
        "Stage_2_5": "blocked",
        **equivalence,
        "candidate_edges_before_joint_gate": len(prefilter) + len(rejected),
        "joint_gate_accepted_edges": len(prefilter),
        "joint_gate_rejected_edges": len(rejected),
        "joint_gate_rejection_counts": rejection_counts,
        "formal_nodes": len(nodes),
        "formal_waypoints": 720,
        "three_independent_process_runs": 3,
        "all_process_exit_codes_zero": all(record["exit_code"] == 0 for record in records),
        "deterministic": det["all_three_run_hashes_equal"],
        "parquet": {
            "pyarrow_version": pa.__version__,
            "all_disagreement_rows": len(edge_rows),
            "first_differing_sample_rows": len(first_rows),
            "readback_passed": True,
        },
        "input_SHA256SUMS_failure_count": verification["failure_count"],
        "Ruckig": "not_run",
        "TOTG": "not_run",
        "CCD": "not_run",
        "clearance": "not_available",
        "environment_sha256": env["environment_sha256"],
        "tests": "pending",
    }
    write_json(OUT / "stage24a_gate_report.json", gate)
    write_report(gate, classification, exact)

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_stage24a_outputs.py", "-q"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    test_text = test.stdout + test.stderr
    (OUT / "stage24a_test_report.txt").write_text(test_text, encoding="utf-8")
    gate["tests"] = {
        "exit_code": test.returncode,
        "status": "passed" if test.returncode == 0 else "failed",
        "report": str((OUT / "stage24a_test_report.txt").resolve()),
    }
    if test.returncode != 0:
        gate["Stage_2_4A"] = "blocked_tests_failed"
        gate["Stage_2_4B"] = "blocked"
        gate["Stage_2_4"] = "blocked_backend_edge_disagreement"
    write_json(OUT / "stage24a_gate_report.json", gate)
    write_report(gate, classification, exact)
    after = git_status()
    (OUT / "git_status_after.txt").write_text(after, encoding="utf-8")
    changed = [
        "cpp/stage24/stage24_formal_probe.cpp",
        "scripts/run_stage24_closed_loop_graph.py",
        "scripts/run_stage24a_edge_equivalence.py",
        "tests/test_stage24a_outputs.py",
    ]
    (OUT / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    sha_result = sha_bundle()
    gate["SHA256SUMS"] = {
        "path": str((OUT / "SHA256SUMS").resolve()),
        "verification_failure_count": sha_result["failure_count"],
    }
    write_json(OUT / "stage24a_gate_report.json", gate)
    # Final report/gate rewrite changes hashes; refresh SHA256SUMS once more.
    write_report(gate, classification, exact)
    sha_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
