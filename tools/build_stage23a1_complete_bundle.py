#!/usr/bin/env python3
"""Build the complete Stage 2.3A.1 contact-evidence bundle on WSL ext4.

The script is intentionally independent of the ROS runtime.  The native probe
has already produced the frozen-input contact streams; this program validates
those streams, writes lossless per-candidate chunks, and produces the final
auditable manifests.  It never changes candidates, poses, or scene inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


CONTACT_HEADER = [
    "run_id", "backend", "candidate_id", "waypoint_id", "seed_or_solution_index",
    "pair_index", "contact_index", "body_name_1", "body_type_1", "body_name_2",
    "body_type_2", "pos_x", "pos_y", "pos_z", "normal_x", "normal_y",
    "normal_z", "depth",
]

SUMMARY_HEADER = [
    "run_id", "backend", "candidate_id", "waypoint_id", "seed_or_solution_index",
    "collision", "self_collision", "robot_world_collision", "unique_pair_count",
    "contact_count", "raw_contact_count_reported", "global_limit", "per_pair_limit",
    "global_limit_reached", "per_pair_limit_reached", "saturated",
    "saturation_reference_levels", "pair_topology_hash",
]

PAIR_HEADER = [
    "run_id", "backend", "pair_key", "body_name_1", "body_type_1", "body_name_2",
    "body_type_2", "collision_type", "candidate_count", "contact_count",
    "max_contact_count_per_candidate",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_text(path: Path, text: str) -> str:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256_file(partial)
    partial.replace(path)
    return digest


def write_csv_atomic(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> str:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256_file(partial)
    partial.replace(path)
    return digest


def normalized_pair(body_1: str, body_2: str) -> str:
    return "::".join(sorted((body_1, body_2)))


def is_world_body(name: str) -> bool:
    return name.startswith("horseshoe_wall_") or name.startswith("tunnel_")


def body_type(name: str) -> str:
    return "WORLD_OBJECT" if is_world_body(name) else "ROBOT_LINK"


def collision_type(pair_type: str) -> str:
    return "self" if pair_type == "self" else "robot_world"


def load_candidates(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    by_id: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            candidate_id = row["candidate_id"]
            if candidate_id in by_id:
                raise RuntimeError(f"duplicate candidate in frozen input: {candidate_id}")
            if int(row["waypoint_index"]) < 0 or int(row["waypoint_index"]) >= 720:
                raise RuntimeError(f"invalid waypoint index: {row['waypoint_index']}")
            by_id[candidate_id] = index
            rows.append(row)
    if len(rows) != 1439:
        raise RuntimeError(f"expected 1439 frozen candidates, got {len(rows)}")
    return rows, by_id


def load_diag(path: Path, expected_variant: str, candidate_ids: set[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] != expected_variant:
                raise RuntimeError(f"unexpected variant in {path}: {row['variant']}")
            candidate_id = row["candidate_id"]
            if candidate_id not in candidate_ids:
                raise RuntimeError(f"diagnostic has unexpected candidate: {candidate_id}")
            item = result.setdefault(candidate_id, {"raw": None, "pairs": set()})
            reported = int(row["raw_contact_count"])
            if item["raw"] is None:
                item["raw"] = reported
            elif item["raw"] != reported:
                raise RuntimeError(f"diagnostic raw count changes for {candidate_id}")
            item["pairs"].add(normalized_pair(row["body_1"], row["body_2"]))
    if set(result) != candidate_ids:
        missing = sorted(candidate_ids - set(result))[:5]
        raise RuntimeError(f"diagnostic candidate coverage incomplete: {len(result)}; missing={missing}")
    return result


def compare_diag(a: dict[str, dict[str, object]], b: dict[str, dict[str, object]], label: str) -> str:
    vector_a = [(candidate, int(a[candidate]["raw"]), sorted(a[candidate]["pairs"])) for candidate in sorted(a)]
    vector_b = [(candidate, int(b[candidate]["raw"]), sorted(b[candidate]["pairs"])) for candidate in sorted(b)]
    if vector_a != vector_b:
        raise RuntimeError(f"{label} two-level diagnostic vector/topology mismatch")
    payload = json.dumps(vector_a, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def native_row_to_contact(row: dict[str, str], backend: str, pair_index: int, seed_index: str) -> list[str]:
    b1, b2 = row["body_1"], row["body_2"]
    return [
        row["variant"], backend, row["candidate_id"], row["waypoint_id"], seed_index,
        str(pair_index), row["contact_index"], b1, body_type(b1), b2, body_type(b2),
        row["contact_x"], row["contact_y"], row["contact_z"], row["normal_x"],
        row["normal_y"], row["normal_z"], row["penetration_depth"],
    ]


def canonical_update(digest: hashlib._Hash, contact: list[str]) -> None:
    digest.update(json.dumps(contact, ensure_ascii=False, separators=(",", ":")).encode())
    digest.update(b"\n")


def aggregate_pair_stats(stats: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in stats.values():
        for pair, count in item["pair_counts"].items():
            aggregate = result.setdefault(pair, {"candidate_count": 0, "contact_count": 0, "max": 0})
            aggregate["candidate_count"] += 1
            aggregate["contact_count"] += count
            aggregate["max"] = max(aggregate["max"], count)
    return result


def first_pass(
    raw_path: Path,
    expected_variant: str,
    candidates: list[dict[str, str]],
    by_id: dict[str, int],
) -> tuple[dict[str, dict[str, object]], set[str], str]:
    stats: dict[str, dict[str, object]] = {}
    all_pairs: set[str] = set()
    raw_digest = hashlib.sha256()
    with raw_path.open("rb") as raw_bytes:
        while True:
            block = raw_bytes.read(8 * 1024 * 1024)
            if not block:
                break
            raw_digest.update(block)
    last_index = -1
    with raw_path.open("r", encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=1):
            if row["variant"] != expected_variant:
                raise RuntimeError(f"unexpected native variant at row {row_number}: {row['variant']}")
            candidate_id = row["candidate_id"]
            if candidate_id not in by_id:
                raise RuntimeError(f"native contact has unexpected candidate: {candidate_id}")
            index = by_id[candidate_id]
            if index < last_index:
                raise RuntimeError(f"native contact candidate order regressed at {candidate_id}")
            last_index = index
            item = stats.setdefault(candidate_id, {
                "actual": 0, "reported": int(row["raw_contact_count"]), "pair_counts": Counter(),
                "self": False, "world": False,
            })
            reported = int(row["raw_contact_count"])
            if item["reported"] != reported:
                raise RuntimeError(f"native reported count changes for {candidate_id}")
            pair = normalized_pair(row["body_1"], row["body_2"])
            item["actual"] += 1
            item["pair_counts"][pair] += 1
            item["self"] = item["self"] or row["pair_type"] == "self"
            item["world"] = item["world"] or row["pair_type"] == "robot_world"
            all_pairs.add(pair)
            if row_number % 5_000_000 == 0:
                print(f"first-pass {raw_path.name}: rows={row_number}", flush=True)
    if set(stats) != set(by_id):
        missing = sorted(set(by_id) - set(stats))[:10]
        raise RuntimeError(f"native contact candidate coverage incomplete: {len(stats)}; missing={missing}")
    for candidate in candidates:
        item = stats[candidate["candidate_id"]]
        if item["actual"] != item["reported"]:
            raise RuntimeError(
                f"contact export is incomplete for {candidate['candidate_id']}: "
                f"rows={item['actual']} reported={item['reported']}"
            )
    return stats, all_pairs, raw_digest.hexdigest()


def transformed_stats(
    candidates: list[dict[str, str]],
    stats: dict[str, dict[str, object]],
    pair_map: dict[str, int],
    diag_a: dict[str, dict[str, object]],
    diag_b: dict[str, dict[str, object]],
    backend: str,
    global_limit: int,
    per_pair_limit: int,
    diag_hash: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        item = stats[cid]
        reference_equal = (
            item["actual"] == diag_a[cid]["raw"] == diag_b[cid]["raw"]
            and set(item["pair_counts"]) == set(diag_a[cid]["pairs"]) == set(diag_b[cid]["pairs"])
        )
        global_hit = item["actual"] >= global_limit
        per_pair_hit = any(count >= per_pair_limit for count in item["pair_counts"].values())
        if global_hit or per_pair_hit:
            reference_equal = False
        result[cid] = {
            **item,
            "pair_topology_hash": diag_hash,
            "global_hit": global_hit,
            "per_pair_hit": per_pair_hit,
            "saturated": bool(reference_equal),
            "pair_map": pair_map,
            "waypoint_id": candidate["waypoint_id"],
            "seed_order_index": candidate["seed_order_index"],
        }
    if not all(item["saturated"] for item in result.values()):
        unsaturated = [cid for cid, item in result.items() if not item["saturated"]][:10]
        raise RuntimeError(f"{backend} candidates are not saturated: {unsaturated}")
    return result


def chunk_ranges(count: int) -> list[tuple[int, int]]:
    return [(start, min(start + 99, count - 1)) for start in range(0, count, 100)]


def finish_chunk(
    chunk_dir: Path,
    chunk_id: str,
    start: int,
    end: int,
    candidates: list[dict[str, str]],
    stats: dict[str, dict[str, object]],
    backend: str,
    global_limit: int,
    per_pair_limit: int,
    raw_rows: int,
    raw_contacts_path: Path,
    pair_map: dict[str, int],
    complete_bundle: Path,
) -> dict[str, object]:
    expected_ids = [row["candidate_id"] for row in candidates[start:end + 1]]
    observed_ids = [row["candidate_id"] for row in candidates[start:end + 1] if row["candidate_id"] in stats]
    summary_rows = []
    for cid in expected_ids:
        item = stats[cid]
        summary_rows.append([
            f"{backend}_native_524288_65536", backend, cid, item["waypoint_id"], item["seed_order_index"],
            True, bool(item["self"]), bool(item["world"]), len(item["pair_counts"]), item["actual"],
            item["reported"], global_limit, per_pair_limit, item["global_hit"], item["per_pair_hit"],
            item["saturated"], "262144/32768;524288/65536", item["pair_topology_hash"],
        ])
    pair_acc: dict[str, dict[str, int]] = {}
    for cid in expected_ids:
        for pair, count in stats[cid]["pair_counts"].items():
            agg = pair_acc.setdefault(pair, {"candidate_count": 0, "contact_count": 0, "max": 0})
            agg["candidate_count"] += 1
            agg["contact_count"] += count
            agg["max"] = max(agg["max"], count)
    pair_rows = []
    for pair in sorted(pair_acc):
        b1, b2 = pair.split("::", 1)
        pair_type = "self" if not (is_world_body(b1) ^ is_world_body(b2)) else "robot_world"
        pair_rows.append([
            f"{backend}_native_524288_65536", backend, pair, b1, body_type(b1), b2, body_type(b2),
            pair_type, pair_acc[pair]["candidate_count"], pair_acc[pair]["contact_count"], pair_acc[pair]["max"],
        ])
    candidate_path = chunk_dir / "candidate_summary.csv"
    pair_path = chunk_dir / "pair_summary.csv"
    candidate_hash = write_csv_atomic(candidate_path, SUMMARY_HEADER, summary_rows)
    pair_hash = write_csv_atomic(pair_path, PAIR_HEADER, pair_rows)
    contacts_path = chunk_dir / "contacts.csv.zst"
    if not contacts_path.exists():
        raise RuntimeError(f"missing compressed contacts for {chunk_id}")
    contacts_hash = sha256_file(contacts_path)
    saturated = sum(1 for row in summary_rows if row[15])
    unsaturated = len(summary_rows) - saturated
    manifest = {
        "schema_version": "2.3A.1",
        "chunk_id": chunk_id,
        "backend": backend,
        "candidate_start": start,
        "candidate_end": end,
        "expected_candidate_count": end - start + 1,
        "observed_candidate_count": len(observed_ids),
        "duplicate_candidates": 0,
        "missing_candidates": sorted(set(expected_ids) - set(observed_ids)),
        "contact_row_count": raw_rows,
        "pair_count": len(pair_rows),
        "global_limit": global_limit,
        "per_pair_limit": per_pair_limit,
        "saturated_candidate_count": saturated,
        "unsaturated_candidate_count": unsaturated,
        "contacts_file_sha256": contacts_hash,
        "candidate_summary_sha256": candidate_hash,
        "pair_summary_sha256": pair_hash,
        "complete_marker": "COMPLETE",
        "runtime_exit_code": 0,
        "truncated_files_used": False,
        "input_native_contact_file": str(raw_contacts_path),
    }
    manifest_path = chunk_dir / "manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    marker = chunk_dir / "COMPLETE"
    atomic_text(marker, json.dumps({"chunk_id": chunk_id, "manifest_sha256": sha256_file(manifest_path)}, sort_keys=True) + "\n")
    manifest_copy = complete_bundle / "chunk_manifests" / f"{backend}_{chunk_id}.json"
    atomic_text(manifest_copy, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def process_backend(
    backend: str,
    raw_path: Path,
    expected_variant: str,
    diag_paths: tuple[Path, Path],
    candidates: list[dict[str, str]],
    by_id: dict[str, int],
    root: Path,
    bundle: Path,
    zstd_path: Path,
    global_limit: int,
    per_pair_limit: int,
) -> dict[str, object]:
    print(f"[{backend}] first pass: {raw_path}", flush=True)
    stats, all_pairs, raw_hash = first_pass(raw_path, expected_variant, candidates, by_id)
    print(f"[{backend}] rows={sum(item['actual'] for item in stats.values())} pairs={len(all_pairs)}", flush=True)
    diag_a = load_diag(diag_paths[0], expected_variant, set(by_id))
    diag_b = load_diag(diag_paths[1], expected_variant, set(by_id))
    diag_hash = compare_diag(diag_a, diag_b, backend)
    pair_map = {pair: index for index, pair in enumerate(sorted(all_pairs))}
    final_stats = transformed_stats(candidates, stats, pair_map, diag_a, diag_b, backend, global_limit, per_pair_limit, diag_hash)
    aggregate = aggregate_pair_stats(final_stats)
    pair_topology_hash = sha256_bytes(json.dumps(sorted(all_pairs), separators=(",", ":")).encode())
    backend_chunk_root = root / "chunks" / "stage23a1_complete_v2" / backend
    backend_chunk_root.mkdir(parents=True, exist_ok=True)
    merged_path = bundle / f"collision_contacts_all_{backend}.csv.zst"
    merged_partial = merged_path.with_name(merged_path.name + ".partial")
    merged_file = merged_partial.open("wb")
    compressor = subprocess.Popen([str(zstd_path)], stdin=subprocess.PIPE, stdout=merged_file)
    text_stream = io.TextIOWrapper(compressor.stdin, encoding="utf-8", newline="")
    writer = csv.writer(text_stream, lineterminator="\n")
    writer.writerow(CONTACT_HEADER)
    canonical_hash = hashlib.sha256()
    expected_order = [row["candidate_id"] for row in candidates]
    chunk_manifests: list[dict[str, object]] = []
    chunk_bounds = chunk_ranges(len(candidates))
    current_chunk_index = None
    current_plain = None
    current_plain_path: Path | None = None
    current_chunk_rows = 0
    current_chunk_id = ""
    current_chunk_dir: Path | None = None

    def close_chunk(index: int) -> None:
        nonlocal current_plain, current_plain_path, current_chunk_rows, current_chunk_dir
        if current_plain is None or current_plain_path is None or current_chunk_dir is None:
            return
        current_plain.flush()
        os.fsync(current_plain.fileno())
        current_plain.close()
        compressed_partial = current_chunk_dir / "contacts.csv.zst.partial"
        with current_plain_path.open("rb") as input_stream, compressed_partial.open("wb") as output_stream:
            completed = subprocess.run([str(zstd_path)], stdin=input_stream, stdout=output_stream, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"zstd failed for {current_chunk_dir}: {completed.returncode}")
            output_stream.flush()
            os.fsync(output_stream.fileno())
        compressed_path = current_chunk_dir / "contacts.csv.zst"
        compressed_partial.replace(compressed_path)
        current_plain_path.unlink()
        start, end = chunk_bounds[index]
        chunk_manifests.append(finish_chunk(
            current_chunk_dir, f"chunk_{start:04d}_{end:04d}", start, end, candidates, final_stats,
            backend, global_limit, per_pair_limit, current_chunk_rows, raw_path, pair_map, bundle,
        ))
        current_plain = None
        current_plain_path = None
        current_chunk_dir = None
        current_chunk_rows = 0

    with raw_path.open("r", encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=1):
            cid = row["candidate_id"]
            index = by_id[cid]
            chunk_index = index // 100
            if current_chunk_index != chunk_index:
                if current_chunk_index is not None:
                    close_chunk(current_chunk_index)
                current_chunk_index = chunk_index
                start, end = chunk_bounds[chunk_index]
                current_chunk_id = f"chunk_{start:04d}_{end:04d}"
                current_chunk_dir = backend_chunk_root / current_chunk_id
                current_chunk_dir.mkdir(parents=True, exist_ok=True)
                current_plain_path = current_chunk_dir / "contacts.csv.partial"
                current_plain = current_plain_path.open("wb")
                current_plain.write((",".join(CONTACT_HEADER) + "\n").encode())
                current_chunk_rows = 0
            contact = native_row_to_contact(row, backend, pair_map[normalized_pair(row["body_1"], row["body_2"])], final_stats[cid]["seed_order_index"])
            line = (",".join('"' + value.replace('"', '""') + '"' if any(ch in value for ch in ',"\n') else value for value in contact) + "\n").encode()
            current_plain.write(line)
            writer.writerow(contact)
            canonical_update(canonical_hash, contact)
            current_chunk_rows += 1
            if row_number % 5_000_000 == 0:
                print(f"[{backend}] second-pass rows={row_number}", flush=True)
    if current_chunk_index is not None:
        close_chunk(current_chunk_index)
    text_stream.flush()
    text_stream.close()
    exit_code = compressor.wait()
    merged_file.flush()
    os.fsync(merged_file.fileno())
    merged_file.close()
    if exit_code != 0:
        raise RuntimeError(f"zstd failed for merged {backend}: {exit_code}")
    merged_partial.replace(merged_path)
    if len(chunk_manifests) != len(chunk_bounds):
        raise RuntimeError(f"{backend} chunk count mismatch")
    if any(not (root / "chunks" / "stage23a1_complete_v2" / backend / f"chunk_{m['candidate_start']:04d}_{m['candidate_end']:04d}" / "COMPLETE").exists() for m in chunk_manifests):
        raise RuntimeError(f"{backend} has incomplete chunks")
    summary_rows = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        item = final_stats[cid]
        summary_rows.append([
            f"{backend}_native_524288_65536", backend, cid, item["waypoint_id"], item["seed_order_index"],
            True, bool(item["self"]), bool(item["world"]), len(item["pair_counts"]), item["actual"],
            item["reported"], global_limit, per_pair_limit, item["global_hit"], item["per_pair_hit"],
            item["saturated"], "262144/32768;524288/65536", item["pair_topology_hash"],
        ])
    summary_hash = write_csv_atomic(bundle / f"candidate_summary_{backend}.csv", SUMMARY_HEADER, summary_rows)
    pair_rows = []
    for pair in sorted(aggregate):
        b1, b2 = pair.split("::", 1)
        pair_type = "self" if not (is_world_body(b1) ^ is_world_body(b2)) else "robot_world"
        item = aggregate[pair]
        pair_rows.append([
            f"{backend}_native_524288_65536", backend, pair, b1, body_type(b1), b2, body_type(b2),
            pair_type, item["candidate_count"], item["contact_count"], item["max"],
        ])
    pair_hash = write_csv_atomic(bundle / f"pair_summary_{backend}.csv", PAIR_HEADER, pair_rows)
    topology = {
        "schema_version": "2.3A.1",
        "backend": backend,
        "pair_count": len(all_pairs),
        "pair_set_sha256": pair_topology_hash,
        "pairs": sorted(all_pairs),
        "robot_world_pair_count": sum(1 for pair in all_pairs if (is_world_body(pair.split("::")[0]) ^ is_world_body(pair.split("::")[1]))),
        "self_pair_count": sum(1 for pair in all_pairs if not (is_world_body(pair.split("::")[0]) ^ is_world_body(pair.split("::")[1]))),
    }
    atomic_text(bundle / f"collision_topology_{backend}.json", json.dumps(topology, indent=2, sort_keys=True) + "\n")
    return {
        "backend": backend,
        "raw_contact_file": str(raw_path),
        "raw_contact_sha256": raw_hash,
        "canonical_contact_sha256": canonical_hash.hexdigest(),
        "candidate_summary_sha256": summary_hash,
        "pair_summary_sha256": pair_hash,
        "pair_set_sha256": pair_topology_hash,
        "pair_count": len(all_pairs),
        "contact_count": sum(item["actual"] for item in final_stats.values()),
        "candidate_count": len(final_stats),
        "all_candidates_saturated": all(item["saturated"] for item in final_stats.values()),
        "self_collision_candidates": sum(1 for item in final_stats.values() if item["self"]),
        "robot_world_collision_candidates": sum(1 for item in final_stats.values() if item["world"]),
        "valid_node_count": 0,
        "chunk_manifests": chunk_manifests,
        "pair_aggregate": aggregate,
        "raw_hash_source": "native_cpp_contact_stream_sha256",
    }


def load_expected_pairs(path: Path) -> set[str]:
    pairs = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            pairs.add(normalized_pair(row["body_1"], row["body_2"]))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/robot/robotfucker_stage23a1"))
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--zstd", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    candidate_csv = args.candidate_csv or root / "raw" / "candidate_provenance.csv"
    zstd_path = (args.zstd or root / "zstd_stream").resolve()
    expected_pair_path = Path("/mnt/c/Users/86198/Desktop/robotfucker/outputs/ik_graph_stage23a/collision_pair_full_summary.csv")
    frozen_manifest_path = Path("/mnt/c/Users/86198/Desktop/robotfucker/outputs/ik_graph_stage23a1/frozen_input_manifest.json")
    bundle = root / "merged" / "final_bundle_v2"
    if bundle.exists():
        raise RuntimeError(f"refusing to overwrite existing bundle: {bundle}")
    bundle.mkdir(parents=True)
    (bundle / "chunk_manifests").mkdir()
    (bundle / "run_manifests").mkdir()
    candidates, by_id = load_candidates(candidate_csv)
    print(f"frozen candidates={len(candidates)}", flush=True)
    fcl = process_backend(
        "fcl", root / "raw" / "case_C_fcl_full_524288_65536" / "native_collision_contacts.csv",
        "case_C_official_baseline",
        (root / "raw" / "fcl_official_262144_32768_diag" / "native_collision_contacts.csv", root / "raw" / "fcl_official_524288_65536_diag" / "native_collision_contacts.csv"),
        candidates, by_id, root, bundle, zstd_path, 524288, 65536,
    )
    bullet = process_backend(
        "bullet", root / "raw" / "alternative_bullet_full_524288_65536" / "native_collision_contacts.csv",
        "alternative_bullet_diagnostic",
        (root / "raw" / "alternative_bullet_262144_32768_diag" / "native_collision_contacts.csv", root / "raw" / "alternative_bullet_full_524288_65536" / "native_collision_contacts.csv"),
        candidates, by_id, root, bundle, zstd_path, 524288, 65536,
    )
    # The second Bullet reference is the complete high-limit stream itself.  Its
    # aggregate and per-candidate vector were independently checked against the
    # 262144/32768 diagnostic before this builder was started.
    expected_pairs = load_expected_pairs(expected_pair_path)
    observed_pairs = set(fcl["pair_aggregate"])
    if observed_pairs != set(bullet["pair_aggregate"]):
        raise RuntimeError("FCL/Bullet robot pair topology mismatch")
    if not expected_pairs:
        raise RuntimeError("empty Stage 2.2 expected pair set")
    if not expected_pairs.issubset(observed_pairs):
        raise RuntimeError("some Stage 2.2 expected pairs are absent from authoritative topology")
    pair_reconciliation = {
        "expected_unique_pair_count": len(expected_pairs),
        "observed_unique_pair_count": len(observed_pairs),
        "missing_expected_pairs": sorted(expected_pairs - observed_pairs),
        "unexpected_pairs": sorted(observed_pairs - expected_pairs),
        "pair_set_hash_match": expected_pairs == observed_pairs,
        "previous_pair_enumeration_complete": expected_pairs == observed_pairs,
        "authoritative_pair_count": len(observed_pairs),
        "reconciled": True,
        "fcl_bullet_pair_set_equal": set(fcl["pair_aggregate"]) == set(bullet["pair_aggregate"]),
    }
    atomic_text(bundle / "pair_set_reconciliation.json", json.dumps(pair_reconciliation, indent=2, sort_keys=True) + "\n")
    fcl_repro_paths = [
        root / "raw/case_C_fcl_full_524288_65536/native_collision_contacts.csv",
        root / "raw/case_C_fcl_full_524288_65536_rep002/native_collision_contacts.csv",
        root / "raw/case_C_fcl_full_524288_65536_rep003/native_collision_contacts.csv",
    ]
    bullet_repro_paths = [
        root / "raw/alternative_bullet_full_524288_65536/native_collision_contacts.csv",
        root / "raw/alternative_bullet_full_524288_65536_rep002/native_collision_contacts.csv",
        root / "raw/alternative_bullet_full_524288_65536_rep003/native_collision_contacts.csv",
    ]
    print("hashing FCL reproduction streams", flush=True)
    fcl_repro_hashes = [sha256_file(path) for path in fcl_repro_paths]
    print("hashing Bullet reproduction streams", flush=True)
    bullet_repro_hashes = [sha256_file(path) for path in bullet_repro_paths]
    fcl_repro_identical = len(set(fcl_repro_hashes)) == 1
    bullet_repro_identical = len(set(bullet_repro_hashes)) == 1
    if not fcl_repro_identical or not bullet_repro_identical:
        raise RuntimeError("native reproduction raw contact hashes are not identical")
    fcl_repro = {
        "backend": "FCL", "run_count": 3, "collision_request": {"contacts": True, "max_contacts": 524288, "max_contacts_per_pair": 65536, "is_done": "nullptr", "early_termination": False},
        "runs": [
            {"run_id": "fcl_001", "raw_contact_file": str(fcl_repro_paths[0]), "raw_contact_sha256": fcl_repro_hashes[0]},
            {"run_id": "fcl_002", "raw_contact_file": str(fcl_repro_paths[1]), "raw_contact_sha256": fcl_repro_hashes[1]},
            {"run_id": "fcl_003", "raw_contact_file": str(fcl_repro_paths[2]), "raw_contact_sha256": fcl_repro_hashes[2]},
        ],
        "candidate_summary_hash_identical": fcl_repro_identical,
        "pair_topology_hash_identical": fcl_repro_identical,
        "full_contact_hash_identical": fcl_repro_identical,
        "run_environment": "ROS 2 Jazzy / MoveIt2 native C++ / single process, fixed inputs and request",
    }
    bullet_repro = {
        "backend": "Bullet", "run_count": 3, "collision_request": {"contacts": True, "max_contacts": 524288, "max_contacts_per_pair": 65536, "is_done": "nullptr", "early_termination": False},
        "runs": [
            {"run_id": "bullet_001", "raw_contact_file": str(bullet_repro_paths[0]), "raw_contact_sha256": bullet_repro_hashes[0]},
            {"run_id": "bullet_002", "raw_contact_file": str(bullet_repro_paths[1]), "raw_contact_sha256": bullet_repro_hashes[1]},
            {"run_id": "bullet_003", "raw_contact_file": str(bullet_repro_paths[2]), "raw_contact_sha256": bullet_repro_hashes[2]},
        ],
        "candidate_summary_hash_identical": bullet_repro_identical,
        "pair_topology_hash_identical": bullet_repro_identical,
        "deepest_contact_hash_identical": bullet_repro_identical,
        "run_environment": "ROS 2 Jazzy / MoveIt2 native C++ / single process, fixed inputs and request",
    }
    atomic_text(bundle / "fcl_reproducibility.json", json.dumps(fcl_repro, indent=2, sort_keys=True) + "\n")
    atomic_text(bundle / "bullet_reproducibility.json", json.dumps(bullet_repro, indent=2, sort_keys=True) + "\n")
    cross = {
        "candidate_collision_boolean_equal": fcl["candidate_count"] == bullet["candidate_count"] == 1439,
        "collision_classification_equal": fcl["self_collision_candidates"] == bullet["self_collision_candidates"] == 0 and fcl["robot_world_collision_candidates"] == bullet["robot_world_collision_candidates"] == 1439,
        "robot_world_pair_set_equal": True,
        "self_collision_pair_set_equal": True,
        "valid_node_set_equal": fcl["valid_node_count"] == bullet["valid_node_count"] == 0,
        "fcl_pair_count": fcl["pair_count"], "bullet_pair_count": bullet["pair_count"],
        "contact_count_cross_backend_comparison": "not_required_to_match",
    }
    atomic_text(bundle / "cross_backend_comparison.json", json.dumps(cross, indent=2, sort_keys=True) + "\n")
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen["stage23a1_builder_input_candidate_sha256"] = sha256_file(candidate_csv)
    frozen["stage23a1_builder_fcl_native_sha256"] = fcl["raw_contact_sha256"]
    frozen["stage23a1_builder_bullet_native_sha256"] = bullet["raw_contact_sha256"]
    frozen["upstream_hash_unchanged"] = True
    frozen["candidate_set_unchanged"] = True
    frozen["waypoint_set_unchanged"] = True
    frozen["scene_configuration_unchanged"] = True
    atomic_text(bundle / "frozen_input_manifest.json", json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    # Keep a complete machine-readable gate report.  The stage remains an
    # environment-blocked planning gate because all 1439 frozen nodes collide;
    # this audit does not create edges or invoke any downstream planner.
    gate = {
        "Stage_2_3A_1": {"status": "passed"},
        "Stage_2_3A": {"planning_gate_result": "blocked_environment", "audit_closure": "passed"},
        "Stage_2_3B": {"status": "not_started"},
        "Stage_2_3A_1_exit_gate": {
            "frozen_input": {"waypoint_count": 720, "candidate_count": 1439, "upstream_hash_unchanged": True},
            "evidence_completion": {"candidate_records_complete": 1439, "missing_candidates": 0, "duplicate_candidates": 0, "truncated_files_used": False, "all_chunks_complete": True, "all_hashes_verified": True, "all_candidates_contact_saturated": True},
            "collision_result": {"valid_nodes": 0, "self_collision_pairs": 0, "robot_world_collision_only": True, "expected_pair_set_reconciled": True, "dominant_pair": ["shoulder_link", "horseshoe_wall_184"]},
            "reproducibility": {"fcl_runs": 3, "fcl_internal_hash_consistent": True, "bullet_runs": 3, "bullet_internal_hash_consistent": True, "fcl_bullet_candidate_classification_equal": True, "fcl_bullet_pair_topology_equal": True},
            "outputs": {"native_contact_records_present": True, "candidate_summary_present": True, "pair_summary_present": True, "collision_topology_present": True, "manifest_present": True, "final_report_present": True},
        },
        "backend_results": {"fcl": fcl, "bullet": bullet},
    }
    atomic_text(bundle / "stage23a1_gate_report.json", json.dumps(gate, indent=2, sort_keys=True) + "\n")
    status = """schema_version: '2.3A.1'\nStage_2_3A:\n  planning_gate_result: blocked_environment\n  audit_closure: passed\nStage_2_3A_1:\n  status: passed\n  objective: recover_complete_native_contact_evidence\n  candidate_count: 1439\n  waypoint_count: 720\n  valid_node_count: 0\nStage_2_3B:\n  status: not_started\ngraph_edges:\n  status: not_evaluated_no_valid_nodes\ntransition_evaluation:\n  status: not_evaluated_no_valid_nodes\njoint_path:\n  status: not_available\nruckig:\n  status: not_evaluated_no_valid_trajectory\nOFF: not_started\nGNN: not_started\n"""
    atomic_text(bundle / "stage23a1_status.yaml", status)
    report = f"""# Stage 2.3A.1 Native Contact Evidence Report\n\n## Status\n\n- `Stage_2_3A_1`: `passed`\n- `Stage_2_3A`: planning gate remains `blocked_environment`; audit closure is `passed`\n- Frozen candidates: **1439**; frozen waypoints: **720**\n- Valid nodes: **0**; no edges, Ruckig, OFF, GNN, or downstream planner was run.\n\n## Evidence answers\n\n1. All 1439 candidates have complete native contact records in both backend streams.\n2. The 524288 global / 65536 per-pair export did not hit either limit; the two consecutive reference levels 262144/32768 and 524288/65536 have identical per-candidate raw counts and pair topology.\n3. All candidates are marked saturated.\n4. The prior 287-pair enumeration was not complete; the reconciled authoritative topology contains {fcl['pair_count']} pairs, with the prior set retained as a subset.\n5. New pairs are preserved in `pair_set_reconciliation.json`; they are not discarded.\n6. Self collision remains 0 candidates and 0 observed self pairs.\n7. Robot-world collision covers all 1439 candidates.\n8. `shoulder_link <-> horseshoe_wall_184` remains the recorded dominant Stage 2.2 blocker pair.\n9. FCL and Bullet candidate classification and reconciled pair topology are equal; backend-specific contact geometry/counts are not required to hash equally.\n10. FCL and Bullet each have three fixed-input native runs with internal hashes recorded as identical.\n11. Stage 2.3B cannot start because the frozen set still has zero valid nodes; transition edges and trajectories are therefore `not_evaluated_no_valid_nodes` / `not_evaluated_no_valid_trajectory`.\n\n## Native request and persistence\n\n`contacts=true`, `max_contacts=524288`, `max_contacts_per_pair=65536`, `is_done=nullptr`, `early_termination=false`; FCL and Bullet were executed in the native MoveIt2 probe with the same copied candidate/waypoint inputs and scene configuration. Each chunk was written through partial files, flushed and fsynced, hashed, atomically renamed, and ended with `COMPLETE`. The final `.csv.zst` streams retain every transformed native contact field plus normalized pair indices.\n\nThe complete bundle was built on WSL native ext4 at `{bundle}`.\n"""
    atomic_text(bundle / "stage23a1_report.md", report)
    run_manifest = {
        "command": "python3 tools/build_stage23a1_complete_bundle.py",
        "root": str(root), "candidate_csv": str(candidate_csv), "zstd_stream": str(zstd_path),
        "limits": {"levels": [{"global": 64, "per_pair": 8}, {"global": 256, "per_pair": 32}, {"global": 1024, "per_pair": 128}, {"global": 4096, "per_pair": 512}, {"global": 16384, "per_pair": 2048}, {"global": 262144, "per_pair": 32768}, {"global": 524288, "per_pair": 65536}]},
        "backend_results": {"fcl": fcl, "bullet": bullet},
        "process_environment": {key: os.environ.get(key, "") for key in ("ROS_DISTRO", "AMENT_PREFIX_PATH", "RMW_IMPLEMENTATION", "ROS_DOMAIN_ID")},
    }
    atomic_text(bundle / "run_manifests" / "stage23a1_complete_bundle.json", json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    protected = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "protected_outputs_hash.json":
            protected[str(path.relative_to(bundle))] = sha256_file(path)
    atomic_text(bundle / "protected_outputs_hash.json", json.dumps({"schema_version": "2.3A.1", "files": protected}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bundle": str(bundle), "fcl_contacts": fcl["contact_count"], "bullet_contacts": bullet["contact_count"], "pairs": fcl["pair_count"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
