#!/usr/bin/env python3
"""Phase A exact-duplicate discovery and path-preserving hard-link actions.

This tool deliberately operates only on byte-for-byte identical JSONL files.
It never copies payloads.  Discovery uses a size prefilter, then full SHA-256
for only same-size candidates.  Hard-link replacement uses same-directory
rename/link steps so rollback needs names only, not a byte backup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(r"D:\robotfucker")
WORK = ROOT / "phase_a_exact_dedup_20260902"
EXCLUDED_PARTS = (
    "outputs\\D47_STAGE4B_ROLLING_CHAMPION_V1\\PROMOTED\\B3",
    "outputs\\D64_PROJECT_FREEZE",
    "outputs\\D65_FINAL_CLOSURE",
)
PROTECTED_FLOOR_TOKEN = "c4_clearance_adversarial0101_amp005_consistent_0025"
STAGE5A_TOKEN = "stage5a"
STAGE5A_METADATA_SUFFIXES = {
    ".csv",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
STAGE5A_MANIFEST_NAME = "STAGE5A_DEDUP_EXCLUSION_MANIFEST.jsonl"
STAGE5A_MANIFEST_SUMMARY_NAME = "STAGE5A_DEDUP_EXCLUSION_MANIFEST_SUMMARY.json"
STAGE5A_VIOLATION_NAME = "STAGE5A_PROTECTION_VIOLATION.json"
STAGE5A_MANIFEST_PATH = WORK / STAGE5A_MANIFEST_NAME
STAGE5A_MANIFEST_SUMMARY_PATH = WORK / STAGE5A_MANIFEST_SUMMARY_NAME
STAGE5A_REFERENCE_METADATA_TOKENS = (
    "archive",
    "authority",
    "input",
    "ledger",
    "manifest",
    "provenance",
    "report",
    "state",
    "summary",
)
STAGE5A_REFERENCE_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/mnt/[A-Za-z]/|(?:outputs|tmp|tools|scripts|src|tests|stage5a_handoff_staging_20260902)[\\/])[^\s\"'`,;:=()\[\]{}<>]+"
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def norm(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\").lower()


def excluded(path: Path) -> bool:
    n = norm(path)
    return any(part.lower() in n for part in EXCLUDED_PARTS) or PROTECTED_FLOOR_TOKEN in n


def is_stage5a_path(path: Path) -> bool:
    """Return whether a path belongs to the immutable Stage5A exclusion box."""
    n = norm(path)
    # The exclusion manifest is control metadata outside Stage5A itself.  Its
    # filename necessarily contains the token but must not freeze itself.
    if n in {norm(STAGE5A_MANIFEST_PATH), norm(STAGE5A_MANIFEST_SUMMARY_PATH), norm(WORK / STAGE5A_VIOLATION_NAME)}:
        return False
    return STAGE5A_TOKEN in n


def load_stage5a_manifest() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    if not STAGE5A_MANIFEST_PATH.exists():
        return entries
    with STAGE5A_MANIFEST_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                item = json.loads(line)
                entries[norm(Path(item["path"]))] = item
            except (KeyError, json.JSONDecodeError):
                continue
    return entries


def normalize_reference(raw: str) -> Path | None:
    value = raw.strip().strip(" \t[]{}()<>;:").replace("\\\\", "\\")
    lower = value.lower()
    if lower.startswith("/mnt/d/robotfucker/"):
        value = "D:/robotfucker/" + value[len("/mnt/d/robotfucker/"):].replace("\\", "/")
    elif re.match(r"^[A-Za-z]:[\\/]", value):
        value = value.replace("\\", "/")
    elif lower.startswith(("outputs/", "outputs\\", "tmp/", "tmp\\", "tools/", "tools\\", "scripts/", "scripts\\", "src/", "src\\", "tests/", "tests\\", "stage5a_handoff_staging_20260902/", "stage5a_handoff_staging_20260902\\")):
        value = str(ROOT / value.replace("\\", "/"))
    else:
        return None
    candidate = Path(value)
    try:
        return candidate.resolve() if candidate.exists() else None
    except OSError:
        return None


def iter_stage5a_files():
    """Use rg's path index and filter complete Stage5A directory subtrees."""
    try:
        proc = subprocess.run(
            [
                "rg",
                "--files",
                "--hidden",
                "--no-ignore-vcs",
                ".",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        for raw in proc.stdout.splitlines():
            path = (ROOT / raw).resolve()
            if path.is_file() and is_stage5a_path(path):
                yield path
        return
    except OSError:
        pass
    # Bounded fallback for environments without rg.  This is intentionally
    # fail-closed and only emits files whose path itself contains Stage5A.
    for path in iter_files():
        if is_stage5a_path(path):
            yield path


def stage5a_related_files() -> tuple[dict[str, set[str]], list[str]]:
    """Find Stage5A paths plus files explicitly referenced by Stage5A metadata."""
    reasons: dict[str, set[str]] = defaultdict(set)
    unresolved: set[str] = set()
    all_files = list(iter_stage5a_files())
    for path in all_files:
        if is_stage5a_path(path):
            reasons[norm(path)].add("STAGE5A_FROZEN_EXCLUSION")

    # Search only small metadata files already inside the Stage5A path box.
    # JSONL payloads are included above but are deliberately not read as text.
    for path in all_files:
        if (
            not is_stage5a_path(path)
            or path.suffix.lower() not in STAGE5A_METADATA_SUFFIXES
            or path.suffix.lower() == ".jsonl"
            or not any(token in path.stem.lower() for token in STAGE5A_REFERENCE_METADATA_TOKENS)
        ):
            continue
        try:
            if path.stat().st_size > 64 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in STAGE5A_REFERENCE_RE.finditer(text):
            raw = match.group(0).rstrip(" \t\r\n'\".,;:)]}")
            reference = normalize_reference(raw)
            if reference is None:
                if raw.startswith(("D:", "d:", "/mnt/d/", "outputs", "outputs\\", "tmp", "tmp\\")):
                    # Runtime command fragments and non-file directories are
                    # not payloads.  Keep only unresolved file-like refs in
                    # the manifest summary so the ledger stays actionable.
                    if Path(raw.rstrip("/\\")).suffix or "launch_params_" not in raw:
                        unresolved.add(raw)
                continue
            if reference.is_file() and not is_stage5a_path(reference):
                reasons[norm(reference)].add("EXPLICIT_STAGE5A_METADATA_REFERENCE")
    return reasons, sorted(unresolved)


def create_stage5a_manifest() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    started = now()
    reasons, unresolved = stage5a_related_files()
    previous = load_stage5a_manifest()
    entries: list[dict] = []
    cache_hits = 0
    recomputed = 0
    failures: list[dict] = []
    for key in sorted(reasons):
        path = Path(key)
        try:
            before = path.stat()
            cached = previous.get(key)
            if cached and cached.get("size_bytes") == before.st_size and cached.get("mtime_ns") == before.st_mtime_ns and cached.get("stable"):
                digest = cached["sha256"]
                identity = cached.get("file_identity")
                stable = True
                cache_hits += 1
            else:
                hashed = sha256_file(path)
                digest = hashed["sha256"]
                identity = file_id(path)
                stable = bool(hashed["stable"])
                recomputed += 1
            after = path.stat()
            stable = stable and before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
            if not stable:
                failures.append({"path": str(path), "reason": "changed_during_manifest_hash"})
            entries.append(
                {
                    "path": str(path),
                    "size_bytes": before.st_size,
                    "sha256": digest,
                    "mtime_ns": before.st_mtime_ns,
                    "file_identity_stat": {"st_dev": before.st_dev, "st_ino": before.st_ino},
                    "file_identity": identity,
                    "reason": ";".join(sorted(reasons[key])),
                    "stable": stable,
                }
            )
        except Exception as exc:  # noqa: BLE001 - persisted as fail-closed evidence
            failures.append({"path": str(path), "reason": type(exc).__name__, "detail": str(exc)})

    with STAGE5A_MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema": "stage5a-dedup-exclusion-manifest-v1",
        "generated_at": now(),
        "started_at": started,
        "root": str(ROOT),
        "policy": "STAGE5A_FROZEN_EXCLUSION; read/hash/stat only; zero authorized mutation",
        "file_count": len(entries),
        "logical_bytes": sum(item["size_bytes"] for item in entries),
        "logical_gib": sum(item["size_bytes"] for item in entries) / 2**30,
        "hash_cache_hits": cache_hits,
        "hashes_recomputed": recomputed,
        "hash_failures": len(failures),
        "unresolved_reference_count": len(unresolved),
        "unresolved_references": unresolved[:200],
        "manifest_path": str(STAGE5A_MANIFEST_PATH),
        "failure_details": failures[:200],
    }
    write_json(STAGE5A_MANIFEST_SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if not failures else 2


def verify_stage5a_manifest() -> dict:
    entries = load_stage5a_manifest()
    result = {
        "manifest_present": STAGE5A_MANIFEST_PATH.exists(),
        "checked_count": len(entries),
        "mismatches": [],
        "status": "PASS",
    }
    if not result["manifest_present"]:
        result["status"] = "FAIL_STAGE5A_PROTECTION_VIOLATION"
        result["mismatches"].append({"reason": "manifest_missing"})
        return result
    for key, expected in entries.items():
        path = Path(expected["path"])
        try:
            before = path.stat()
            observed = sha256_file(path)
            expected_stat_identity = expected.get("file_identity_stat")
            observed_stat_identity = {"st_dev": before.st_dev, "st_ino": before.st_ino}
            stat_identity_changed = expected_stat_identity is not None and observed_stat_identity != expected_stat_identity
            identity = file_id(path) if stat_identity_changed else expected.get("file_identity")
            if (
                before.st_size != expected.get("size_bytes")
                or before.st_mtime_ns != expected.get("mtime_ns")
                or observed["sha256"] != expected.get("sha256")
                or not observed["stable"]
                or stat_identity_changed
            ):
                result["mismatches"].append(
                    {
                        "path": str(path),
                        "expected_size": expected.get("size_bytes"),
                        "observed_size": before.st_size,
                        "expected_mtime_ns": expected.get("mtime_ns"),
                        "observed_mtime_ns": before.st_mtime_ns,
                        "expected_sha256": expected.get("sha256"),
                        "observed_sha256": observed["sha256"],
                        "expected_file_identity": expected.get("file_identity"),
                        "observed_file_identity": identity,
                        "expected_file_identity_stat": expected_stat_identity,
                        "observed_file_identity_stat": observed_stat_identity,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - fail closed on any unreadable frozen file
            result["mismatches"].append({"path": str(path), "reason": type(exc).__name__, "detail": str(exc)})
    if result["mismatches"]:
        result["status"] = "FAIL_STAGE5A_PROTECTION_VIOLATION"
    return result


def iter_files(suffix: str | None = None):
    """Yield regular files without following directory symlinks/junctions."""
    stack = [ROOT]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            path = Path(entry.path)
                            if suffix is None or path.suffix.lower() == suffix.lower():
                                yield path
                    except OSError:
                        continue
        except OSError:
            continue


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> dict:
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    after = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": before.st_size,
        "sha256": h.hexdigest(),
        "mtime_ns_before": before.st_mtime_ns,
        "mtime_ns_after": after.st_mtime_ns,
        "size_bytes_after": after.st_size,
        "stable": before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_hash_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    path = WORK / "candidate_hashes.jsonl"
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                item = json.loads(line)
                cache[norm(Path(item["path"]))] = item
            except (KeyError, json.JSONDecodeError):
                continue
    return cache


def discovery() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not STAGE5A_MANIFEST_PATH.exists():
        print(json.dumps({"status": "FAIL_STAGE5A_PROTECTION_VIOLATION", "reason": "stage5a_exclusion_manifest_missing"}), flush=True)
        return 2
    started = now()
    all_files = list(iter_files())
    jsonl = [p for p in all_files if p.suffix.lower() == ".jsonl"]
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in jsonl:
        try:
            by_size[path.stat().st_size].append(path)
        except OSError:
            pass
    size_candidates = [p for paths in by_size.values() if len(paths) > 1 for p in paths]
    size_groups = [paths for paths in by_size.values() if len(paths) > 1]

    cache = load_hash_cache()
    hashes: list[dict] = []
    failures: list[dict] = []
    cache_hits = 0
    to_hash: list[Path] = []
    for path in size_candidates:
        try:
            stat = path.stat()
            cached = cache.get(norm(path))
            if cached and cached.get("stable") and cached.get("size_bytes") == stat.st_size and cached.get("size_bytes_after") == stat.st_size and cached.get("mtime_ns_before") == stat.st_mtime_ns and cached.get("mtime_ns_after") == stat.st_mtime_ns:
                hashes.append(cached)
                cache_hits += 1
            else:
                to_hash.append(path)
        except OSError:
            to_hash.append(path)
    completed = 0
    print(f"{started} jsonl_files={len(jsonl)} same_size_groups={len(size_groups)} candidates={len(size_candidates)} cache_hits={cache_hits} hash_required={len(to_hash)}", flush=True)
    workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sha256_file, path): path for path in to_hash}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                if result["stable"]:
                    hashes.append(result)
                else:
                    failures.append({"path": str(path.resolve()), "reason": "changed_during_hash", **result})
            except Exception as exc:  # noqa: BLE001 - persisted as fail-closed evidence
                failures.append({"path": str(path.resolve()), "reason": type(exc).__name__, "detail": str(exc)})
            completed += 1
            if completed % 250 == 0 or completed == len(to_hash):
                print(f"hashed={completed}/{len(to_hash)} failures={len(failures)}", flush=True)

    by_sha: dict[str, list[dict]] = defaultdict(list)
    for item in hashes:
        by_sha[item["sha256"]].append(item)
    stage5a_manifest_keys = set(load_stage5a_manifest())
    groups = []
    for digest, members in by_sha.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda item: item["path"].lower())
        stage5a_paths = [item["path"] for item in members if is_stage5a_path(Path(item["path"])) or norm(Path(item["path"])) in stage5a_manifest_keys]
        groups.append(
            {
                "sha256": digest,
                "size_bytes": members[0]["size_bytes"],
                "count": len(members),
                "paths": [item["path"] for item in members],
                "stage5a_protection": bool(stage5a_paths),
                "stage5a_paths": stage5a_paths,
            }
        )
    groups.sort(key=lambda group: (-group["size_bytes"], group["sha256"]))
    duplicate_rows = sum(group["count"] for group in groups)
    reclaim = sum(group["size_bytes"] * (group["count"] - 1) for group in groups)
    stage5a_groups = [group for group in groups if group["stage5a_protection"]]
    authorized_groups = [group for group in groups if not group["stage5a_protection"] and not any(excluded(Path(path)) for path in group["paths"])]
    frozen_reclaim = sum(group["size_bytes"] * (group["count"] - 1) for group in stage5a_groups)
    authorized_reclaim = sum(group["size_bytes"] * (group["count"] - 1) for group in authorized_groups)
    candidate_bytes = sum(item["size_bytes"] for item in hashes)
    totals_bytes = 0
    stat_failures = 0
    unique_physical_project: dict[tuple[int, int], int] = {}
    project_stat_failure_paths: list[str] = []
    for item in all_files:
        try:
            stat = item.stat()
            totals_bytes += stat.st_size
            unique_physical_project.setdefault((stat.st_dev, stat.st_ino), stat.st_size)
        except OSError:
            stat_failures += 1
            if len(project_stat_failure_paths) < 100:
                project_stat_failure_paths.append(str(item))
    jsonl_bytes = 0
    for item in jsonl:
        try:
            jsonl_bytes += item.stat().st_size
        except OSError:
            stat_failures += 1
    unique_physical_jsonl: dict[tuple[int, int], int] = {}
    unique_jsonl_stat_failures = 0
    for item in jsonl:
        try:
            stat = item.stat()
            unique_physical_jsonl.setdefault((stat.st_dev, stat.st_ino), stat.st_size)
        except OSError:
            unique_jsonl_stat_failures += 1

    with (WORK / "candidate_hashes.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for item in sorted(hashes, key=lambda x: x["path"].lower()):
            fh.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (WORK / "hash_failures.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for item in failures:
            fh.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (WORK / "exact_groups.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for group in groups:
            fh.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "schema": "phase-a-exact-dedup-discovery-v1",
        "generated_at": now(),
        "root": str(ROOT),
        "filesystem": "NTFS",
        "scope": "current regular .jsonl files; same-size prefilter then full SHA-256",
        "all_files_stat_count": len(all_files),
        "stat_failures_fail_closed": stat_failures,
        "project_logical_bytes": totals_bytes,
        "project_logical_gib": totals_bytes / 2**30,
        "unique_physical_project_file_count": len(unique_physical_project),
        "unique_physical_project_bytes": sum(unique_physical_project.values()),
        "unique_physical_project_gib": sum(unique_physical_project.values()) / 2**30,
        "project_stat_failure_paths_sample": project_stat_failure_paths,
        "jsonl_file_count": len(jsonl),
        "jsonl_bytes": jsonl_bytes,
        "jsonl_gib": jsonl_bytes / 2**30,
        "unique_physical_jsonl_file_count": len(unique_physical_jsonl),
        "unique_physical_jsonl_bytes": sum(unique_physical_jsonl.values()),
        "unique_physical_jsonl_gib": sum(unique_physical_jsonl.values()) / 2**30,
        "unique_physical_jsonl_identity_method": "os.stat(st_dev,st_ino)_NTFS_file_index_cross_check",
        "jsonl_stat_failures": unique_jsonl_stat_failures,
        "same_size_groups": len(size_groups),
        "size_candidate_files": len(size_candidates),
        "hash_cache_hits": cache_hits,
        "hashes_recomputed": len(to_hash),
        "hashed_stable_files": len(hashes),
        "hash_failures": len(failures),
        "exact_duplicate_groups": len(groups),
        "exact_duplicate_files": duplicate_rows,
        "exact_duplicate_content_bytes": sum(group["size_bytes"] * group["count"] for group in groups),
        "exact_duplicate_content_gib": sum(group["size_bytes"] * group["count"] for group in groups) / 2**30,
        "theoretical_reclaim_bytes": reclaim,
        "theoretical_reclaim_gib": reclaim / 2**30,
        "stage5a_frozen_duplicate_groups": len(stage5a_groups),
        "stage5a_frozen_theoretical_reclaim_bytes": frozen_reclaim,
        "stage5a_frozen_theoretical_reclaim_gib": frozen_reclaim / 2**30,
        "stage5a_actual_reclaim_bytes": 0,
        "stage5a_mutation_count": 0,
        "current_authorized_groups": len(authorized_groups),
        "current_authorized_theoretical_reclaim_bytes": authorized_reclaim,
        "current_authorized_theoretical_reclaim_gib": authorized_reclaim / 2**30,
        "candidate_hash_bytes": candidate_bytes,
        "excluded_path_rules": list(EXCLUDED_PARTS) + [PROTECTED_FLOOR_TOKEN],
        "stage5a_exclusion_manifest": str(STAGE5A_MANIFEST_PATH),
        "artifacts": ["candidate_hashes.jsonl", "hash_failures.jsonl", "exact_groups.jsonl"],
    }
    write_json(WORK / "discovery_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 2


def load_plan_tiers() -> dict[str, set[str]]:
    tiers: dict[str, set[str]] = defaultdict(set)
    plan = ROOT / "hardlink_dedup_plan.csv"
    if not plan.exists():
        return tiers
    with plan.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("sha256") and row.get("tier"):
                tiers[row["sha256"]].add(row["tier"])
    return tiers


def classify_groups() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    groups = load_groups()
    plan_tiers = load_plan_tiers()
    actions: dict[str, dict] = {}
    for action_path in WORK.glob("action_*.json"):
        try:
            action = json.loads(action_path.read_text(encoding="utf-8"))
            actions[action["sha256"]] = {"status": action.get("status"), "artifact": str(action_path)}
        except (OSError, KeyError, json.JSONDecodeError):
            continue

    rows: list[dict] = []
    for sha, group in sorted(groups.items()):
        file_ids: set[tuple[int, int]] = set()
        group_stat_failures = 0
        for raw_path in group["paths"]:
            try:
                stat = Path(raw_path).stat()
                file_ids.add((stat.st_dev, stat.st_ino))
            except OSError:
                group_stat_failures += 1
        current_physical_reclaim = group["size_bytes"] * max(0, len(file_ids) - 1) if not group_stat_failures else None
        action = actions.get(sha)
        if action and action["status"] in {"COMMITTED", "FINALIZED"}:
            status = "CLOSED"
            reason = "FIRST_SAFE_SUBSET_CLOSED_AND_VALIDATED"
        elif group.get("stage5a_protection"):
            status = "EXCLUDED"
            reason = "EXCLUDED_STAGE5A_PROTECTION"
        elif any(excluded(Path(path)) for path in group["paths"]):
            status = "EXCLUDED"
            reason = "EXCLUDED_PROTECTED_CANONICAL_D64_D65_OR_FLOOR"
        elif any(tier == "TIER_0_NOT_SAFE" for tier in plan_tiers.get(sha, set())):
            status = "EXCLUDED"
            reason = "EXCLUDED_EXISTING_PLAN_WRITE_OR_MUTATION_RISK"
        elif any(tier == "TIER_2_LIKELY_SAFE_BUT_REQUIRES_REPLAY" for tier in plan_tiers.get(sha, set())):
            status = "EXCLUDED"
            reason = "EXCLUDED_REPLAY_REQUIRED_BEFORE_HARDLINK"
        elif any(tier == "TIER_3_UNKNOWN" for tier in plan_tiers.get(sha, set())):
            status = "EXCLUDED"
            reason = "EXCLUDED_SAFETY_EVIDENCE_INSUFFICIENT"
        elif not plan_tiers.get(sha):
            status = "EXCLUDED"
            reason = "EXCLUDED_NO_EXISTING_PROVENANCE_CONSUMER_OR_WRITE_GUARD_EVIDENCE"
        else:
            status = "EXCLUDED"
            reason = "EXCLUDED_UNSUPPORTED_PLAN_STATE"
        rows.append(
            {
                "sha256": sha,
                "size_bytes": group["size_bytes"],
                "count": group["count"],
                "theoretical_reclaim_bytes": group["size_bytes"] * (group["count"] - 1),
                "current_unique_file_id_count": len(file_ids),
                "current_physical_reclaim_bytes": current_physical_reclaim,
                "group_stat_failures": group_stat_failures,
                "status": status,
                "reason": reason,
                "stage5a_path_count": len(group.get("stage5a_paths", [])),
                "plan_tiers": sorted(plan_tiers.get(sha, set())),
                "action_status": action["status"] if action else None,
                "action_artifact": action["artifact"] if action else None,
            }
        )
    ledger = WORK / "EXCLUSION_LEDGER.jsonl"
    with ledger.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = defaultdict(lambda: {"groups": 0, "files": 0, "theoretical_reclaim_bytes": 0})
    for row in rows:
        item = counts[row["reason"]]
        item["groups"] += 1
        item["files"] += row["count"]
        item["theoretical_reclaim_bytes"] += row["theoretical_reclaim_bytes"]
    summary = {
        "schema": "phase-a-exact-dedup-group-classification-v1",
        "generated_at": now(),
        "group_count": len(rows),
        "closed_groups": sum(row["status"] == "CLOSED" for row in rows),
        "excluded_groups": sum(row["status"] == "EXCLUDED" for row in rows),
        "stage5a_actual_reclaim_bytes": 0,
        "stage5a_mutation_count": 0,
        "current_physical_reclaim_bytes_for_excluded_or_open_groups": sum(
            row["current_physical_reclaim_bytes"] or 0
            for row in rows
            if row["status"] == "EXCLUDED"
        ),
        "current_physical_reclaim_bytes_for_nonclosed_nonstage5a_groups": sum(
            row["current_physical_reclaim_bytes"] or 0
            for row in rows
            if row["status"] == "EXCLUDED" and row["reason"] != "EXCLUDED_STAGE5A_PROTECTION"
        ),
        "by_reason": dict(sorted(counts.items())),
        "ledger_path": str(ledger),
    }
    write_json(WORK / "EXCLUSION_LEDGER_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def load_groups() -> dict[str, dict]:
    groups = {}
    with (WORK / "exact_groups.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            groups[item["sha256"]] = item
    return groups


def file_id(path: Path) -> dict:
    # fsutil output is small and read-only; file index is useful evidence that
    # replacement changed names only and did not create a payload copy.
    try:
        proc = subprocess.run(["fsutil", "file", "queryfileid", str(path)], capture_output=True, text=True, check=False)
        text = (proc.stdout or proc.stderr).strip().replace("\r", "")
        return {"raw": text, "returncode": proc.returncode}
    except OSError as exc:
        return {"raw": str(exc), "returncode": -1}


def count_lines_and_jsonl(path: Path, full: bool = True) -> tuple[int | str, bool]:
    """Check JSONL without needlessly parsing every record in huge duplicates.

    Full line count and JSON parsing are performed for one group member.  For
    the remaining members, exact SHA-256 equality already proves byte-for-byte
    equality (and therefore identical line count), while first/last blocks
    provide a cheap readability check at each path.
    """
    try:
        if full:
            lines = 0
            last_byte = b""
            with path.open("rb", buffering=0) as fh:
                while True:
                    block = fh.read(8 * 1024 * 1024)
                    if not block:
                        break
                    lines += block.count(b"\n")
                    last_byte = block[-1:]
            if path.stat().st_size and last_byte != b"\n":
                lines += 1
        else:
            lines = "byte_identity_inherited"
        with path.open("rb") as fh:
            head = fh.read(128 * 1024)
            if path.stat().st_size > 128 * 1024:
                fh.seek(max(0, path.stat().st_size - 128 * 1024))
                tail = fh.read(128 * 1024)
            else:
                tail = head
        samples = [line for line in (head + b"\n" + tail).splitlines() if line.strip()]
        for raw in samples[:20]:
            json.loads(raw.decode("utf-8"))
        return lines, True
    except Exception:
        return 0, False


def consumer_hints(group: dict) -> list[str]:
    # This is intentionally a bounded source-name probe for the requested
    # group, not a project-wide content audit.  Exact full-path references are
    # handled by the caller's group ledger and remain a review gate.
    hints: list[str] = []
    basenames = sorted({Path(p).name for p in group["paths"]})
    source_roots = [ROOT / "tools", ROOT / "scripts", ROOT / "src", ROOT / "tests", ROOT / "ros2_moveit_bridge", ROOT / "README.md"]
    for base in basenames:
        try:
            proc = subprocess.run(
                ["rg", "-n", "-F", base, *[str(p) for p in source_roots], "--glob", "*.py", "--glob", "*.cpp", "--glob", "*.h", "--glob", "*.sh", "--glob", "*.md"],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = [line for line in proc.stdout.splitlines() if line]
            hints.extend(lines[:30])
        except OSError:
            hints.append(f"rg_unavailable:{base}")
    return hints


def validate_group(group: dict) -> dict:
    paths = [Path(p) for p in group["paths"]]
    stage5a_manifest = load_stage5a_manifest()
    stage5a_paths = [str(path) for path in paths if is_stage5a_path(path) or norm(path) in stage5a_manifest]
    result = {
        "sha256": group["sha256"],
        "size_bytes": group["size_bytes"],
        "count": group["count"],
        "paths": group["paths"],
        "excluded_protected_path": any(excluded(p) for p in paths),
        "excluded_stage5a_path": bool(stage5a_paths),
        "stage5a_paths": stage5a_paths,
        "all_regular": True,
        "all_same_size": True,
        "pre_hashes": {},
        "hash_stable": {},
        "line_counts": {},
        "jsonl_readable": {},
        "file_ids_before": {},
        "consumer_hints": consumer_hints(group),
    }
    for index, path in enumerate(paths):
        try:
            st = path.stat()
            result["all_regular"] &= path.is_file()
            result["all_same_size"] &= st.st_size == group["size_bytes"]
            hashed = sha256_file(path)
            result["pre_hashes"][str(path)] = hashed["sha256"]
            result["hash_stable"][str(path)] = hashed["stable"]
            lines, readable = count_lines_and_jsonl(path, full=index == 0)
            result["line_counts"][str(path)] = lines
            result["jsonl_readable"][str(path)] = readable
            result["file_ids_before"][str(path)] = file_id(path)
        except Exception as exc:  # noqa: BLE001 - persisted as fail-closed evidence
            result["all_regular"] = False
            result.setdefault("errors", []).append({"path": str(path), "error": type(exc).__name__, "detail": str(exc)})
    result["pre_hashes_match"] = set(result["pre_hashes"].values()) == {group["sha256"]}
    result["all_hashes_stable"] = bool(result["hash_stable"]) and all(result["hash_stable"].values())
    result["line_count_semantics"] = "full_count_for_first_member;_exact_SHA256_proves_duplicate_line_count_equal"
    return result


def replace_with_hardlink(survivor: Path, duplicate: Path) -> dict:
    """Replace duplicate pathname with a hardlink, using name-only rollback."""
    if survivor.resolve() == duplicate.resolve():
        raise ValueError("survivor_equals_duplicate")
    if is_stage5a_path(survivor) or is_stage5a_path(duplicate) or norm(survivor) in load_stage5a_manifest() or norm(duplicate) in load_stage5a_manifest():
        raise RuntimeError("FAIL_STAGE5A_PROTECTION_VIOLATION")
    if survivor.parent != duplicate.parent:
        # Temporary names must stay beside the duplicate for rename atomicity;
        # the hardlink itself is still valid across directories on one volume.
        pass
    token = uuid.uuid4().hex
    temp_link = duplicate.with_name(f".phase_a_link_{token}.tmp")
    old_name = duplicate.with_name(f".phase_a_old_{token}.tmp")
    if temp_link.exists() or old_name.exists():
        raise FileExistsError("temporary_name_collision")
    before_id = file_id(duplicate)
    try:
        os.link(survivor, temp_link)
        os.replace(duplicate, old_name)
        os.replace(temp_link, duplicate)
        # Keep the old name as a zero-copy rollback handle until the complete
        # group validation has passed.  The caller commits its removal.
    except Exception:
        # Best-effort rollback of names; no payload backup exists or is needed.
        try:
            if temp_link.exists():
                temp_link.unlink()
        except OSError:
            pass
        try:
            if old_name.exists() and not duplicate.exists():
                os.replace(old_name, duplicate)
        except OSError:
            pass
        raise
    return {"before_duplicate_file_id": before_id, "after_duplicate_file_id": file_id(duplicate), "temp_link": str(temp_link), "old_name": str(old_name)}


def pilot(args: argparse.Namespace) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    groups = load_groups()
    if args.sha not in groups:
        raise SystemExit(f"unknown group sha: {args.sha}")
    group = groups[args.sha]
    validation = validate_group(group)
    action = {
        "schema": "phase-a-exact-dedup-action-v1",
        "started_at": now(),
        "mode": "hardlink_path_preserving",
        "sha256": args.sha,
        "group": group,
        "pre_validation": validation,
        "status": "REVIEW",
        "operations": [],
    }
    if validation["excluded_protected_path"]:
        action["status"] = "SKIP_PROTECTED"
    elif validation["excluded_stage5a_path"]:
        action["status"] = "EXCLUDED_STAGE5A_PROTECTION"
        action["stage5a_actual_reclaim_bytes"] = 0
        action["stage5a_mutation_count"] = 0
    elif not STAGE5A_MANIFEST_PATH.exists():
        action["status"] = "FAIL_STAGE5A_PROTECTION_VIOLATION"
        action["failure"] = {"type": "missing_manifest", "detail": str(STAGE5A_MANIFEST_PATH)}
    elif (stage5a_pre := verify_stage5a_manifest())["status"] != "PASS":
        action["status"] = "FAIL_STAGE5A_PROTECTION_VIOLATION"
        action["stage5a_pre_validation"] = stage5a_pre
    elif not (validation["all_regular"] and validation["all_same_size"] and validation["pre_hashes_match"] and validation["all_hashes_stable"] and all(validation["jsonl_readable"].values())):
        action["status"] = "SKIP_PRECONDITION"
    elif not args.commit:
        action["status"] = "PILOT_READY_NOT_COMMITTED"
    else:
        survivor = Path(group["paths"][0])
        # Caller must explicitly pass a reviewed survivor index; lexical first
        # is only a deterministic fallback for a pilot and is never canonical
        # promotion logic.
        if args.survivor_index < 0 or args.survivor_index >= len(group["paths"]):
            raise SystemExit("survivor_index_out_of_range")
        survivor = Path(group["paths"][args.survivor_index])
        try:
            for index, raw in enumerate(group["paths"]):
                if index == args.survivor_index:
                    continue
                duplicate = Path(raw)
                op = replace_with_hardlink(survivor, duplicate)
                op.update({"survivor": str(survivor), "duplicate": str(duplicate), "size_bytes": group["size_bytes"]})
                action["operations"].append(op)
            post = validate_group(group)
            action["post_validation"] = post
            valid = post["pre_hashes_match"] and post["all_regular"] and post["all_same_size"] and post["all_hashes_stable"] and all(post["jsonl_readable"].values())
            action["stage5a_post_validation"] = verify_stage5a_manifest()
            if action["stage5a_post_validation"]["status"] != "PASS":
                raise RuntimeError("FAIL_STAGE5A_PROTECTION_VIOLATION")
            if valid:
                # Keep these same-volume renamed names until external
                # consumer tests pass.  They are rollback handles only and
                # contain no second payload copy.
                action["status"] = "COMMITTED_PENDING_EXTERNAL_VALIDATION"
            else:
                raise RuntimeError("failed_post_validation")
        except Exception as exc:
            action["rollback_error"] = None
            action["failure"] = {"type": type(exc).__name__, "detail": str(exc)}
            for op in reversed(action["operations"]):
                duplicate = Path(op["duplicate"])
                old_name = Path(op["old_name"])
                try:
                    if duplicate.exists():
                        duplicate.unlink()
                    if old_name.exists():
                        os.replace(old_name, duplicate)
                except Exception as rollback_exc:  # noqa: BLE001
                    action["rollback_error"] = {"type": type(rollback_exc).__name__, "detail": str(rollback_exc), "duplicate": str(duplicate)}
            action["status"] = "ROLLED_BACK" if action["rollback_error"] is None else "ROLLBACK_FAILED"
            action["post_rollback_validation"] = validate_group(group)
    action["finished_at"] = now()
    out = WORK / f"action_{args.sha[:16]}.json"
    write_json(out, action)
    print(json.dumps({"status": action["status"], "artifact": str(out), "operations": len(action["operations"])}, indent=2))
    return 0 if action["status"] in {"PILOT_READY_NOT_COMMITTED", "COMMITTED", "COMMITTED_PENDING_EXTERNAL_VALIDATION"} else 2


def load_action(sha: str) -> tuple[Path, dict]:
    path = WORK / f"action_{sha[:16]}.json"
    if not path.exists():
        raise SystemExit(f"missing_action:{path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def finalize(args: argparse.Namespace) -> int:
    action_path, action = load_action(args.sha)
    if action.get("status") != "COMMITTED_PENDING_EXTERNAL_VALIDATION":
        raise SystemExit(f"action_not_pending:{action.get('status')}")
    pre_stage5a = verify_stage5a_manifest()
    action["stage5a_finalize_pre_validation"] = pre_stage5a
    if pre_stage5a["status"] != "PASS":
        action["status"] = "FAIL_STAGE5A_PROTECTION_VIOLATION"
        action["finished_at"] = now()
        write_json(action_path, action)
        return 2
    group = action["group"]
    post = validate_group(group)
    action["finalize_validation"] = post
    valid = post["pre_hashes_match"] and post["all_regular"] and post["all_same_size"] and post["all_hashes_stable"] and all(post["jsonl_readable"].values())
    if not valid:
        action["status"] = "PENDING_EXTERNAL_VALIDATION_FAILED_INTERNAL_RECHECK"
        action["finished_at"] = now()
        write_json(action_path, action)
        return 2
    for op in action["operations"]:
        old_name = Path(op["old_name"])
        if old_name.exists():
            old_name.unlink()
    action["stage5a_finalize_post_validation"] = verify_stage5a_manifest()
    action["status"] = "FINALIZED" if action["stage5a_finalize_post_validation"]["status"] == "PASS" else "FAIL_STAGE5A_PROTECTION_VIOLATION"
    action["finished_at"] = now()
    write_json(action_path, action)
    print(json.dumps({"status": action["status"], "artifact": str(action_path), "operations": len(action["operations"])}, indent=2))
    return 0


def rollback(args: argparse.Namespace) -> int:
    action_path, action = load_action(args.sha)
    if action.get("status") != "COMMITTED_PENDING_EXTERNAL_VALIDATION":
        raise SystemExit(f"action_not_pending:{action.get('status')}")
    errors = []
    for op in reversed(action["operations"]):
        duplicate = Path(op["duplicate"])
        old_name = Path(op["old_name"])
        try:
            if duplicate.exists():
                duplicate.unlink()
            if old_name.exists():
                os.replace(old_name, duplicate)
        except Exception as exc:  # noqa: BLE001
            errors.append({"duplicate": str(duplicate), "type": type(exc).__name__, "detail": str(exc)})
    action["status"] = "ROLLED_BACK" if not errors else "ROLLBACK_FAILED"
    action["rollback_errors"] = errors
    action["finished_at"] = now()
    write_json(action_path, action)
    print(json.dumps({"status": action["status"], "artifact": str(action_path), "errors": errors}, indent=2))
    return 0 if not errors else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("discover")
    sub.add_parser("stage5a-manifest")
    sub.add_parser("stage5a-verify")
    sub.add_parser("classify")
    p = sub.add_parser("pilot")
    p.add_argument("--sha", required=True)
    p.add_argument("--survivor-index", type=int, default=0)
    p.add_argument("--commit", action="store_true")
    f = sub.add_parser("finalize")
    f.add_argument("--sha", required=True)
    b = sub.add_parser("rollback")
    b.add_argument("--sha", required=True)
    args = parser.parse_args()
    if args.command == "discover":
        return discovery()
    if args.command == "stage5a-manifest":
        return create_stage5a_manifest()
    if args.command == "stage5a-verify":
        result = verify_stage5a_manifest()
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0 if result["status"] == "PASS" else 2
    if args.command == "classify":
        return classify_groups()
    if args.command == "pilot":
        return pilot(args)
    if args.command == "finalize":
        return finalize(args)
    if args.command == "rollback":
        return rollback(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
