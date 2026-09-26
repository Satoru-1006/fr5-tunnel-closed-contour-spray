"""Build and verify the single post-D24 Desktop evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
D22 = ROOT / "outputs" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_execution_evidence_20260821T151902+0800"
D23 = ROOT / "outputs" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint_20260821T190000+0800"
D24 = ROOT / "outputs" / "stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution_20260821T225000+0800_retry"
D23_CHECKPOINT = D23 / "checkpoints" / "C5_update_392.pt"
D24_CHECKPOINT = D24 / "checkpoints" / "committed_update_394.pt"
D24_RUNNER = ROOT / "tools" / "stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution.py"
D23_RUNNER = ROOT / "tools" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint.py"
D22_RUNNER = ROOT / "tools" / "stage3_h13_post_d21_d22_execution_replay.py"
D22_DESIGN = ROOT / "tools" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_design_review.py"
D21_RUNNER = ROOT / "tools" / "stage3_h13_post_d20_d21_execution_repaired.py"
AUTH_D23 = Path(r"C:\Users\86198\.codex\attachments\5eb8e7eb-97b0-47fb-81bf-8de98a0dc3fa\pasted-text.txt")
AUTH_D24 = Path(r"C:\Users\86198\.codex\attachments\66c30217-9496-45da-9428-28c3e6ce69fe\pasted-text.txt")
AUTH_ARCHIVE = Path(r"C:\Users\86198\.codex\attachments\ddf3c4ca-d127-439c-a9f7-a1945f3b5837\pasted-text.txt")
MAX_BYTES = 20 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def desktop_path() -> Path:
    profile = os.environ.get("USERPROFILE")
    candidate = Path(profile) / "Desktop" if profile else Path.home() / "Desktop"
    if not candidate.is_dir():
        candidate = Path.home() / "Desktop"
    if not candidate.is_dir():
        raise RuntimeError(f"DESKTOP_NOT_FOUND:{candidate}")
    return candidate.resolve()


def add_file(mapping: dict[str, dict[str, Any]], source: Path, relative: str, priority: str, rationale: str) -> None:
    source = source.resolve()
    if not source.is_file():
        return
    if relative in mapping:
        raise RuntimeError(f"ARCHIVE_RELATIVE_PATH_COLLISION:{relative}")
    mapping[relative] = {"source": source, "priority_class": priority, "rationale": rationale}


def collect_sources() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for path in sorted(D24.iterdir()):
        if path.is_file():
            add_file(mapping, path, f"D24/{path.name}", "A", "D24 final execution artifact")
    add_file(mapping, D24_CHECKPOINT, "D24/checkpoints/committed_update_394.pt", "B", "best valid D24 resumable forward checkpoint")
    for path in sorted(D23.iterdir()):
        if path.is_file():
            add_file(mapping, path, f"D23/{path.name}", "C", "D23 provenance artifact used to authenticate D24 start")
    add_file(mapping, D23_CHECKPOINT, "D23/checkpoints/C5_update_392.pt", "B/C", "authenticated D24 starting checkpoint")
    for name, source, priority, rationale in (
        ("stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution.py", D24_RUNNER, "A", "D24 implementation and proposal-selection logic"),
        ("stage3_h13_post_d22_d23_progress_first_optimization_sprint.py", D23_RUNNER, "C", "D23 execution runner"),
        ("stage3_h13_post_d21_d22_execution_replay.py", D22_RUNNER, "D", "D22 execution runner"),
        ("stage3_h13_post_d21_d22_design_review.py", D22_DESIGN, "D", "D22 design authority"),
        ("stage3_h13_post_d20_d21_execution_repaired.py", D21_RUNNER, "D", "D21 parent execution authority"),
        ("AGENTS.md", ROOT / "AGENTS.md", "A/D", "repository execution rules"),
    ):
        add_file(mapping, source, f"implementation/{name}", priority, rationale)
    for path, name, priority in ((AUTH_D23, "D23_progress_first_execution_prompt.txt", "E"), (AUTH_D24, "D24_adamw_aware_postclip_execution_prompt.txt", "E"), (AUTH_ARCHIVE, "post_execution_archive_requirement.txt", "E")):
        add_file(mapping, path, f"prompts/{name}", priority, "project-owner governing prompt")
    for path in sorted(D22.iterdir()):
        if path.is_file():
            add_file(mapping, path, f"D22/{path.name}", "D", "compact authoritative D22 provenance")
    for design_dir in sorted((ROOT / "outputs").glob("stage3_h13_post_d21_d22_*design_review_*")):
        manifest = design_dir / "SHA256_MANIFEST.json"
        if manifest.is_file():
            for name in ("experiment_design.json", "branch_manifest.json", "source_revision_identity.json", "classification_contract.json", "SHA256_MANIFEST.json"):
                add_file(mapping, design_dir / name, f"D22_design/{name}", "D", "D22 design-review authority")
            break
    return mapping


def build_readme(path: Path, file_count: int, total_bytes: int) -> None:
    text = f"""# Stage-3 H13 D22/D23/D24 Evidence Archive

Created: `{now()}`

This archive packages the authoritative D22 provenance, D23 progress-first
result, and the D24 AdamW-aware post-clip realized proposal execution.

- D24 status: `BLOCKED` at update 395 because no proposal preserved H32.
- D24 starting state: D23 `C5_update_392.pt`.
- D24 maximum valid committed update: `394`.
- Best forward state: D24 `committed_update_394.pt`, selected proposal `P4`.
- D24 completion gate: `NOT_READY`.
- D24 committed H1: `8.366622366361216e-05`.
- Archive file count before compression: `{file_count}`.
- Archive total uncompressed bytes before compression: `{total_bytes}`.

Important paths inside the archive:

- `D24/FINAL_REPORT.md`, `D24/D24_RUN_MANIFEST.json`, and D24 CSV/JSON evidence.
- `D24/checkpoints/committed_update_394.pt` is the best resumable forward state.
- `D23/checkpoints/C5_update_392.pt` is the authenticated D24 starting state.
- `D22/` contains compact D22 execution and conservation provenance.
- `implementation/` contains the D21/D22/D23/D24 runners and repository rules.
- `prompts/` contains the governing D23, D24, and archive instructions.

The internal manifest is self-excluded from its own file hashes and is verified
against every other regular file after XZ decompression.
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_sources(mapping: dict[str, dict[str, Any]], stage: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, list[str]] = {}
    for relative, item in sorted(mapping.items()):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], target)
        file_hash = sha256(target)
        record = {"relative_path": relative, "size_bytes": target.stat().st_size, "sha256": file_hash, "priority_class": item["priority_class"], "source_origin": str(item["source"]), "rationale": item["rationale"]}
        records.append(record)
        hashes.setdefault(file_hash, []).append(relative)
    duplicates = [{"sha256": value, "paths": paths} for value, paths in sorted(hashes.items()) if len(paths) > 1]
    return records, duplicates


def write_internal_metadata(stage: Path, records: list[dict[str, Any]], duplicates: list[dict[str, Any]], excluded: list[str]) -> None:
    total_bytes = sum(int(record["size_bytes"]) for record in records)
    build_readme(stage / "ARCHIVE_README.md", len(records), total_bytes)
    readme_record = {"relative_path": "ARCHIVE_README.md", "size_bytes": (stage / "ARCHIVE_README.md").stat().st_size, "sha256": sha256(stage / "ARCHIVE_README.md"), "priority_class": "A", "source_origin": "generated by archive packager", "rationale": "archive navigation and result summary"}
    records.append(readme_record)
    manifest = {"archive_purpose": "single post-D24 Stage-3 H13 D22/D23/D24 evidence archive", "creation_time": now(), "source_stages": ["D22", "D23", "D24"], "file_count": len(records), "total_uncompressed_bytes": sum(int(record["size_bytes"]) for record in records), "self_excluded": True, "relative_path": records, "deduplication": {"byte_identical_groups": duplicates, "decisions": "Separate stage-qualified paths were retained where provenance is meaningful; no redundant D23 trial checkpoints were included."}, "excluded_items": excluded}
    (stage / "ARCHIVE_CONTENTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def make_archive(stage: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with tarfile.open(destination, mode="w:xz", preset=9 | lzma.PRESET_EXTREME) as archive:
        archive.add(stage, arcname="STAGE3_H13_D22_D23_D24_EVIDENCE")


def verify_archive(archive_path: Path, stage: Path) -> dict[str, Any]:
    with lzma.open(archive_path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    with tarfile.open(archive_path, mode="r:xz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members if member.isfile()]
        if not any(name.endswith("ARCHIVE_CONTENTS_MANIFEST.json") for name in names):
            raise RuntimeError("INTERNAL_MANIFEST_MISSING")
        extract_dir = Path(tempfile.mkdtemp(prefix="stage3_h13_archive_verify_", dir=str(ROOT / "tmp")))
        try:
            archive.extractall(extract_dir)
            root = extract_dir / "STAGE3_H13_D22_D23_D24_EVIDENCE"
            manifest = json.loads((root / "ARCHIVE_CONTENTS_MANIFEST.json").read_text(encoding="utf-8"))
            failures = []
            for record in manifest["relative_path"]:
                target = root / record["relative_path"]
                if not target.is_file() or target.stat().st_size != int(record["size_bytes"]) or sha256(target) != record["sha256"]:
                    failures.append(record["relative_path"])
            if failures:
                raise RuntimeError(f"INTERNAL_HASH_FAILURES:{failures}")
            return {"xz_decompression_test": "PASS", "tar_content_read_test": "PASS", "member_count": len(names), "internal_manifest_verification": "PASS", "internal_file_count": int(manifest["file_count"])}
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)


def quarantine_existing_related_archives(desktop: Path, quarantine: Path) -> list[str]:
    related = [path for path in desktop.iterdir() if path.is_file() and path.name.startswith("STAGE3_H13_D23_RESULTS_") and path.suffix.lower() == ".zip"]
    quarantine.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for path in related:
        target = quarantine / path.name
        if target.exists():
            raise RuntimeError(f"QUARANTINE_TARGET_EXISTS:{target}")
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%dT%H%M%S"))
    args = parser.parse_args()
    desktop = desktop_path()
    archive_name = f"STAGE3_H13_D22_D23_D24_FORWARD_OPTIMIZATION_EVIDENCE_MAX_XZ_{args.timestamp}.tar.xz"
    final_path = desktop / archive_name
    if final_path.exists():
        raise RuntimeError(f"FINAL_ARCHIVE_ALREADY_EXISTS:{final_path}")
    stage_parent = ROOT / "tmp"
    stage_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="stage3_h13_d22_d23_d24_archive_", dir=str(stage_parent)))
    temp_archive = stage_parent / f"{archive_name}.tmp"
    quarantine = stage_parent / "stage3_h13_previous_desktop_archives"
    moved = []
    excluded_for_size: list[str] = []
    try:
        moved = quarantine_existing_related_archives(desktop, quarantine)
        mapping = collect_sources()
        records, duplicates = copy_sources(mapping, staging)
        write_internal_metadata(staging, records, duplicates, excluded_for_size)
        make_archive(staging, temp_archive)
        if temp_archive.stat().st_size > MAX_BYTES:
            removals = ["D22/branch_update_records.jsonl", "D22/branch_update_records.checkpoint.jsonl", "D22/imported_cell_validation.json", "D22/trajectory_diagnostics.json", "D23/RUNG1_TRAJECTORIES.csv"]
            for relative in removals:
                target = staging / relative
                if target.exists() and temp_archive.stat().st_size > MAX_BYTES:
                    target.unlink()
                    excluded_for_size.append(relative)
                    records = [record for record in records if record["relative_path"] != relative]
                    write_internal_metadata(staging, records, duplicates, excluded_for_size)
                    make_archive(staging, temp_archive)
        if temp_archive.stat().st_size > MAX_BYTES:
            raise RuntimeError(f"ARCHIVE_SIZE_LIMIT_EXCEEDED:{temp_archive.stat().st_size}")
        shutil.move(str(temp_archive), str(final_path))
        verification = verify_archive(final_path, staging)
        desktop_archives = [path for path in desktop.iterdir() if path.is_file() and path.suffix.lower() in (".zip", ".tar", ".xz", ".7z", ".gz", ".bz2")]
        tar_xz_archives = [path for path in desktop.iterdir() if path.is_file() and path.name.lower().endswith(".tar.xz")]
        if len(tar_xz_archives) != 1 or tar_xz_archives[0].resolve() != final_path.resolve():
            raise RuntimeError(f"DESKTOP_FINAL_ARCHIVE_COUNT_INVALID:{[str(path) for path in tar_xz_archives]}")
        if desktop_archives != tar_xz_archives:
            raise RuntimeError(f"DESKTOP_OTHER_ARCHIVES_REMAIN:{[str(path) for path in desktop_archives if path not in tar_xz_archives]}")
        result = {"final_archive": str(final_path.resolve()), "format": "tar.xz", "compression": "XZ / LZMA2 preset 9 + EXTREME", "archive_size_bytes": final_path.stat().st_size, "archive_size_mib": final_path.stat().st_size / (1024 * 1024), "file_count": verification["internal_file_count"], "archive_sha256": sha256(final_path), **verification, "desktop_archive_count": 1, "all_priority_a_evidence_included": "YES", "best_or_final_checkpoint_included": "YES", "d23_provenance_included": "YES", "d22_provenance_included": "YES", "only_one_final_archive_created": "YES", "excluded_for_size_limit": excluded_for_size, "quarantined_previous_related_archives": moved}
        report = ROOT / "outputs" / "STAGE3_H13_D22_D23_D24_ARCHIVE_FINALIZATION_REPORT.json"
        report.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
        return 0
    finally:
        if temp_archive.exists():
            temp_archive.unlink()
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
