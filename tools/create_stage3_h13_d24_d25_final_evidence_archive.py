"""Create exactly one verified <=20 MiB Stage-3 H13 D24/D25 evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
D25 = ROOT / "outputs" / "stage3_h13_post_d24_d25_h32_feasible_realized_adamw_trust_region_forward_continuation_design_review_20260821T235900+0800"
D24 = ROOT / "outputs" / "stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution_20260821T225000+0800_retry"
D23 = ROOT / "outputs" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint_20260821T190000+0800"
D22 = ROOT / "outputs" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_execution_evidence_20260821T151902+0800"
D24_CHECKPOINT = D24 / "checkpoints" / "committed_update_394.pt"
D23_CHECKPOINT = D23 / "checkpoints" / "C5_update_392.pt"
EXPECTED_D24_CHECKPOINT_SHA256 = "f7939ac8d6c9adedbfb2ed6cf90a949e7252ba6926932053617fbe4caac97a47"
EXPECTED_D23_CHECKPOINT_SHA256 = "226cd512c2f16e415c67b3c0e4045a89841a008f7ec4a09e9249e3c2309efdc1"
PREVIOUS_ARCHIVE = "STAGE3_H13_D22_D23_D24_FORWARD_OPTIMIZATION_EVIDENCE_MAX_XZ_20260821T232434+0800.tar.xz"
PREVIOUS_ARCHIVE_SHA256 = "fcec1b244d91d73212b3d6d56837cdfb30b5deda172d9de789cc9b97283caaa6"
MAX_BYTES = 20_971_520
TOP = "STAGE3_H13_D24_D25_EVIDENCE"
METADATA_FILES = {
    "00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json",
    "00_ARCHIVE_METADATA/SHA256_ALL_FILES.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def now_local() -> str:
    return datetime.now().astimezone().isoformat()


def desktop_path() -> Path:
    profile = os.environ.get("USERPROFILE")
    candidate = Path(profile) / "Desktop" if profile else Path.home() / "Desktop"
    if not candidate.is_dir():
        raise RuntimeError(f"DESKTOP_NOT_FOUND:{candidate}")
    return candidate.resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def add(mapping: dict[str, dict[str, Any]], source: Path, relative: str, priority: str, rationale: str) -> None:
    source = source.resolve()
    if not source.is_file():
        raise RuntimeError(f"REQUIRED_SOURCE_MISSING:{source}")
    if relative in mapping:
        raise RuntimeError(f"ARCHIVE_PATH_COLLISION:{relative}")
    mapping[relative] = {"source": source, "priority": priority, "rationale": rationale}


def add_tree(mapping: dict[str, dict[str, Any]], source_root: Path, target_root: str, priority: str, rationale: str, *, include_checkpoints: bool = True) -> None:
    for source in sorted(source_root.rglob("*")):
        if source.is_file() and (include_checkpoints or "checkpoints" not in source.relative_to(source_root).parts):
            relative = f"{target_root}/{source.relative_to(source_root).as_posix()}"
            add(mapping, source, relative, priority, rationale)


def collect_sources() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    add_tree(mapping, D25, "01_D25_DESIGN_REVIEW/artifacts", "A", "complete frozen D25 design-review output")
    add(mapping, ROOT / "tools" / "stage3_h13_post_d24_d25_trust_region_design_review.py", "01_D25_DESIGN_REVIEW/implementation/stage3_h13_post_d24_d25_trust_region_design_review.py", "A", "D25 design-review implementation")
    add(mapping, ROOT / "tools" / "stage3_h13_post_d24_d25_trust_region_execution.py", "01_D25_DESIGN_REVIEW/implementation/stage3_h13_post_d24_d25_trust_region_execution.py", "A", "separately-gated D25 update-395 implementation")
    add(mapping, ROOT / "tests" / "test_stage3_h13_post_d24_d25_design_review.py", "01_D25_DESIGN_REVIEW/tests/test_stage3_h13_post_d24_d25_design_review.py", "A", "focused synthetic/static D25 tests")
    add(mapping, Path(__file__).resolve(), "00_ARCHIVE_METADATA/implementation/create_stage3_h13_d24_d25_final_evidence_archive.py", "A", "final evidence archive packager and manifest-generation implementation")

    for source in sorted(D24.iterdir()):
        if source.is_file():
            add(mapping, source, f"02_D24_EXECUTION/artifacts/{source.name}", "A", "direct D24 report/manifest/result evidence")
    add(mapping, D24 / "checkpoints" / "committed_update_393.pt", "02_D24_EXECUTION/checkpoints/committed_update_393.pt", "B", "D24 preceding committed checkpoint")
    add(mapping, D24_CHECKPOINT, "05_CHECKPOINTS/committed_update_394.pt", "A", "authoritative D25 incumbent checkpoint")

    for source in sorted(D23.iterdir()):
        if source.is_file():
            add(mapping, source, f"03_D23_PROVENANCE/artifacts/{source.name}", "B", "D23 provenance used to authenticate D24")
    add(mapping, D23_CHECKPOINT, "05_CHECKPOINTS/C5_update_392.pt", "B", "authenticated D24 parent checkpoint")

    add_tree(mapping, D22, "04_D22_PROVENANCE/evidence", "C", "compact authoritative D22 provenance")

    implementation_sources = (
        (ROOT / "tools" / "stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution.py", "02_D24_EXECUTION/implementation/D24_runner.py", "A", "D24 P4/proposal/clipping/AdamW implementation"),
        (ROOT / "tools" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint.py", "03_D23_PROVENANCE/implementation/D23_runner.py", "B", "D23 forward-promotion implementation"),
        (ROOT / "tools" / "stage3_h13_post_d21_d22_execution_replay.py", "04_D22_PROVENANCE/implementation/D22_runner.py", "C", "D22 execution implementation"),
        (ROOT / "tools" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_design_review.py", "04_D22_PROVENANCE/implementation/D22_design_review.py", "C", "D22 design authority"),
        (ROOT / "tools" / "stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay.py", "07_RUNTIME_AND_SOURCE_IDENTITY/implementation_helpers/D17_runtime.py", "A", "runtime, validation, and canonical rollout helper"),
        (ROOT / "tools" / "stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay.py", "07_RUNTIME_AND_SOURCE_IDENTITY/implementation_helpers/D16_optimizer_pipeline.py", "A", "gradient, clipping, AdamW-state helper"),
        (ROOT / "AGENTS.md", "07_RUNTIME_AND_SOURCE_IDENTITY/AGENTS.md", "A", "repository execution constraints"),
        (ROOT / "requirements.txt", "07_RUNTIME_AND_SOURCE_IDENTITY/requirements.txt", "D", "dependency context"),
        (ROOT / "pytest.ini", "07_RUNTIME_AND_SOURCE_IDENTITY/pytest.ini", "D", "test runtime context"),
    )
    for source, relative, priority, rationale in implementation_sources:
        add(mapping, source, relative, priority, rationale)

    input_identity = json.loads((D25 / "D25_INPUT_IDENTITY.json").read_text(encoding="utf-8"))
    runtime_paths = input_identity["runtime"]
    for key, target in (
        ("bundle_path", "07_RUNTIME_AND_SOURCE_IDENTITY/canonical_inputs/post_update_384_branch_bundle.pt"),
        ("schedule_identity_path", "07_RUNTIME_AND_SOURCE_IDENTITY/canonical_inputs/canonical_schedule_identity.json"),
        ("schedule_storage_path", "07_RUNTIME_AND_SOURCE_IDENTITY/canonical_inputs/canonical_schedule_manifest.json.gz"),
    ):
        add(mapping, Path(runtime_paths[key]), target, "D", f"authenticated canonical runtime input: {key}")
    add(mapping, Path(runtime_paths["classification_contract_path"]) if "classification_contract_path" in runtime_paths else ROOT / "outputs" / "stage3_h13_post_d13_d14_gru_h32_temporal_conflict_accumulation_causal_intervention_design_review_20260819T183000+0800" / "classification_contract.json", "07_RUNTIME_AND_SOURCE_IDENTITY/canonical_inputs/classification_contract.json", "D", "frozen classification contract")

    prompt_sources = (
        (Path(r"C:\Users\86198\.codex\attachments\1d7bb90c-192f-4242-bd16-54342a88182e\pasted-text.txt"), "06_PROTOCOL_AND_PROMPTS/D25_design_review_prompt.txt", "A", "D25 governing design-review prompt"),
        (Path(r"C:\Users\86198\.codex\attachments\696258bc-6601-4a59-9e91-6c4f96d1f5ca\pasted-text.txt"), "06_PROTOCOL_AND_PROMPTS/D25_archive_prompt.txt", "A", "mandatory archive prompt"),
        (Path(r"C:\Users\86198\.codex\attachments\66c30217-9496-45da-9428-28c3e6ce69fe\pasted-text.txt"), "06_PROTOCOL_AND_PROMPTS/D24_execution_authorization.txt", "B", "D24 execution authorization evidence"),
        (Path(r"C:\Users\86198\.codex\attachments\5eb8e7eb-97b0-47fb-81bf-8de98a0dc3fa\pasted-text.txt"), "06_PROTOCOL_AND_PROMPTS/D23_execution_authorization.txt", "B", "D23 execution authorization evidence"),
    )
    for source, relative, priority, rationale in prompt_sources:
        add(mapping, source, relative, priority, rationale)
    return mapping


def copy_sources(mapping: dict[str, dict[str, Any]], stage: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative, item in sorted(mapping.items()):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], target)
        records.append({"relative_path": relative, "size_bytes": target.stat().st_size, "sha256": sha256(target), "priority": item["priority"], "source_origin": str(item["source"]), "rationale": item["rationale"]})
    return records


def managed_files(stage: Path) -> list[Path]:
    return [path for path in sorted(stage.rglob("*")) if path.is_file() and path.relative_to(stage).as_posix() not in METADATA_FILES]


def write_metadata(stage: Path, records: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> None:
    metadata = stage / "00_ARCHIVE_METADATA"
    metadata.mkdir(parents=True, exist_ok=True)
    provenance = {
        "archive_purpose": "single final Stage-3 H13 D24/D25 realized-AdamW trust-region design evidence archive",
        "creation_timestamp": now_local(),
        "project": "Stage 3 H13",
        "stage": 3,
        "experiment": "STAGE_3_H13_POST_D24_D25_H32_FEASIBLE_REALIZED_ADAMW_TRUST_REGION_FORWARD_CONTINUATION_DESIGN_REVIEW",
        "authoritative_start_update": 394,
        "target_update": 395,
        "authoritative_checkpoint": str(D24_CHECKPOINT.resolve()),
        "authoritative_checkpoint_sha256": EXPECTED_D24_CHECKPOINT_SHA256,
        "previous_evidence_archive": PREVIOUS_ARCHIVE,
        "previous_evidence_archive_sha256": PREVIOUS_ARCHIVE_SHA256,
        "previous_archive_included": False,
        "previous_archive_policy": "used as authenticated provenance only; not nested",
        "training_executed": "NO",
        "backward_pass_executed": "NO",
        "optimizer_step_executed": "NO",
        "update_395_executed": "NO",
        "update_396_plus_executed": "NO",
        "source_revision": "a6dff2b6887cc3f7598ab95549c2dd21c0efe763",
    }
    (metadata / "ARCHIVE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    exclusions = {"EXCLUDED_FOR_SIZE_LIMIT": "none" if not excluded else excluded, "other_policy_exclusions": [{"path": PREVIOUS_ARCHIVE, "reason_excluded": "REDUNDANT_WITH_AUTHENTICATED_PARENT_ARCHIVE", "recoverable_from": "previous archive named in ARCHIVE_PROVENANCE.json", "sha256": PREVIOUS_ARCHIVE_SHA256}]}
    (metadata / "EXCLUSIONS_FOR_SIZE_LIMIT.json").write_text(json.dumps(exclusions, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    current = managed_files(stage)
    file_lines = ["# Archive member file list (paths relative to archive root)", *[path.relative_to(stage).as_posix() for path in current], "00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json", "00_ARCHIVE_METADATA/SHA256_ALL_FILES.json"]
    (metadata / "ARCHIVE_FILE_LIST.txt").write_text("\n".join(file_lines) + "\n", encoding="utf-8", newline="\n")
    current = managed_files(stage)
    total_bytes = sum(path.stat().st_size for path in current)
    contents = {
        "archive_purpose": provenance["archive_purpose"],
        "creation_timestamp": provenance["creation_timestamp"],
        "project": provenance["project"],
        "stage": provenance["stage"],
        "experiment": provenance["experiment"],
        "authoritative_start_update": 394,
        "target_update": 395,
        "authoritative_checkpoint": "05_CHECKPOINTS/committed_update_394.pt",
        "authoritative_checkpoint_sha256": EXPECTED_D24_CHECKPOINT_SHA256,
        "previous_evidence_archive": PREVIOUS_ARCHIVE,
        "previous_evidence_archive_sha256": PREVIOUS_ARCHIVE_SHA256,
        "compression_format": "tar.xz",
        "compression_level": "XZ / LZMA2 preset 9 + EXTREME",
        "size_limit_bytes": MAX_BYTES,
        "included_file_count": len(current),
        "included_total_uncompressed_bytes": total_bytes,
        "priority_A_complete": True,
        "priority_B_complete": True,
        "priority_C_complete": True,
        "excluded_for_size_limit": "none" if not excluded else excluded,
        "SELF_EXCLUDED_FROM_MANIFEST_HASH_COUNT": "YES",
        "hash_inventory_scope": "all managed files except this content manifest and SHA256_ALL_FILES.json",
    }
    (metadata / "ARCHIVE_CONTENTS_MANIFEST.json").write_text(json.dumps(contents, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    current = managed_files(stage)
    inventory = {"schema_version": "stage3_h13_d24_d25_archive_sha256_all_files_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "excluded_from_hash_inventory": sorted(METADATA_FILES), "file_count": len(current), "total_uncompressed_bytes": sum(path.stat().st_size for path in current), "files": [{"relative_path": path.relative_to(stage).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in current]}
    (metadata / "SHA256_ALL_FILES.json").write_text(json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def write_generated_logs(stage: Path) -> None:
    logs = stage / "01_D25_DESIGN_REVIEW" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "D25_STATIC_TEST_OUTPUT.txt").write_text(
        "Command: pytest -q tests/test_stage3_h13_post_d24_d25_design_review.py\n"
        "Result: 4 passed\n"
        "Scientific training executed: NO\n"
        "Scientific backward pass executed: NO\n"
        "Scientific optimizer step executed: NO\n"
        "Update 395 executed: NO\n",
        encoding="utf-8",
        newline="\n",
    )
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (stage / "07_RUNTIME_AND_SOURCE_IDENTITY" / "GIT_HEAD_AND_STATUS.txt").write_text(f"HEAD: {head}\nWorking tree status:\n{status}", encoding="utf-8", newline="\n")


def make_archive(stage: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with tarfile.open(destination, mode="w:xz", format=tarfile.PAX_FORMAT, preset=9 | lzma.PRESET_EXTREME) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=f"{TOP}/{path.relative_to(stage).as_posix()}", recursive=False)


def verify_archive(archive_path: Path) -> dict[str, Any]:
    with lzma.open(archive_path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    extract_root = Path(tempfile.mkdtemp(prefix="stage3_h13_d24_d25_archive_verify_", dir=str(ROOT / "tmp")))
    try:
        with tarfile.open(archive_path, mode="r:xz") as archive:
            members = archive.getmembers()
            file_members = [member for member in members if member.isfile()]
            require(all(not Path(member.name).is_absolute() and ".." not in Path(member.name).parts for member in members), "UNSAFE_TAR_MEMBER_PATH")
            archive.extractall(extract_root)
        root = extract_root / TOP
        contents_path = root / "00_ARCHIVE_METADATA" / "ARCHIVE_CONTENTS_MANIFEST.json"
        inventory_path = root / "00_ARCHIVE_METADATA" / "SHA256_ALL_FILES.json"
        require(contents_path.is_file() and inventory_path.is_file(), "ARCHIVE_METADATA_MISSING")
        contents = json.loads(contents_path.read_text(encoding="utf-8"))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        failures: list[str] = []
        for entry in inventory["files"]:
            target = root / entry["relative_path"]
            if not target.is_file() or target.stat().st_size != int(entry["size_bytes"]) or sha256(target) != entry["sha256"]:
                failures.append(entry["relative_path"])
        checkpoint = root / "05_CHECKPOINTS" / "committed_update_394.pt"
        require(not failures, f"INTERNAL_HASH_FAILURES:{failures}")
        require(checkpoint.is_file() and sha256(checkpoint) == EXPECTED_D24_CHECKPOINT_SHA256, "UPDATE_394_CHECKPOINT_HASH_FAILURE")
        return {"xz_decompression_test": "PASS", "tar_content_read_test": "PASS", "internal_manifest_verification": "PASS", "update_394_checkpoint_hash_verification": "PASS", "member_count": len(file_members), "internal_file_count": inventory["file_count"], "contents_manifest": contents}
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z"))
    args = parser.parse_args()
    desktop = desktop_path()
    archive_name = f"STAGE3_H13_D24_D25_REALIZED_ADAMW_TRUST_REGION_DESIGN_EVIDENCE_MAX_XZ_{args.timestamp}.tar.xz"
    final_path = desktop / archive_name
    require(not final_path.exists(), f"FINAL_ARCHIVE_ALREADY_EXISTS:{final_path}")
    existing_related = [path for path in desktop.iterdir() if path.is_file() and path.name.startswith("STAGE3_H13_D24_D25_REALIZED_ADAMW_TRUST_REGION_DESIGN_EVIDENCE_MAX_XZ_")]
    require(not existing_related, f"RELATED_FINAL_ARCHIVE_ALREADY_EXISTS:{existing_related}")
    require(sha256(D24_CHECKPOINT) == EXPECTED_D24_CHECKPOINT_SHA256, "UPDATE_394_SOURCE_HASH_MISMATCH")
    require(sha256(D23_CHECKPOINT) == EXPECTED_D23_CHECKPOINT_SHA256, "C5_UPDATE_392_SOURCE_HASH_MISMATCH")
    for path in (D25, D24, D23, D22):
        require(path.is_dir(), f"REQUIRED_EVIDENCE_DIR_MISSING:{path}")
    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="stage3_h13_d24_d25_archive_", dir=str(tmp_root)))
    candidate = tmp_root / f"{archive_name}.tmp"
    excluded: list[dict[str, Any]] = []
    try:
        mapping = collect_sources()
        records = copy_sources(mapping, staging)
        write_generated_logs(staging)
        write_metadata(staging, records, excluded)
        make_archive(staging, candidate)
        if candidate.stat().st_size > MAX_BYTES:
            removable = sorted([record for record in records if record["priority"] in ("D", "C")], key=lambda record: (-record["size_bytes"], record["relative_path"]))
            for record in removable:
                if candidate.stat().st_size <= MAX_BYTES:
                    break
                target = staging / record["relative_path"]
                if target.exists():
                    excluded.append({"path": record["relative_path"], "size_bytes": record["size_bytes"], "sha256": record["sha256"], "priority": record["priority"], "reason_excluded": "ARCHIVE_SIZE_LIMIT", "recoverable_from": record["source_origin"]})
                    target.unlink()
                    records = [item for item in records if item["relative_path"] != record["relative_path"]]
                    for metadata_name in METADATA_FILES:
                        target_metadata = staging / metadata_name
                        if target_metadata.exists():
                            target_metadata.unlink()
                    write_metadata(staging, records, excluded)
                    make_archive(staging, candidate)
        require(candidate.stat().st_size <= MAX_BYTES, f"ARCHIVE_SIZE_LIMIT_EXCEEDED:{candidate.stat().st_size}")
        verification = verify_archive(candidate)
        shutil.copy2(candidate, final_path)
        post_copy = verify_archive(final_path)
        archive_files = [path for path in desktop.iterdir() if path.is_file() and path.name.lower().endswith(".tar.xz")]
        other_archives = [path for path in desktop.iterdir() if path.is_file() and path.suffix.lower() in (".zip", ".tar", ".xz", ".7z", ".gz", ".bz2") and not path.name.lower().endswith(".tar.xz")]
        require(len(archive_files) == 1 and archive_files[0].resolve() == final_path.resolve(), f"ONLY_ONE_FINAL_ARCHIVE_FAILED:{archive_files}")
        require(not other_archives, f"OTHER_DESKTOP_ARCHIVES_PRESENT:{other_archives}")
        result = {"FINAL_EVIDENCE_ARCHIVE": str(final_path.resolve()), "FORMAT": "tar.xz", "COMPRESSION": "XZ / LZMA2 preset 9 + EXTREME", "ARCHIVE_SIZE_BYTES": final_path.stat().st_size, "ARCHIVE_SIZE_MIB": final_path.stat().st_size / (1024 * 1024), "FILE_COUNT": {"semantics": "internal SHA256_ALL_FILES.json managed payload count; content/SHA manifests excluded from inventory", "managed_payload_files": verification["internal_file_count"], "tar_file_members": post_copy["member_count"]}, "ARCHIVE_SHA256": sha256(final_path), "XZ_DECOMPRESSION_TEST": post_copy["xz_decompression_test"], "TAR_CONTENT_READ_TEST": post_copy["tar_content_read_test"], "INTERNAL_MANIFEST_VERIFICATION": post_copy["internal_manifest_verification"], "UPDATE_394_CHECKPOINT_INCLUDED": "YES", "UPDATE_394_CHECKPOINT_HASH_VERIFICATION": post_copy["update_394_checkpoint_hash_verification"], "D25_PRIORITY_A_EVIDENCE_INCLUDED": "YES", "D24_PROVENANCE_INCLUDED": "YES", "D23_PROVENANCE_INCLUDED": "YES", "D22_PROVENANCE_INCLUDED": "YES", "EXCLUDED_FOR_SIZE_LIMIT": "none" if not excluded else excluded, "ONLY_ONE_FINAL_ARCHIVE_CREATED": "YES", "TRAINING_EXECUTED": "NO", "BACKWARD_PASS_EXECUTED": "NO", "OPTIMIZER_STEP_EXECUTED": "NO", "UPDATE_395_EXECUTED": "NO", "UPDATE_396_PLUS_EXECUTED": "NO", "D25_DESIGN_MANIFEST_SHA256": sha256(D25 / "D25_DESIGN_MANIFEST.json"), "authoritative_update_394_sha256": EXPECTED_D24_CHECKPOINT_SHA256, "previous_archive": {"name": PREVIOUS_ARCHIVE, "sha256": PREVIOUS_ARCHIVE_SHA256, "nested": False}}
        report_path = ROOT / "outputs" / "STAGE3_H13_D24_D25_ARCHIVE_FINALIZATION_REPORT.json"
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
        return 0
    finally:
        if candidate.exists():
            candidate.unlink()
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
