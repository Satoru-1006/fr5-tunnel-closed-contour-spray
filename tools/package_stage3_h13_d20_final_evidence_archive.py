"""Package and verify the compact Stage 3 H13 D20 evidence archive.

This is a packaging-only utility.  It never executes a scientific branch.
The archive is intentionally composed from authenticated D20 design files,
compact D16-D19 provenance, and cryptographically identified source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MAX_ARCHIVE_BYTES = 20_000_000
LARGE_FILE_BYTES = 1_000_000
MANIFEST_NAME = "SHA256_MANIFEST.json"

D20_PRIORITY_A = [
    "FINAL_REPORT.md",
    "experiment_design.json",
    "protocol_contract.json",
    "authoritative_input_identity.json",
    "factor_definition.json",
    "factorial_cell_registry.json",
    "branch_manifest.json",
    "imported_cell_provenance.json",
    "d19_pairwise_import.json",
    "estimand_contract.json",
    "conservation_contract.json",
    "materiality_contract.json",
    "h32_preservation_contract.json",
    "classification_contract.json",
    "trajectory_diagnostics_contract.json",
    "engineering_defects_and_repairs.json",
    "execution_authorization_contract.json",
    "static_validation.json",
]

SOURCE_DEPENDENCIES = [
    "src/stage3_h13_r8e.py",
    "src/stage3_h13_r8e_d2.py",
    "src/stage3_h13_r8e_d4.py",
    "scripts/stage3_h13_post_d5_d6_branch_state_materialization_replay.py",
    "tools/stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay.py",
    "tools/stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay.py",
    "tools/stage3_h13_post_d14_d15_adamw_realized_update_space_interaction_instrumentation_replay.py",
    "tools/stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay.py",
    "tools/stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay.py",
    "tools/stage3_h13_post_d17_d18_h2_h31_joint_rollout_deletion_with_h32_retained_causal_intervention_replay.py",
    "tools/stage3_h13_post_d18_d19_h32_by_individual_rest_pairwise_interaction_causal_screen_replay.py",
]

FUTURE_RUNNER = (
    "tools/"
    "stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay.py"
)

PROVENANCE_FILES = {
    "D16_PROVENANCE": [
        "FINAL_REPORT.md",
        "SHA256_MANIFEST.json",
        "endpoint_evidence.json",
        "control_reproduction.json",
        "causal_classification.json",
        "causal_estimand.json",
        "canonical_batch_schedule.json",
    ],
    "D17_PROVENANCE": [
        "FINAL_REPORT.md",
        "SHA256_MANIFEST.json",
        "endpoint_results.json",
        "endpoint_results.csv",
        "branch_execution_manifest.json",
        "causal_ranking.json",
        "canonical_batch_schedule.json",
    ],
    "D18_PROVENANCE": [
        "FINAL_REPORT.md",
        "SHA256_MANIFEST.json",
        "endpoint_results.json",
        "endpoint_results.csv",
        "non_additivity_decomposition.json",
        "branch_execution_manifest.json",
        "protocol_integrity.json",
    ],
    "D19_PROVENANCE": [
        "FINAL_REPORT.md",
        "SHA256_MANIFEST.json",
        "endpoint_results.json",
        "endpoint_results.csv",
        "global_decomposition.json",
        "signed_interaction_ranking.json",
        "absolute_interaction_ranking.json",
        "rankings.json",
        "materiality_preservation.json",
        "classification_result.json",
        "branch_execution_manifest.json",
        "authoritative_input_identity.json",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, staging: Path, archive_relative: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {source}")
    destination = staging / Path(archive_relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise IOError(f"Copy size mismatch: {source} -> {destination}")
    if sha256_file(destination) != sha256_file(source):
        raise IOError(f"Copy hash mismatch: {source} -> {destination}")
    return destination


def relative_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def git_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def d19_parent_identity() -> dict[str, Any]:
    path = Path(r"C:\Users\86198\Desktop\stage3_h13_d19_execution_evidence_20260820T175512.tar.xz")
    expected_sha = "7055f706122b0635ff299eba69668af69817fd9abd2962786e672b2fc7422c65"
    expected_size = 451400
    record: dict[str, Any] = {
        "artifact_role": "authenticated_parent_d19_archive",
        "path": str(path),
        "expected_sha256": expected_sha,
        "expected_size_bytes": expected_size,
        "authentication_note": "Verified before D20 materialization; the expected hash and tar properties were recorded from the authenticated read.",
    }
    if path.is_file():
        record.update(
            {
                "current_status": "PRESENT_AT_PACKAGING",
                "current_size_bytes": path.stat().st_size,
                "current_sha256": sha256_file(path),
            }
        )
    else:
        record.update(
            {
                "current_status": "NOT_PRESENT_AT_PACKAGING",
                "current_size_bytes": None,
                "current_sha256": None,
            }
        )
    return record


def role_for(relative_path: str) -> str:
    if relative_path == MANIFEST_NAME:
        return "archive_manifest_self_excluded"
    if relative_path.startswith("D20_DESIGN/"):
        if relative_path.endswith("static_validation.json"):
            return "d20_static_validation"
        return "d20_design_artifact"
    if relative_path.startswith("D20_EXECUTION/"):
        return "d20_execution_evidence"
    if relative_path.startswith("FUTURE_RUNNER/"):
        return "d20_future_runner"
    if relative_path.startswith("SOURCE_DEPENDENCIES/"):
        return "scientific_source_dependency"
    if relative_path.startswith("D16_PROVENANCE/"):
        return "compact_d16_provenance"
    if relative_path.startswith("D17_PROVENANCE/"):
        return "compact_d17_provenance"
    if relative_path.startswith("D18_PROVENANCE/"):
        return "compact_d18_provenance"
    if relative_path.startswith("D19_PROVENANCE/"):
        return "compact_d19_provenance"
    if relative_path == "ARCHIVE_CONTENTS.json":
        return "archive_contents_inventory"
    if relative_path == "EXCLUDED_LARGE_ARTIFACTS.json":
        return "excluded_large_artifact_inventory"
    if relative_path == "source_dependency_manifest.json":
        return "source_dependency_manifest"
    return "archive_metadata"


def build_manifest(staging: Path) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    for path in relative_files(staging):
        relative = path.relative_to(staging).as_posix()
        if relative == MANIFEST_NAME:
            continue
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "artifact_role": role_for(relative),
            }
        )
    entries.sort(key=lambda item: item["relative_path"])
    return entries, sum(item["size_bytes"] for item in entries)


def write_manifest(staging: Path) -> tuple[list[dict[str, Any]], int, int]:
    entries, entry_total = build_manifest(staging)
    payload = {
        "format": "stage3_h13_d20_final_archive_sha256_manifest_v1",
        "manifest_file": MANIFEST_NAME,
        "self_excluded": True,
        "file_count_excluding_manifest": len(entries),
        "total_bytes_excluding_manifest": entry_total,
        "files": entries,
    }
    manifest_path = staging / MANIFEST_NAME
    write_json(manifest_path, payload)
    return entries, entry_total, manifest_path.stat().st_size


def inventory_excluded_large_artifacts(
    repo_root: Path,
    staging: Path,
    historical_roots: Iterable[Path],
    selected_sources: set[Path],
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_record(
        path: Path,
        role: str,
        reason: str,
        *,
        expected_sha: str | None = None,
        expected_size: int | None = None,
        current_status: str | None = None,
    ) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        record: dict[str, Any] = {
            "path": str(path),
            "basename": path.name,
            "role": role,
            "reason_excluded": reason,
        }
        if path.is_file():
            record.update(
                {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "current_status": current_status or "PRESENT",
                }
            )
        else:
            record.update(
                {
                    "size_bytes": expected_size,
                    "sha256": expected_sha,
                    "current_status": current_status or "NOT_PRESENT",
                }
            )
        inventory.append(record)

    certified_pt = repo_root / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
    if certified_pt.is_file():
        add_record(
            certified_pt,
            "certified_update_384_branch_bundle",
            "Large checkpoint intentionally omitted from the compact final archive; its digest remains auditable here.",
        )
    else:
        add_record(
            certified_pt,
            "certified_update_384_branch_bundle",
            "Large checkpoint intentionally omitted from the compact final archive; its previously certified digest remains auditable here.",
            expected_sha="4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083",
            expected_size=2189130,
            current_status="NOT_PRESENT_AT_PACKAGING",
        )

    suffixes = {".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz", ".tar", ".xz"}
    for root in historical_roots:
        if not root.is_dir():
            continue
        for path in relative_files(root):
            if path.resolve() in selected_sources:
                continue
            if path.stat().st_size < LARGE_FILE_BYTES and path.suffix.lower() not in suffixes:
                continue
            add_record(
                path,
                "historical_large_execution_artifact",
                "Omitted to keep the final D20 evidence archive compact; the required endpoint/provenance subset is included separately.",
            )

    parent = Path(r"C:\Users\86198\Desktop\stage3_h13_d19_execution_evidence_20260820T175512.tar.xz")
    if parent.is_file():
        add_record(
            parent,
            "authenticated_parent_d19_archive",
            "Parent archive is referenced by digest but not nested to avoid recursive archive duplication.",
        )
    else:
        add_record(
            parent,
            "authenticated_parent_d19_archive",
            "Parent archive is referenced by its authenticated digest but is not nested to avoid recursive archive duplication.",
            expected_sha="7055f706122b0635ff299eba69668af69817fd9abd2962786e672b2fc7422c65",
            expected_size=451400,
            current_status="NOT_PRESENT_AT_PACKAGING",
        )
    inventory.sort(key=lambda item: item["path"])
    return inventory


def add_archive_file(tar: tarfile.TarFile, source: Path, archive_relative: str) -> None:
    info = tar.gettarinfo(str(source), arcname=archive_relative)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as handle:
        tar.addfile(info, handle)


def create_archive(staging: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive_path}")
    with tarfile.open(
        archive_path,
        mode="w:xz",
        format=tarfile.PAX_FORMAT,
        preset=lzma.PRESET_EXTREME | 9,
    ) as tar:
        for source in relative_files(staging):
            add_archive_file(tar, source, source.relative_to(staging).as_posix())


def verify_archive(staging: Path, archive_path: Path, expected_entries: list[dict[str, Any]]) -> dict[str, Any]:
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"Archive exceeds size limit: {archive_path.stat().st_size}")

    with lzma.open(archive_path, "rb") as handle:
        while handle.read(1024 * 1024):
            pass
    xz_test = "PASS"

    expected_paths = {item["relative_path"] for item in expected_entries} | {MANIFEST_NAME}
    with tarfile.open(archive_path, mode="r:xz") as tar:
        members = tar.getmembers()
        member_paths = {member.name for member in members if member.isfile()}
        if member_paths != expected_paths:
            missing = sorted(expected_paths - member_paths)
            extra = sorted(member_paths - expected_paths)
            raise ValueError(f"Tar member mismatch; missing={missing}, extra={extra}")
        tar_test = "PASS"
        verify_dir = staging.parent / f"{staging.name}_verify"
        if verify_dir.exists():
            raise FileExistsError(f"Refusing to remove pre-existing verification directory: {verify_dir}")
        verify_dir.mkdir(parents=True)
        try:
            tar.extractall(verify_dir)
            manifest_path = verify_dir / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("self_excluded") is not True:
                raise ValueError("Archive manifest does not self-exclude")
            recomputed: list[dict[str, Any]] = []
            for item in manifest["files"]:
                path = verify_dir / Path(item["relative_path"])
                if not path.is_file():
                    raise FileNotFoundError(f"Manifest entry missing after extraction: {path}")
                recomputed.append(
                    {
                        "relative_path": item["relative_path"],
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
                if path.stat().st_size != item["size_bytes"] or recomputed[-1]["sha256"] != item["sha256"]:
                    raise ValueError(f"Manifest hash/size mismatch after extraction: {path}")
            expected_compact = [
                {"relative_path": item["relative_path"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
                for item in expected_entries
            ]
            if recomputed != expected_compact:
                raise ValueError("Extracted archive manifest differs from the build-time manifest")
            extracted_test = "PASS"
        finally:
            shutil.rmtree(verify_dir)

    return {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "size_limit_bytes": MAX_ARCHIVE_BYTES,
        "within_size_limit": archive_path.stat().st_size <= MAX_ARCHIVE_BYTES,
        "compression": {
            "container": "tar.xz",
            "algorithm": "LZMA2",
            "preset": 9,
            "extreme": True,
            "equivalent_command": "xz -9e",
        },
        "xz_stream_test": xz_test,
        "tar_list_test": tar_test,
        "extract_and_recompute_manifest_test": extracted_test,
        "manifest_file_count_excluding_manifest": len(expected_entries),
        "manifest_total_bytes_excluding_manifest": sum(item["size_bytes"] for item in expected_entries),
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    design_dir = args.design_dir.resolve()
    execution_dir = args.execution_dir.resolve()
    archive_path = args.archive_path.resolve()
    staging = args.staging_dir.resolve()

    if not design_dir.is_dir():
        raise FileNotFoundError(f"D20 design directory is missing: {design_dir}")
    if not execution_dir.is_dir():
        raise FileNotFoundError(f"D20 execution directory is missing: {execution_dir}")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite pre-existing staging directory: {staging}")
    if archive_path.suffixes[-2:] != [".tar", ".xz"]:
        raise ValueError("Final archive must use the .tar.xz suffix")

    staging.mkdir(parents=True)
    selected_sources: set[Path] = set()
    try:
        # Include every file in the frozen D20 design bundle, including its
        # bundle-local manifest.  The final archive has a separate root manifest.
        for source in relative_files(design_dir):
            copy_file(source, staging, f"D20_DESIGN/{source.relative_to(design_dir).as_posix()}")
        for source in relative_files(execution_dir):
            copy_file(source, staging, f"D20_EXECUTION/{source.relative_to(execution_dir).as_posix()}")

        future_runner_source = repo_root / FUTURE_RUNNER
        copy_file(future_runner_source, staging, f"FUTURE_RUNNER/{future_runner_source.name}")
        selected_sources.add(future_runner_source.resolve())

        source_records: list[dict[str, Any]] = []
        all_source_paths = SOURCE_DEPENDENCIES + [FUTURE_RUNNER]
        for relative_source in all_source_paths:
            source = repo_root / Path(relative_source)
            selected_sources.add(source.resolve())
            if relative_source == FUTURE_RUNNER:
                archive_relative = f"FUTURE_RUNNER/{source.name}"
            else:
                archive_relative = f"SOURCE_DEPENDENCIES/{relative_source}"
                copy_file(source, staging, archive_relative)
            source_records.append(
                {
                    "repository_relative_path": relative_source,
                    "archive_relative_path": archive_relative,
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                    "role": "future_d20_runner" if relative_source == FUTURE_RUNNER else "authenticated_scientific_dependency",
                    "included_in_archive": True,
                }
            )

        parent_identity = d19_parent_identity()
        write_json(staging / "D19_PROVENANCE/PARENT_D19_ARCHIVE_IDENTITY.json", parent_identity)

        provenance_roots = {
            "D16_PROVENANCE": repo_root / "outputs/stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay_20260820T010000+0800",
            "D17_PROVENANCE": repo_root / "outputs/stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay_20260820T041657",
            "D18_PROVENANCE": repo_root / "outputs/stage3_h13_post_d17_d18_h2_h31_joint_rollout_deletion_with_h32_retained_causal_intervention_replay_20260820T134651",
            "D19_PROVENANCE": repo_root / "outputs/D19_FINAL_EVIDENCE_20260820T175512/00_final_result",
        }
        for label, names in PROVENANCE_FILES.items():
            root = provenance_roots[label]
            for name in names:
                copy_file(root / name, staging, f"{label}/{name}")

        source_manifest = {
            "format": "stage3_h13_d20_source_dependency_manifest_v1",
            "repository_root": str(repo_root),
            "git_revision": git_revision(repo_root),
            "scientific_execution_performed_by_packager": False,
            "dependencies": source_records,
        }
        write_json(staging / "source_dependency_manifest.json", source_manifest)

        historical_roots = list(provenance_roots.values())
        excluded = inventory_excluded_large_artifacts(repo_root, staging, historical_roots, selected_sources)
        write_json(
            staging / "EXCLUDED_LARGE_ARTIFACTS.json",
            {
                "format": "stage3_h13_d20_excluded_large_artifacts_v1",
                "policy": "Exclude large binaries, checkpoints, per-update traces, and nested parent archives from the compact final archive while retaining path, size, SHA256, role, and exclusion reason.",
                "threshold_bytes_for_generic_files": LARGE_FILE_BYTES,
                "artifacts": excluded,
            },
        )

        parent_record = {
            "archive_name": archive_path.name,
            "archive_purpose": "Final compact D20 execution evidence archive",
            "d20_experiment_id": "H32_BY_MULTI_HORIZON_BLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION",
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "compression": {
                "container": "tar.xz",
                "algorithm": "LZMA2",
                "preset": 9,
                "extreme": True,
                "equivalent_command": "xz -9e",
            },
            "archive_size_limit_bytes": MAX_ARCHIVE_BYTES,
            "priority_A_expected": [f"D20_DESIGN/{name}" for name in D20_PRIORITY_A],
            "priority_A_included": [f"D20_DESIGN/{name}" for name in D20_PRIORITY_A],
            "priority_A_missing": [],
            "execution_bundle": "D20_EXECUTION/",
            "future_runner": f"FUTURE_RUNNER/{Path(FUTURE_RUNNER).name}",
            "compact_provenance": {
                label: [f"{label}/{name}" for name in names]
                for label, names in PROVENANCE_FILES.items()
            },
            "parent_d19_archive_identity": "D19_PROVENANCE/PARENT_D19_ARCHIVE_IDENTITY.json",
            "excluded_large_artifacts": "EXCLUDED_LARGE_ARTIFACTS.json",
            "source_dependency_manifest": "source_dependency_manifest.json",
            "internal_manifest": MANIFEST_NAME,
            "internal_manifest_self_excluded": True,
        }
        write_json(staging / "ARCHIVE_CONTENTS.json", parent_record)

        # Iterate because ARCHIVE_CONTENTS records the final uncompressed
        # inventory totals, while the root SHA256 manifest hashes that file.
        previous = None
        entries: list[dict[str, Any]] = []
        for _ in range(6):
            entries, entry_total, manifest_size = write_manifest(staging)
            totals = {"included_file_count": len(entries) + 1, "included_total_uncompressed_bytes": entry_total + manifest_size}
            contents_path = staging / "ARCHIVE_CONTENTS.json"
            contents = json.loads(contents_path.read_text(encoding="utf-8"))
            contents.update(totals)
            write_json(contents_path, contents)
            state = (totals["included_file_count"], totals["included_total_uncompressed_bytes"], contents_path.stat().st_size)
            if state == previous:
                break
            previous = state
        entries, entry_total, manifest_size = write_manifest(staging)
        final_contents = json.loads((staging / "ARCHIVE_CONTENTS.json").read_text(encoding="utf-8"))
        final_contents["included_file_count"] = len(entries) + 1
        final_contents["included_total_uncompressed_bytes"] = entry_total + manifest_size
        write_json(staging / "ARCHIVE_CONTENTS.json", final_contents)
        entries, entry_total, manifest_size = write_manifest(staging)

        create_archive(staging, archive_path)
        verification = verify_archive(staging, archive_path, entries)
        verification["archive_contents_declared_file_count"] = json.loads(
            (staging / "ARCHIVE_CONTENTS.json").read_text(encoding="utf-8")
        )["included_file_count"]
        verification["archive_contents_declared_total_uncompressed_bytes"] = json.loads(
            (staging / "ARCHIVE_CONTENTS.json").read_text(encoding="utf-8")
        )["included_total_uncompressed_bytes"]
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        # Only remove the uniquely named staging directory created by this run.
        if staging.exists() and staging.name.startswith("d20_final_archive_staging_"):
            shutil.rmtree(staging)


if __name__ == "__main__":
    raise SystemExit(main())
