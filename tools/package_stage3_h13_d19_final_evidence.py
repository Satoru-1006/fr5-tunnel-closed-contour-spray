from __future__ import annotations

import hashlib
import json
import lzma
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\robotfucker")
OUTPUTS = ROOT / "outputs"
DESKTOP = Path(r"C:\Users\86198\Desktop")

D19_OUT = OUTPUTS / "stage3_h13_post_d18_d19_h32_by_individual_rest_pairwise_interaction_causal_screen_replay_20260820T151342"
DESIGN_DIR = OUTPUTS / "stage3_h13_post_d18_d19_h32_by_individual_rest_pairwise_interaction_causal_screen_design_review_20260820T142946+0800"
D16_DIR = OUTPUTS / "stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay_20260820T010000+0800"
D17_DIR = OUTPUTS / "stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay_20260820T041657"
D18_DIR = OUTPUTS / "stage3_h13_post_d17_d18_h2_h31_joint_rollout_deletion_with_h32_retained_causal_intervention_replay_20260820T134651"
BUNDLE_DIR = OUTPUTS / "stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry"

OBSOLETE_DESKTOP_ARCHIVE = DESKTOP / "stage3_h13_d19_design_review_evidence_20260820T142946.tar.xz"
SOURCE_REVISION_FALLBACK = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return SOURCE_REVISION_FALLBACK


class PackageBuilder:
    def __init__(self) -> None:
        for required in (D19_OUT, DESIGN_DIR, D16_DIR, D17_DIR, D18_DIR, BUNDLE_DIR):
            if not required.is_dir():
                raise FileNotFoundError(f"Required input directory is missing: {required}")
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.stage = OUTPUTS / f"D19_FINAL_EVIDENCE_{stamp}"
        self.archive = DESKTOP / f"stage3_h13_d19_execution_evidence_{stamp}.tar.xz"
        if self.stage.exists() or self.archive.exists():
            raise FileExistsError(f"Refusing to overwrite existing packaging target: {self.stage} or {self.archive}")
        self.stage.mkdir(parents=True)
        self.records: dict[str, dict[str, object]] = {}
        self.source_revision = git_revision()

    def archive_rel(self, dest: Path) -> str:
        return "D19_FINAL_EVIDENCE/" + dest.relative_to(self.stage).as_posix()

    def copy_file(self, source: Path, relative: str, category: str, reason: str) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Required evidence file is missing: {source}")
        destination = self.stage / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        self.records[self.archive_rel(destination)] = {
            "original_path": str(source),
            "category": category,
            "reason": reason,
        }

    def copy_optional(self, source: Path, relative: str, category: str, reason: str) -> bool:
        if not source.is_file():
            return False
        self.copy_file(source, relative, category, reason)
        return True

    def write_generated(self, relative: str, value: object, category: str, reason: str) -> None:
        destination = self.stage / Path(relative)
        json_dump(destination, value)
        self.records[self.archive_rel(destination)] = {
            "original_path": f"generated://{self.archive_rel(destination)}",
            "category": category,
            "reason": reason,
        }

    def write_text(self, relative: str, value: str, category: str, reason: str) -> None:
        destination = self.stage / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(value, encoding="utf-8")
        self.records[self.archive_rel(destination)] = {
            "original_path": f"generated://{self.archive_rel(destination)}",
            "category": category,
            "reason": reason,
        }

    def copy_d19_output(self) -> None:
        for source in sorted(D19_OUT.iterdir()):
            if source.is_file():
                self.copy_file(
                    source,
                    f"00_final_result/{source.name}",
                    "Priority A / D19 final result",
                    "Complete authenticated D19 execution output copied without alteration.",
                )

    def write_d19_derivatives(self) -> None:
        endpoint_path = D19_OUT / "endpoint_results.json"
        rankings_path = D19_OUT / "rankings.json"
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
        classification = endpoint["classification"]
        decomposition = endpoint["global_decomposition"]
        rows = endpoint["primary_pairwise_rows"]

        self.write_generated(
            "00_final_result/classification_result.json",
            {
                "source": "endpoint_results.json",
                "classification": classification,
                "classification_exact": classification["classification"],
                "validity": classification["validity"],
                "decision_rule": classification["rule"],
            },
            "Priority A / exact classification",
            "Compact discoverable copy of the exact D19 classification and decision rule.",
        )
        self.write_generated(
            "00_final_result/materiality_preservation.json",
            {
                "source": "endpoint_results.json",
                "pairwise_row_count": len(rows),
                "all_rows_valid": all(row.get("VALIDITY") == "VALID" for row in rows),
                "h1_material_effect_pass_count": sum(
                    bool(row.get("H1_MATERIAL_EFFECT_PASS_FOR_PAIR_BRANCH")) for row in rows
                ),
                "h1_material_effect_pass_all_30": all(
                    bool(row.get("H1_MATERIAL_EFFECT_PASS_FOR_PAIR_BRANCH")) for row in rows
                ),
                "h32_preserved_count": sum(bool(row.get("H32_PRESERVED")) for row in rows),
                "h32_preserved_all_30": all(bool(row.get("H32_PRESERVED")) for row in rows),
                "horizons": sorted(row["HORIZON"] for row in rows),
                "thresholds": endpoint.get("thresholds"),
                "classification": classification,
                "global_decomposition": decomposition,
            },
            "Priority A / materiality and preservation",
            "Compact exact materiality, preservation, threshold, and higher-order remainder record.",
        )
        self.write_generated(
            "00_final_result/signed_interaction_ranking.json",
            {
                "source": "rankings.json",
                "order": rankings["signed_I_32_i_descending"],
            },
            "Priority A / signed ranking",
            "Complete signed interaction ranking for all 30 horizons.",
        )
        self.write_generated(
            "00_final_result/absolute_interaction_ranking.json",
            {
                "source": "rankings.json",
                "order": rankings["absolute_I_32_i_descending"],
            },
            "Priority A / absolute ranking",
            "Complete absolute interaction ranking for all 30 horizons.",
        )

    def copy_execution_integrity(self) -> None:
        names = [
            "authoritative_input_identity.json",
            "branch_execution_manifest.json",
            "design_bundle_identity.json",
            "execution_configuration.json",
            "execution_progress.json",
            "imported_baseline_validation.json",
            "protocol_integrity.json",
            "manifest_verification.json",
        ]
        for name in names:
            self.copy_file(
                D19_OUT / name,
                f"02_execution_integrity/{name}",
                "Priority A / execution integrity",
                "D19 execution, protocol, identity, branch, or imported-evidence integrity record.",
            )

    def copy_diagnostics_and_repairs(self) -> None:
        for name in ("trajectory_diagnostics.json", "per_update_instrumentation.jsonl"):
            self.copy_file(
                D19_OUT / name,
                f"03_trajectory_diagnostics/{name}",
                "Priority B / trajectory diagnostics",
                "Complete D19 trajectory or per-update diagnostic record.",
            )
        self.copy_file(
            D19_OUT / "engineering_defects_and_repairs.json",
            "07_engineering_logs/engineering_defects_and_repairs.json",
            "Priority B / engineering repair",
            "Authenticated engineering defect and repair record for the D19 runner integration.",
        )
        self.write_text(
            "07_engineering_logs/execution_notes.md",
            "# D19 execution notes\n\n"
            "The final D19 replay completed after one pre-execution runner integration repair. "
            "The initial draft called a nonexistent `d17.run_control_branch`; the repaired runner "
            "uses the authenticated local `run_reference_control` path, which calls D17's "
            "`branch_update(\"CONTROL\", ...)` over schedule indices 384..391. No scientific "
            "branch executed before the repair, and treatment semantics were unchanged. See "
            "`engineering_defects_and_repairs.json` for the exact record.\n",
            "Priority B / engineering notes",
            "Human-readable pointer to the authenticated engineering repair record.",
        )

    def copy_sources(self) -> None:
        source_relatives = [
            "tools/stage3_h13_post_d18_d19_h32_by_individual_rest_pairwise_interaction_causal_screen_replay.py",
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
        ]
        source_records = []
        for relative in source_relatives:
            source = ROOT / Path(relative)
            archive_relative = f"04_source_code/repository/{relative}"
            self.copy_file(
                source,
                archive_relative,
                "Priority C / source code",
                "Relevant D19 runner or authenticated helper source, preserving repository-relative path.",
            )
            if relative.startswith("tools/stage3_h13_post_d18_d19_"):
                self.copy_file(
                    source,
                    f"04_source_code/{source.name}",
                    "Priority C / exact D19 runner",
                    "Exact D19 runner source at a discoverable archive path; repository-relative copy is also retained.",
                )
            source_records.append(
                {
                    "original_path": str(source),
                    "repository_relative_path": relative,
                    "archive_paths": [self.archive_rel(self.stage / Path(archive_relative))]
                    + ([self.archive_rel(self.stage / Path(f"04_source_code/{source.name}"))] if relative.startswith("tools/stage3_h13_post_d18_d19_") else []),
                    "bytes": source.stat().st_size,
                    "sha256": sha256(source),
                    "source_revision": self.source_revision,
                }
            )
        self.write_generated(
            "04_source_code/source_identity.json",
            {
                "source_revision": self.source_revision,
                "source_revision_context": "git HEAD at packaging time; source bytes and hashes below are authoritative for this archive.",
                "files": source_records,
            },
            "Priority C / source identity",
            "Source paths, repository-relative paths, byte sizes, SHA256 hashes, and revision context.",
        )

    def copy_design_review(self) -> None:
        expected = [
            "experiment_design.json",
            "branch_manifest.json",
            "imported_evidence_identity.json",
            "analysis_contract.json",
            "classification_contract.json",
            "trajectory_instrumentation_contract.json",
            "provenance_record.json",
            "FINAL_REPORT.md",
            "SHA256_MANIFEST.json",
        ]
        for name in expected:
            self.copy_file(
                DESIGN_DIR / name,
                f"05_design_contract/design_review/{name}",
                "Priority D / D19 design review",
                "Complete D19 design-review contract or review evidence file.",
            )

    def copy_imported_evidence(self) -> None:
        selections = {
            "D16": (
                D16_DIR,
                [
                    "FINAL_REPORT.md",
                    "SHA256_MANIFEST.json",
                    "endpoint_evidence.json",
                    "execution_configuration.json",
                    "authoritative_input_identity.json",
                    "protocol_integrity.json",
                    "causal_classification.json",
                    "causal_estimand.json",
                    "control_reproduction.json",
                    "design_bundle_identity.json",
                    "runtime_source_identity.json",
                    "canonical_batch_schedule.json",
                    "terminal_certificate.json",
                    "repair_log.json",
                    "result.json",
                ],
            ),
            "D17": (
                D17_DIR,
                [
                    "FINAL_REPORT.md",
                    "SHA256_MANIFEST.json",
                    "endpoint_results.csv",
                    "endpoint_results.json",
                    "branch_execution_manifest.json",
                    "execution_configuration.json",
                    "experiment_execution.json",
                    "protocol_integrity.json",
                    "runtime_source_identity.json",
                    "d16_import_record.json",
                    "design_bundle_identity.json",
                ],
            ),
            "D18": (
                D18_DIR,
                [
                    "FINAL_REPORT.md",
                    "SHA256_MANIFEST.json",
                    "endpoint_results.csv",
                    "endpoint_results.json",
                    "non_additivity_decomposition.json",
                    "branch_execution_manifest.json",
                    "execution_configuration.json",
                    "protocol_integrity.json",
                    "trajectory_diagnostics.json",
                    "authoritative_input_identity.json",
                ],
            ),
        }
        missing: dict[str, list[str]] = {}
        for label, (directory, names) in selections.items():
            for name in names:
                if not self.copy_optional(
                    directory / name,
                    f"06_imported_D16_D17_D18/{label}/{name}",
                    f"Priority E / imported {label} evidence",
                    f"Compact authenticated {label} predecessor evidence relevant to D19 provenance.",
                ):
                    missing.setdefault(label, []).append(name)
        self.write_generated(
            "06_imported_D16_D17_D18/imported_evidence_index.json",
            {
                "source_directories": {"D16": str(D16_DIR), "D17": str(D17_DIR), "D18": str(D18_DIR)},
                "missing_optional_requested_files": missing,
                "note": "Only compact authenticated predecessor evidence is included; raw predecessor directories and large tensors remain outside the archive.",
            },
            "Priority E / imported evidence index",
            "Index of compact predecessor evidence and any requested names absent from the authoritative output directories.",
        )

    def write_excluded_binaries(self) -> None:
        binary_extensions = {".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz", ".tensor", ".pkl", ".pickle"}
        roots = [D16_DIR, D17_DIR, D18_DIR, BUNDLE_DIR]
        artifacts = []
        seen: set[str] = set()
        for root in roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in binary_extensions:
                    continue
                key = str(path.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                artifacts.append(
                    {
                        "absolute_original_path": str(path),
                        "filename": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "role": "certified branch-state/model/tensor artifact" if "post_update_384" in path.name else "predecessor raw binary/tensor artifact",
                        "reason_excluded": "Large binary/tensor artifacts are explicitly excluded from the final evidence archive per D19 packaging instruction; authenticated cryptographic reference is retained.",
                        "origin": str(root),
                        "remains_available": True,
                    }
                )
        self.write_generated(
            "08_provenance_and_hashes/EXCLUDED_LARGE_ARTIFACTS.json",
            {
                "policy": "No large binary artifacts are included in this final D19 evidence archive.",
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            },
            "Provenance / excluded large artifacts",
            "Absolute paths, sizes, SHA256 hashes, roles, reasons, origins, and availability of excluded binaries.",
        )

    def write_packaging_log(self) -> None:
        self.write_generated(
            "08_provenance_and_hashes/packaging_log.json",
            {
                "packaging_operation": "D19_FINAL_EVIDENCE",
                "package_created_at_local": datetime.now().isoformat(timespec="seconds"),
                "staging_directory": str(self.stage),
                "archive_target": str(self.archive),
                "xz_parameters": "LZMA2 preset 9 + EXTREME",
                "authoritative_d19_output": str(D19_OUT),
                "desktop_cleanup_target": str(OBSOLETE_DESKTOP_ARCHIVE),
                "desktop_cleanup_deferred_until_archive_validation": True,
                "authoritative_outputs_preserved": True,
            },
            "Provenance / packaging log",
            "Packaging inputs, parameters, target, and preservation policy.",
        )

    def write_archive_contents(self) -> None:
        excluded = {
            "D19_FINAL_EVIDENCE/08_provenance_and_hashes/ARCHIVE_CONTENTS.json",
            "D19_FINAL_EVIDENCE/08_provenance_and_hashes/SHA256_MANIFEST.json",
        }
        entries = []
        for path in sorted(self.stage.rglob("*")):
            if not path.is_file():
                continue
            archive_path = self.archive_rel(path)
            if archive_path in excluded:
                continue
            record = self.records.get(archive_path, {})
            entries.append(
                {
                    "archive_relative_path": archive_path,
                    "original_path": record.get("original_path", f"staged://{archive_path}"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "category": record.get("category", "Uncategorized"),
                    "reason": record.get("reason", "Included as part of the final D19 evidence tree."),
                }
            )
        self.write_generated(
            "08_provenance_and_hashes/ARCHIVE_CONTENTS.json",
            {
                "inventory_version": "D19_FINAL_EVIDENCE_ARCHIVE_CONTENTS_V1",
                "self_excluded": True,
                "sha256_manifest_generated_after_this_inventory": True,
                "excluded_metadata_files": sorted(excluded),
                "total_included_file_count_before_final_sha256_manifest": len(entries),
                "total_included_uncompressed_bytes_before_final_sha256_manifest": sum(item["bytes"] for item in entries),
                "excluded_large_artifact_count": len(json.loads((self.stage / "08_provenance_and_hashes/EXCLUDED_LARGE_ARTIFACTS.json").read_text(encoding="utf-8"))["artifacts"]),
                "files": entries,
            },
            "Provenance / archive contents",
            "Per-file archive path, original path, byte size, SHA256, category, and reason inventory.",
        )

    def write_sha256_manifest(self) -> None:
        self_path = "D19_FINAL_EVIDENCE/08_provenance_and_hashes/SHA256_MANIFEST.json"
        entries = []
        for path in sorted(self.stage.rglob("*")):
            if not path.is_file():
                continue
            archive_path = self.archive_rel(path)
            if archive_path == self_path:
                continue
            record = self.records.get(archive_path, {})
            entries.append(
                {
                    "path": archive_path,
                    "original_path": record.get("original_path", f"staged://{archive_path}"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        self.write_generated(
            "08_provenance_and_hashes/SHA256_MANIFEST.json",
            {
                "manifest_version": "D19_FINAL_EVIDENCE_SHA256_MANIFEST_V1",
                "self_excluded": True,
                "total_files": len(entries),
                "total_bytes": sum(item["bytes"] for item in entries),
                "entries": entries,
            },
            "Provenance / final package SHA256 manifest",
            "Final package-level SHA256 manifest; the manifest self-hash is excluded and documented.",
        )

    def build_tree(self) -> None:
        self.copy_d19_output()
        self.write_d19_derivatives()
        self.copy_execution_integrity()
        self.copy_diagnostics_and_repairs()
        self.copy_sources()
        self.copy_design_review()
        self.copy_imported_evidence()
        self.write_excluded_binaries()
        self.write_packaging_log()
        self.write_archive_contents()
        self.write_sha256_manifest()

    def verify_internal_manifest(self) -> tuple[bool, int, int]:
        manifest_path = self.stage / "08_provenance_and_hashes/SHA256_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked = 0
        bytes_total = 0
        for entry in manifest["entries"]:
            archive_path = entry["path"]
            if not archive_path.startswith("D19_FINAL_EVIDENCE/"):
                return False, checked, bytes_total
            relative = archive_path.removeprefix("D19_FINAL_EVIDENCE/")
            path = self.stage / Path(relative)
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                return False, checked, bytes_total
            checked += 1
            bytes_total += path.stat().st_size
        return checked == manifest["total_files"] and bytes_total == manifest["total_bytes"], checked, bytes_total

    def create_archive(self) -> None:
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(self.archive, mode="w:xz", preset=9 | lzma.PRESET_EXTREME) as tar:
            tar.add(self.stage, arcname="D19_FINAL_EVIDENCE")

    def verify_archive(self) -> dict[str, object]:
        required = {
            "D19_FINAL_EVIDENCE/00_final_result/FINAL_REPORT.md",
            "D19_FINAL_EVIDENCE/00_final_result/endpoint_results.csv",
            "D19_FINAL_EVIDENCE/00_final_result/endpoint_results.json",
            "D19_FINAL_EVIDENCE/00_final_result/global_decomposition.json",
            "D19_FINAL_EVIDENCE/00_final_result/rankings.json",
            "D19_FINAL_EVIDENCE/00_final_result/classification_result.json",
            "D19_FINAL_EVIDENCE/00_final_result/materiality_preservation.json",
            "D19_FINAL_EVIDENCE/02_execution_integrity/branch_execution_manifest.json",
            "D19_FINAL_EVIDENCE/03_trajectory_diagnostics/trajectory_diagnostics.json",
            "D19_FINAL_EVIDENCE/03_trajectory_diagnostics/per_update_instrumentation.jsonl",
            "D19_FINAL_EVIDENCE/04_source_code/stage3_h13_post_d18_d19_h32_by_individual_rest_pairwise_interaction_causal_screen_replay.py",
            "D19_FINAL_EVIDENCE/05_design_contract/design_review/SHA256_MANIFEST.json",
            "D19_FINAL_EVIDENCE/08_provenance_and_hashes/ARCHIVE_CONTENTS.json",
            "D19_FINAL_EVIDENCE/08_provenance_and_hashes/SHA256_MANIFEST.json",
            "D19_FINAL_EVIDENCE/08_provenance_and_hashes/EXCLUDED_LARGE_ARTIFACTS.json",
        }
        names: set[str] = set()
        regular_file_count = 0
        with tarfile.open(self.archive, mode="r:xz") as tar:
            members = tar.getmembers()
            for member in members:
                names.add(member.name.rstrip("/"))
                if member.isfile():
                    regular_file_count += 1
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Could not read tar member: {member.name}")
                    while extracted.read(1024 * 1024):
                        pass
        manifest_ok, checked, manifest_bytes = self.verify_internal_manifest()
        endpoint = json.loads((self.stage / "00_final_result/endpoint_results.json").read_text(encoding="utf-8"))
        rows = endpoint["primary_pairwise_rows"]
        excluded = json.loads((self.stage / "08_provenance_and_hashes/EXCLUDED_LARGE_ARTIFACTS.json").read_text(encoding="utf-8"))
        return {
            "archive": str(self.archive),
            "archive_size_bytes": self.archive.stat().st_size,
            "archive_size_mib": self.archive.stat().st_size / (1024 * 1024),
            "archive_sha256": sha256(self.archive),
            "file_count": regular_file_count,
            "internal_manifest_verification": "VERIFIED" if manifest_ok else "FAILED",
            "internal_manifest_files_checked": checked,
            "internal_manifest_bytes_checked": manifest_bytes,
            "xz_decompression_test": "PASSED",
            "tar_content_read_test": "PASSED",
            "required_priority_a_paths_present": required.issubset(names),
            "all_30_pairwise_results": len(rows) == 30 and sorted(row["HORIZON"] for row in rows) == list(range(2, 32)),
            "large_binary_artifacts_included": any(Path(name).suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz", ".tensor", ".pkl", ".pickle"} for name in names),
            "excluded_large_artifacts_count": len(excluded["artifacts"]),
        }

    def cleanup_and_finalize(self, result: dict[str, object]) -> dict[str, object]:
        if result["archive_size_bytes"] > 20 * 1024 * 1024:
            raise RuntimeError(f"Archive exceeds 20 MiB: {result['archive_size_bytes']} bytes")
        if result["internal_manifest_verification"] != "VERIFIED":
            raise RuntimeError("Internal package SHA256 manifest verification failed")
        if not result["required_priority_a_paths_present"] or not result["all_30_pairwise_results"]:
            raise RuntimeError("Mandatory D19 evidence is missing from the archive")
        if result["large_binary_artifacts_included"]:
            raise RuntimeError("A large binary artifact was included in the final archive")
        if OBSOLETE_DESKTOP_ARCHIVE.exists():
            if OBSOLETE_DESKTOP_ARCHIVE.suffix.lower() != ".xz" or not OBSOLETE_DESKTOP_ARCHIVE.name.startswith("stage3_h13_d19_"):
                raise RuntimeError(f"Refusing cleanup of unexpected file: {OBSOLETE_DESKTOP_ARCHIVE}")
            OBSOLETE_DESKTOP_ARCHIVE.unlink()
        desktop_archives = sorted(
            path for path in DESKTOP.iterdir()
            if path.is_file() and path.name.lower().startswith("stage3_h13_d19_") and path.name.lower().endswith(".tar.xz")
        )
        result["desktop_final_d19_archive_count"] = len(desktop_archives)
        result["desktop_archive_paths"] = [str(path) for path in desktop_archives]
        if desktop_archives != [self.archive]:
            raise RuntimeError(f"Desktop D19 archive set is not exactly the final archive: {desktop_archives}")
        result["excluded_large_artifacts_cryptographically_referenced"] = True
        result["all_priority_a_evidence_included"] = bool(result["required_priority_a_paths_present"])
        return result


def main() -> None:
    builder = PackageBuilder()
    builder.build_tree()
    builder.create_archive()
    result = builder.verify_archive()
    result = builder.cleanup_and_finalize(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
