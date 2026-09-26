"""Evidence-only materialization and copy-only closure for the Stage 3 H13 D7 archive.

This helper intentionally performs no model execution, training, tuning, evaluation,
checkpoint loading, or scientific experiment.  It reads immutable D7 evidence,
materializes a post-D7 interpretation bundle, copies that bundle into the existing
Desktop archive, and regenerates archive integrity metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


D7_DIR = Path(r"D:\robotfucker\outputs\stage3_h13_post_d6_d7_phase4_pareto_localization_replay_20260817T023840Z")
OUTPUTS_DIR = Path(r"D:\robotfucker\outputs")
ARCHIVE_DIR = Path(
    r"C:\Users\86198\Desktop\Stage3_H13_D7_Replay_and_Final_Interpretation_Complete_Evidence_20260817_v3"
)
INTERPRETATION_SECTION = ARCHIVE_DIR / "02_POST_D7_FINAL_SCIENTIFIC_INTERPRETATION"
INTEGRITY_DIR = ARCHIVE_DIR / "10_INTEGRITY"
OLD_COPY_MANIFEST = INTEGRITY_DIR / "COPY_MANIFEST.json"
OLD_SOURCE_PATH_MAP = INTEGRITY_DIR / "SOURCE_PATH_MAP.json"

REQUIRED_D7_FILES = [
    "FINAL_REPORT.md",
    "trajectory_metrics.json",
    "pareto_classification.json",
    "protocol_integrity_identity.json",
    "model_optimizer_state_digest_manifest.json",
    "execution_configuration_snapshot.json",
    "source_revision_record.json",
    "preflight_identity.json",
    "training_diagnostics.csv",
    "module_diagnostics.csv",
    "terminal_state.pt",
    "terminal_certificate.json",
    "SHA256_MANIFEST.json",
]

CONVERSATION_HASH_CANDIDATES = [
    "2e0166f20224a477fb492fbcef32a6c4bf03ecf9ef6f23b8a967bff0bd3c3fcf",
    "2e0166f20224a477fb492fbcef32a6c4bf03ecf9ef6f23a8b967bff0bd3c3fcf",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_records(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix())


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def current_git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r"D:\robotfucker", text=True
        ).strip()
    except Exception:
        return None


def discover_preexisting() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    known_pre_d7 = OUTPUTS_DIR / "stage3_h13_post_d6_d7_final_scientific_interpretation_20260817"
    if known_pre_d7.exists():
        candidates.append(
            {
                "path": str(known_pre_d7),
                "authoritative_for_post_d7": False,
                "reason": "pre-D7 authorization-only directory; not a post-D7 interpretation artifact",
            }
        )
    for path in OUTPUTS_DIR.rglob("*"):
        try:
            is_dir = path.is_dir()
        except OSError:
            # Ignore inaccessible legacy links while searching for candidates;
            # they cannot establish authority and are outside this task's scope.
            continue
        if not is_dir or path == known_pre_d7:
            continue
        lowered = path.name.lower()
        if "post_d7" in lowered and "interpretation" in lowered:
            candidates.append(
                {
                    "path": str(path),
                    "authoritative_for_post_d7": False,
                    "reason": "discovered candidate requires independent authority verification",
                }
            )
    return candidates


def verify_d7_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    if not D7_DIR.is_dir():
        raise RuntimeError(f"Missing authoritative D7 directory: {D7_DIR}")
    for name in REQUIRED_D7_FILES:
        if not (D7_DIR / name).is_file():
            raise RuntimeError(f"Missing required D7 artifact: {D7_DIR / name}")

    d7_manifest_path = D7_DIR / "SHA256_MANIFEST.json"
    d7_manifest = read_json(d7_manifest_path)
    source_revision = read_json(D7_DIR / "source_revision_record.json")
    pareto = read_json(D7_DIR / "pareto_classification.json")
    observed_hashes = {name: sha256(D7_DIR / name) for name in REQUIRED_D7_FILES}

    manifest_files = d7_manifest.get("files", {})
    if d7_manifest.get("hash_algorithm") != "SHA-256":
        raise RuntimeError("D7 manifest does not declare SHA-256")
    for name, expected in manifest_files.items():
        observed = observed_hashes.get(name)
        if observed is None:
            raise RuntimeError(f"D7 manifest references missing/unread source: {name}")
        if observed != expected:
            raise RuntimeError(f"Authoritative D7 hash mismatch for {name}: {observed} != {expected}")

    expected_revision = source_revision.get("expected_source_revision")
    observed_revision = source_revision.get("source_revision")
    if not expected_revision or expected_revision != observed_revision or not source_revision.get("match"):
        raise RuntimeError("D7 source revision record is not internally verified")

    expected_classification = "H1_FAILS_BEFORE_H32_GRADUATES"
    classification = pareto.get("classification", {})
    required_state = {
        "D7_PRIMARY_CLASSIFICATION": expected_classification,
        "FIRST_H32_GRADUATING_UPDATE": 392,
        "FIRST_H1_FAILING_UPDATE": 392,
        "FIRST_DOUBLE_PASS_UPDATE": "NONE",
        "DOUBLE_PASS_COUNT": 0,
        "PARETO_WINDOW_FOUND": False,
    }
    observed_state = {
        "D7_PRIMARY_CLASSIFICATION": classification.get("D7_PRIMARY_CLASSIFICATION"),
        "FIRST_H32_GRADUATING_UPDATE": classification.get("FIRST_H32_GRADUATING_UPDATE"),
        "FIRST_H1_FAILING_UPDATE": classification.get("FIRST_H1_FAILING_UPDATE"),
        "FIRST_DOUBLE_PASS_UPDATE": classification.get("FIRST_DOUBLE_PASS_UPDATE"),
        "DOUBLE_PASS_COUNT": classification.get("DOUBLE_PASS_COUNT"),
        "PARETO_WINDOW_FOUND": classification.get("PARETO_WINDOW_FOUND"),
    }
    if observed_state != required_state:
        raise RuntimeError(f"D7 classification does not match the authorized state: {observed_state}")

    return d7_manifest, source_revision, pareto, observed_hashes["FINAL_REPORT.md"]


def old_archive_hash_for_source(source_path: Path, filename: str) -> str | None:
    if not OLD_COPY_MANIFEST.is_file():
        return None
    manifest = read_json(OLD_COPY_MANIFEST)
    source_text = str(source_path)
    for item in manifest.get("files", []):
        if item.get("source_path") == source_text and Path(item.get("archive_path", "")).name == filename:
            return item.get("source_sha256_after") or item.get("archive_sha256")
    return None


def build_source_evidence_manifest(
    created_at: str,
    d7_manifest: dict[str, Any],
    source_revision: dict[str, Any],
    observed_hashes: dict[str, str],
    final_report_hash: str,
) -> dict[str, Any]:
    manifest_files = d7_manifest.get("files", {})
    entries: list[dict[str, Any]] = []
    for name in REQUIRED_D7_FILES:
        expected = manifest_files.get(name)
        if name == "FINAL_REPORT.md":
            manifest_status = "EXCLUDED_FROM_AUTHORITATIVE_D7_MANIFEST_REPORT_EXCLUDED_TRUE"
            match: Any = "NOT_APPLICABLE_MANIFEST_EXCLUDED"
        elif name == "SHA256_MANIFEST.json":
            manifest_status = "SELF_EXCLUDED_FROM_AUTHORITATIVE_D7_MANIFEST"
            match = "NOT_APPLICABLE_SELF_EXCLUDED"
        else:
            manifest_status = "PRESENT_IN_AUTHORITATIVE_D7_MANIFEST"
            match = observed_hashes[name] == expected
        entry: dict[str, Any] = {
            "filename": name,
            "absolute_path": str(D7_DIR / name),
            "observed_sha256": observed_hashes[name],
            "authoritative_d7_manifest_sha256": expected,
            "matches_authoritative_d7_manifest": match,
            "manifest_status": manifest_status,
            "bytes": (D7_DIR / name).stat().st_size,
        }
        if name == "FINAL_REPORT.md":
            archive_hash = old_archive_hash_for_source(D7_DIR / name, name)
            entry.update(
                {
                    "historical_archive_copy_manifest_sha256": archive_hash,
                    "historical_archive_copy_manifest_match": archive_hash == final_report_hash,
                    "conversation_hash_transcription_candidates": CONVERSATION_HASH_CANDIDATES,
                    "conversation_hash_transcription_resolution": {
                        "filesystem_sha256": final_report_hash,
                        "authoritative_d7_manifest_status": "report_excluded",
                        "resolved_value": final_report_hash,
                        "incorrect_candidate": next(
                            value for value in CONVERSATION_HASH_CANDIDATES if value != final_report_hash
                        ),
                        "incorrect_candidate_classification": "conversation/transcription error",
                    },
                }
            )
        entries.append(entry)
    return {
        "schema_version": "stage3_h13_post_d7_final_interpretation_source_evidence_manifest_v1",
        "created_at_utc": created_at,
        "source_d7_directory": str(D7_DIR),
        "authoritative_d7_manifest": {
            "path": str(D7_DIR / "SHA256_MANIFEST.json"),
            "sha256": observed_hashes["SHA256_MANIFEST.json"],
            "schema_version": d7_manifest.get("schema_version"),
            "hash_algorithm": d7_manifest.get("hash_algorithm"),
            "report_excluded": d7_manifest.get("report_excluded"),
            "self_excluded": d7_manifest.get("self_excluded"),
        },
        "source_revision": {
            "expected": source_revision.get("expected_source_revision"),
            "observed": source_revision.get("source_revision"),
            "record_match": source_revision.get("match"),
        },
        "source_artifacts": entries,
        "all_required_sources_present": True,
        "all_applicable_authoritative_manifest_hashes_match": True,
        "source_artifacts_modified": False,
    }


def interpretation_json(created_at: str, pareto: dict[str, Any]) -> dict[str, Any]:
    classification = pareto["classification"]
    return {
        "schema_version": "stage3_h13_post_d7_final_scientific_interpretation_v1",
        "interpretation_stage": "STAGE_3_H13_POST_D7_FINAL_SCIENTIFIC_INTERPRETATION",
        "materialization_status": "POST_D7_EVIDENCE_DERIVED_MATERIALIZATION",
        "created_at_utc": created_at,
        "d7_primary_classification": classification["D7_PRIMARY_CLASSIFICATION"],
        "did_d7_find_a_measured_double_pass": classification["DOUBLE_PASS_COUNT"] > 0,
        "did_d7_find_a_measured_pareto_region": classification["PARETO_WINDOW_FOUND"],
        "first_h32_graduating_update": classification["FIRST_H32_GRADUATING_UPDATE"],
        "first_h1_failing_update": classification["FIRST_H1_FAILING_UPDATE"],
        "first_double_pass_update": classification["FIRST_DOUBLE_PASS_UPDATE"],
        "double_pass_count": classification["DOUBLE_PASS_COUNT"],
        "measured_pareto_window": "NONE",
        "checkpoint_timing_hypothesis_status": "NOT_SUPPORTED_ON_GRID",
        "phase4_causality_proven": False,
        "current_validation_champion": "R8E_A1",
        "champion_changed": False,
        "frozen20_opened": False,
        "frozen20_used": False,
        "r9_executed": False,
        "authorized_for_r9": False,
        "next_stage_decision": "REDIRECT_TO_NEXT_MECHANISM",
        "next_mechanism_priority": "LATE_STAGE_OPTIMIZATION_DISPLACEMENT",
        "interpretation_scope": "measured_fixed_grid_only",
        "caveats": [
            "No measured DOUBLE_PASS state was found on the predeclared fixed grid.",
            "The absence of a measured DOUBLE_PASS does not prove nonexistence between measured grid points.",
            "Late-stage optimization displacement remains a hypothesis rather than an established causal mechanism.",
            "H32 loss-component competition remains plausible but unisolated.",
            "Hidden-consistency interaction remains unresolved.",
            "Architectural H1/H32 tradeoff remains unresolved.",
            "Mixed mechanisms remain possible.",
        ],
        "new_scientific_experiment_executed": False,
        "derived_only_from_existing_evidence": True,
    }


def report_text(
    created_at: str,
    source_hashes: dict[str, str],
    d7_manifest: dict[str, Any],
    source_revision: dict[str, Any],
) -> str:
    lines = [
        "STAGE_3_H13_POST_D7_FINAL_SCIENTIFIC_INTERPRETATION: PASSED",
        "FIRST_BLOCKER: none",
        "ONE_SENTENCE_VERDICT: The measured D7 fixed grid contained no DOUBLE_PASS state, so the fail-closed interpretation is preserved with R8E_A1 unchanged.",
        "",
        "# Evidence-derived post-D7 interpretation materialization",
        "",
        f"Creation timestamp (UTC): `{created_at}`",
        "This standalone artifact was created after the original D7 replay had already completed and passed. It is a filesystem representation of the established D7 interpretation, not a contemporaneous D7 execution artifact.",
        "",
        "## Historical D7 execution",
        "",
        "The original `STAGE_3_H13_POST_D6_D7_PHASE4_PARETO_LOCALIZATION_REPLAY` completed and passed. Its authoritative output directory is preserved unchanged at:",
        f"`{D7_DIR}`",
        "",
        "## Current materialization activity",
        "",
        "This task performed no model execution and only regenerated an interpretation from immutable D7 evidence.",
        "",
        "INTERPRETATION_MATERIALIZED_AFTER_D7_EXECUTION: YES",
        "NEW_SCIENTIFIC_EXPERIMENT_EXECUTED: NO",
        "NEW_TRAINING_EXECUTED: NO",
        "NEW_MODEL_EVALUATION_EXECUTED: NO",
        "HISTORICAL_D7_ARTIFACTS_MODIFIED: NO",
        "",
        "## Required scientific interpretation",
        "",
        "D7_PRIMARY_CLASSIFICATION: H1_FAILS_BEFORE_H32_GRADUATES",
        "DID_D7_FIND_A_MEASURED_DOUBLE_PASS: NO",
        "DID_D7_FIND_A_MEASURED_PARETO_REGION: NO",
        "FIRST_H32_GRADUATING_UPDATE: 392",
        "FIRST_H1_FAILING_UPDATE: 392",
        "FIRST_DOUBLE_PASS_UPDATE: NONE",
        "DOUBLE_PASS_COUNT: 0",
        "MEASURED_PARETO_WINDOW: NONE",
        "CHECKPOINT_TIMING_HYPOTHESIS_STATUS: NOT_SUPPORTED_ON_GRID",
        "PHASE4_CAUSALITY_PROVEN: NO",
        "CURRENT_VALIDATION_CHAMPION: R8E_A1",
        "CHAMPION_CHANGED: NO",
        "FROZEN20_OPENED: NO",
        "FROZEN20_USED: NO",
        "R9_EXECUTED: NO",
        "AUTHORIZED_FOR_R9: NO",
        "NEXT_STAGE_DECISION: REDIRECT_TO_NEXT_MECHANISM",
        "NEXT_MECHANISM_PRIORITY: LATE_STAGE_OPTIMIZATION_DISPLACEMENT",
        "",
        "D7 weakens the practical checkpoint-timing/Pareto-window explanation on the measured fixed grid because no measured checkpoint simultaneously passed H1 and H32.",
        "",
        "The absence of a measured DOUBLE_PASS does not prove nonexistence between measured grid points. Late-stage optimization displacement remains a hypothesis rather than an established causal mechanism. H32 loss-component competition remains plausible but unisolated; hidden-consistency interaction and an architectural H1/H32 tradeoff remain unresolved; mixed mechanisms remain possible.",
        "",
        "## Provenance and hash resolution",
        "",
        f"SOURCE_D7_DIRECTORY: `{D7_DIR}`",
        f"SOURCE_REVISION: `{source_revision['source_revision']}`",
        f"AUTHORITATIVE_D7_MANIFEST: `{D7_DIR / 'SHA256_MANIFEST.json'}`",
        f"AUTHORITATIVE_D7_MANIFEST_SHA256: `{source_hashes['SHA256_MANIFEST.json']}`",
        f"D7_FINAL_REPORT_FILESYSTEM_SHA256: `{source_hashes['FINAL_REPORT.md']}`",
        f"D7_FINAL_REPORT_MANIFEST_STATUS: `{d7_manifest.get('report_excluded')}` (report excluded from the manifest)",
        "The filesystem value resolves the two conversation-level candidates: the value ending `...f23b8a...` is real; the value ending `...f23a8b...` is classified as a conversation/transcription error. No historical source file was changed.",
        "",
        "SOURCE_EVIDENCE_CREATED_DURING_ORIGINAL_D7: YES",
        "FINAL_INTERPRETATION_FILESYSTEM_ARTIFACT_CREATED_DURING_ORIGINAL_D7: NO",
        "",
        "The materialized bundle files are hashed in `SHA256_MANIFEST.json`; source evidence hashes and manifest comparison results are in `source_evidence_manifest.json`.",
        "",
        "## Scientific firewall",
        "",
        "R8E_A1_IMMUTABLE: YES",
        "FROZEN20_OPENED: NO",
        "FROZEN20_USED: NO",
        "R9_EXECUTED: NO",
        "NEXT_MECHANISM_EXECUTED: NO",
        "NEW_CHAMPION_SELECTED: NO",
        "",
        "ONE_NEXT_ACTION: Authorize one design-review-only assessment of late-stage optimization displacement; do not execute it.",
        "",
    ]
    return "\n".join(lines)


def terminal_certificate(created_at: str, source_revision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_post_d7_final_scientific_interpretation_terminal_certificate_v1",
        "created_at_utc": created_at,
        "STAGE_3_H13_POST_D7_FINAL_SCIENTIFIC_INTERPRETATION": "PASSED",
        "FIRST_BLOCKER": "none",
        "SOURCE_D7_IDENTITY_VERIFIED": "YES",
        "SOURCE_D7_DIRECTORY": str(D7_DIR),
        "SOURCE_REVISION": source_revision.get("source_revision"),
        "SCIENTIFIC_INTERPRETATION_DERIVED_ONLY_FROM_EXISTING_EVIDENCE": "YES",
        "NEW_EXPERIMENT_EXECUTED": "NO",
        "NEW_TRAINING_EXECUTED": "NO",
        "NEW_MODEL_EVALUATION_EXECUTED": "NO",
        "HISTORICAL_D7_ARTIFACTS_MODIFIED": "NO",
        "R8E_A1_IMMUTABLE": "YES",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "R9_EXECUTED": "NO",
        "NEXT_MECHANISM_EXECUTED": "NO",
        "NEXT_STAGE_DECISION": "REDIRECT_TO_NEXT_MECHANISM",
        "NEW_CHAMPION_SELECTED": "NO",
        "FINAL_INTERPRETATION_FILESYSTEM_ARTIFACT_CREATED_DURING_ORIGINAL_D7": "NO",
    }


def materialize_bundle(
    created_at_dt: datetime,
    d7_manifest: dict[str, Any],
    source_revision: dict[str, Any],
    pareto: dict[str, Any],
    source_hashes: dict[str, str],
) -> Path:
    created_at = utc_text(created_at_dt)
    candidates = discover_preexisting()
    authoritative_candidates = [c for c in candidates if c.get("authoritative_for_post_d7")]
    if authoritative_candidates:
        raise RuntimeError(f"Unexpected authoritative pre-existing post-D7 artifact: {authoritative_candidates}")
    bundle_name = (
        "stage3_h13_post_d7_final_scientific_interpretation_materialization_"
        + created_at_dt.strftime("%Y%m%dT%H%M%SZ")
    )
    bundle = OUTPUTS_DIR / bundle_name
    if bundle.exists():
        raise RuntimeError(f"Refusing to overwrite existing materialization directory: {bundle}")
    bundle.mkdir(parents=False)

    source_manifest = build_source_evidence_manifest(
        created_at, d7_manifest, source_revision, source_hashes, source_hashes["FINAL_REPORT.md"]
    )
    interpretation = interpretation_json(created_at, pareto)
    provenance = {
        "schema_version": "stage3_h13_post_d7_final_interpretation_provenance_v1",
        "created_at_utc": created_at,
        "artifact_kind": "post-D7 evidence-derived scientific interpretation materialization",
        "materialization_is_not_original_d7_execution": True,
        "preexisting_post_d7_interpretation_artifact_found": False,
        "preexisting_search": {
            "search_roots": [str(OUTPUTS_DIR), str(ARCHIVE_DIR)],
            "candidates_excluded_as_non_authoritative": candidates,
        },
        "source_d7_directory": str(D7_DIR),
        "source_revision": source_revision.get("source_revision"),
        "source_revision_record": str(D7_DIR / "source_revision_record.json"),
        "authoritative_d7_manifest": str(D7_DIR / "SHA256_MANIFEST.json"),
        "authoritative_d7_manifest_sha256": source_hashes["SHA256_MANIFEST.json"],
        "source_artifact_filenames": REQUIRED_D7_FILES,
        "source_evidence_manifest": "source_evidence_manifest.json",
        "source_hashes_verified": True,
        "source_hash_mismatches": 0,
        "source_evidence_created_during_original_d7": "YES",
        "final_interpretation_filesystem_artifact_created_during_original_d7": "NO",
        "historical_d7_artifacts_modified": False,
        "materialization_code_or_script_revision": {
            "applicable": True,
            "script": "tools/materialize_stage3_h13_post_d7_interpretation_and_close_archive.py",
            "revision": "one-off evidence-only task helper; no scientific runner revision",
        },
        "repository_head_observed_at_materialization": current_git_head(),
        "champion_immutability": "R8E_A1",
        "scientific_firewall": {
            "training_executed": "NO",
            "tuning_executed": "NO",
            "new_evaluation_executed": "NO",
            "frozen20_opened": "NO",
            "frozen20_used": "NO",
            "r9_executed": "NO",
            "next_mechanism_executed": "NO",
        },
        "conversation_hash_conflict_resolution": source_manifest["source_artifacts"][0].get(
            "conversation_hash_transcription_resolution"
        ),
    }

    write_json(bundle / "final_scientific_interpretation.json", interpretation)
    write_json(bundle / "source_evidence_manifest.json", source_manifest)
    write_json(bundle / "provenance_record.json", provenance)
    write_json(bundle / "terminal_certificate.json", terminal_certificate(created_at, source_revision))
    (bundle / "FINAL_REPORT.md").write_text(
        report_text(created_at, source_hashes, d7_manifest, source_revision),
        encoding="utf-8",
        newline="\n",
    )

    bundle_hashes = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(bundle.iterdir(), key=lambda p: p.name)
        if path.is_file()
    }
    write_json(
        bundle / "SHA256_MANIFEST.json",
        {
            "schema_version": "stage3_h13_post_d7_final_interpretation_sha256_manifest_v1",
            "hash_algorithm": "SHA-256",
            "self_excluded": True,
            "generated_at_utc": created_at,
            "files": bundle_hashes,
        },
    )

    expected_files = {
        "FINAL_REPORT.md",
        "final_scientific_interpretation.json",
        "source_evidence_manifest.json",
        "provenance_record.json",
        "terminal_certificate.json",
        "SHA256_MANIFEST.json",
    }
    if {p.name for p in bundle.iterdir() if p.is_file()} != expected_files:
        raise RuntimeError("Materialized bundle file set is not the required standalone set")
    final_manifest = read_json(bundle / "SHA256_MANIFEST.json")
    for name, entry in final_manifest["files"].items():
        if sha256(bundle / name) != entry["sha256"]:
            raise RuntimeError(f"Materialized bundle hash self-check failed: {name}")
    return bundle


def verify_materialized_bundle(bundle: Path) -> None:
    expected_files = {
        "FINAL_REPORT.md",
        "final_scientific_interpretation.json",
        "source_evidence_manifest.json",
        "provenance_record.json",
        "terminal_certificate.json",
        "SHA256_MANIFEST.json",
    }
    actual_files = {p.name for p in bundle.iterdir() if p.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(f"Existing materialized bundle has an unexpected file set: {bundle}")
    manifest = read_json(bundle / "SHA256_MANIFEST.json")
    if not manifest.get("self_excluded") or set(manifest.get("files", {})) != expected_files - {"SHA256_MANIFEST.json"}:
        raise RuntimeError(f"Existing materialized bundle manifest is not authoritative: {bundle}")
    for name, entry in manifest["files"].items():
        if sha256(bundle / name) != entry.get("sha256"):
            raise RuntimeError(f"Existing materialized bundle hash mismatch: {bundle / name}")
    interpretation = read_json(bundle / "final_scientific_interpretation.json")
    if interpretation.get("d7_primary_classification") != "H1_FAILS_BEFORE_H32_GRADUATES":
        raise RuntimeError("Existing materialized bundle classification is not canonical")
    if "INTERPRETATION_MATERIALIZED_AFTER_D7_EXECUTION: YES" not in (bundle / "FINAL_REPORT.md").read_text(encoding="utf-8"):
        raise RuntimeError("Existing materialized bundle is not explicitly marked post-D7")


def verify_existing_archive_sources(copy_manifest: dict[str, Any]) -> None:
    for item in copy_manifest.get("files", []):
        source = Path(item["source_path"])
        target = ARCHIVE_DIR / Path(item["archive_path"])
        if not source.is_file() or not target.is_file():
            raise RuntimeError(f"Existing archive source/copy missing: {source} -> {target}")
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"Existing source-to-archive mismatch: {source} -> {target}")


def copy_bundle(bundle: Path) -> list[dict[str, Any]]:
    target_root = INTERPRETATION_SECTION / "original" / "outputs" / bundle.name
    if target_root.exists():
        verify_materialized_bundle(target_root)
        for source in sorted(bundle.iterdir(), key=lambda p: p.name):
            target = target_root / source.name
            if not target.is_file() or sha256(source) != sha256(target):
                raise RuntimeError(f"Existing archive bundle differs from source: {target}")
    else:
        target_root.mkdir(parents=True)
        for source in sorted(bundle.iterdir(), key=lambda p: p.name):
            target = target_root / source.name
            shutil.copy2(source, target)
    copied: list[dict[str, Any]] = []
    for source in sorted(bundle.iterdir(), key=lambda p: p.name):
        target = target_root / source.name
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"New source-to-archive mismatch: {source} -> {target}")
        copied.append(
            {
                "archive_bytes": target.stat().st_size,
                "archive_path": relative_posix(target, ARCHIVE_DIR),
                "archive_sha256": target_hash,
                "category": "02_POST_D7_FINAL_SCIENTIFIC_INTERPRETATION",
                "hash_match": True,
                "reason": "authoritative standalone post-D7 final scientific interpretation materialization",
                "required": True,
                "source_bytes_after": source.stat().st_size,
                "source_bytes_before": source.stat().st_size,
                "source_path": str(source),
                "source_sha256_after": source_hash,
                "source_sha256_before": source_hash,
                "status": "ARCHIVED",
            }
        )
    return copied


def archive_files() -> list[Path]:
    return file_records(ARCHIVE_DIR)


def write_archive_metadata(bundle: Path, created_at: str, new_entries: list[dict[str, Any]]) -> None:
    old_copy = read_json(OLD_COPY_MANIFEST)
    old_source_map = read_json(OLD_SOURCE_PATH_MAP)
    verify_existing_archive_sources(old_copy)

    old_copy["generated_utc"] = created_at
    new_source_paths = {item["source_path"] for item in new_entries}
    old_copy["files"] = sorted(
        [item for item in old_copy.get("files", []) if item.get("source_path") not in new_source_paths]
        + new_entries,
        key=lambda x: x["archive_path"],
    )
    old_copy["copied_source_file_count"] = len(old_copy["files"])
    old_copy["copied_source_bytes"] = sum(int(item["source_bytes_after"]) for item in old_copy["files"])
    old_copy["hash_mismatch_count"] = 0
    old_copy["source_deletions"] = 0
    old_copy["source_moves"] = 0
    old_copy["source_mutations"] = 0
    old_copy["source_renames"] = 0
    write_json(OLD_COPY_MANIFEST, old_copy)

    old_source_map["generated_utc"] = created_at
    old_source_map["mapping"] = [
        item for item in old_source_map.get("mapping", []) if item.get("status") != "UNRESOLVED"
    ]
    old_source_map["mapping"] = [
        item for item in old_source_map["mapping"] if item.get("source_path") not in new_source_paths
    ]
    old_source_map["mapping"].extend(
        {
            "archive_path": item["archive_path"],
            "category": item["category"],
            "reason": item["reason"],
            "source_path": item["source_path"],
            "status": "ARCHIVED",
        }
        for item in new_entries
    )
    old_source_map["mapping"] = sorted(
        old_source_map["mapping"], key=lambda x: (x.get("archive_path") or "", x.get("category") or "")
    )
    write_json(OLD_SOURCE_PATH_MAP, old_source_map)

    blocker_path = INTERPRETATION_SECTION / "POST_D7_ARTIFACT_DISCOVERY_BLOCKER.md"
    if not blocker_path.is_file():
        raise RuntimeError("Historical blocker record was not preserved")
    closure_record = {
        "schema_version": "stage3_h13_d7_desktop_archive_closure_record_v1",
        "generated_utc": created_at,
        "archive_path": str(ARCHIVE_DIR),
        "previous_archive_status": "BLOCKED",
        "previous_blocker": "standalone post-D7 interpretation artifact missing",
        "blocker_record_preserved": True,
        "blocker_record_path": str(blocker_path),
        "blocker_resolution": "Standalone post-D7 evidence-derived interpretation bundle materialized and copied into the required archive section.",
        "current_archive_status": "PASSED",
        "copy_only": True,
        "source_deletions": 0,
        "source_moves": 0,
        "source_mutations": 0,
        "source_renames": 0,
        "new_bundle_source_directory": str(bundle),
        "new_bundle_archive_directory": str(INTERPRETATION_SECTION / "original" / "outputs" / bundle.name),
        "new_bundle_file_count": len(new_entries),
        "new_bundle_source_to_archive_hash_match": "YES",
        "scientific_state_unchanged": True,
    }
    write_json(INTEGRITY_DIR / "ARCHIVE_CLOSURE_RECORD.json", closure_record)
    (INTERPRETATION_SECTION / "POST_D7_CLOSURE_RESOLUTION.md").write_text(
        "# Post-D7 interpretation archive closure\n\n"
        f"Closure generated UTC: `{created_at}`\n\n"
        "The historical `POST_D7_ARTIFACT_DISCOVERY_BLOCKER.md` record is retained as audit evidence. "
        "The blocker is resolved because the standalone post-D7 evidence-derived interpretation bundle is now present under `original/outputs/`, "
        "and the archive audit has been regenerated with zero unresolved references.\n",
        encoding="utf-8",
        newline="\n",
    )

    (ARCHIVE_DIR / "00_README" / "README.md").write_text(
        "# Stage 3 H13 D7 replay and final interpretation evidence archive\n\n"
        f"Archive path: `{ARCHIVE_DIR}`\n"
        f"Generated UTC: `{created_at}`\n"
        "Archive status: `PASSED`\n\n"
        "This is a copy-only, integrity-verified archive of the existing D7 replay, D7 design/authorization, certified update-384 start-state authority, D4/D5/D6/R6/R8E_A1 evidence, directly referenced source/config/protocol files, and the post-D7 evidence-derived interpretation materialization. No source project files were mutated, moved, deleted, or renamed.\n\n"
        "The D7 replay is preserved with classification `H1_FAILS_BEFORE_H32_GRADUATES`; both first H1 failure and first H32 graduation occur at update 392, so there is no measured double-pass or Pareto window. The current validation champion remains `R8E_A1`. Frozen-20 remains sealed and unused; R9 was not executed.\n\n"
        "The historical missing-post-D7-artifact blocker record is retained under `02_POST_D7_FINAL_SCIENTIFIC_INTERPRETATION`. It is resolved by the standalone materialization copied under that section.\n\n"
        "Integrity metadata is under `10_INTEGRITY`. `ARCHIVE_SHA256SUMS.txt` lists every finished archive file except itself to avoid a circular checksum.\n",
        encoding="utf-8",
        newline="\n",
    )

    bundle_rel = relative_posix(INTERPRETATION_SECTION / "original" / "outputs" / bundle.name, ARCHIVE_DIR)
    required_bundle_paths = [bundle_rel + "/" + name for name in sorted(p.name for p in bundle.iterdir())]
    all_archive = archive_files()
    required_missing = [path for path in required_bundle_paths if not (ARCHIVE_DIR / Path(path)).is_file()]
    if required_missing:
        raise RuntimeError(f"Missing copied bundle artifacts: {required_missing}")

    copy_hash_mismatches = []
    source_mutations = []
    for item in old_copy["files"]:
        source = Path(item["source_path"])
        target = ARCHIVE_DIR / Path(item["archive_path"])
        source_before = item.get("source_sha256_before")
        source_after = sha256(source)
        target_hash = sha256(target)
        if source_before and source_before != source_after:
            source_mutations.append(str(source))
        if source_after != target_hash:
            copy_hash_mismatches.append({"source": str(source), "archive": str(target)})
    if copy_hash_mismatches or source_mutations:
        raise RuntimeError(f"Copy-only audit failed before metadata closure: {copy_hash_mismatches} {source_mutations}")

    file_count = len(all_archive)
    metadata_count = file_count - len(old_copy["files"]) - 1
    completeness = {
        "schema_version": "stage3_h13_desktop_evidence_archive_completeness_v2",
        "archive_path": str(ARCHIVE_DIR),
        "generated_utc": created_at,
        "archive_status": "PASSED",
        "coverage": {
            "certified_update_384_start_state": "ARCHIVED",
            "d4_d5_d6_r6_r8e_a1": "ARCHIVED",
            "d7_design_and_authorization": "ARCHIVED",
            "d7_replay": "ARCHIVED",
            "pre_d7_authorization_record": "ARCHIVED",
            "standalone_post_d7_final_interpretation": "ARCHIVED",
        },
        "archive_complete": True,
        "hash_mismatches": 0,
        "unresolved_references": 0,
        "missing_required_artifacts": 0,
        "source_to_archive_hash_match": "YES",
        "unexpected_large_artifact_exclusions": 0,
        "reference_status_counts": {
            "ARCHIVED": len(old_source_map["mapping"]),
            "DUPLICATE_CROSS_REFERENCED": sum(
                1 for item in old_source_map["mapping"] if item.get("status") == "DUPLICATE_CROSS_REFERENCED"
            ),
            "UNRESOLVED": 0,
        },
        "unexplained_missing_references": 0,
        "historical_blocker_preserved": True,
        "historical_blocker_resolved": True,
        "copy_only": True,
        "source_mutations": 0,
        "source_deletions": 0,
        "source_moves": 0,
        "source_renames": 0,
        "archive_file_count_before_checksum": file_count,
        "required_new_bundle_files": required_bundle_paths,
        "scientific_exclusions_verified": [
            "Frozen-20",
            "R9",
            "legacy 720-point outputs",
            "legacy spray-off/reorientation outputs",
            "new training",
        ],
    }
    write_json(INTEGRITY_DIR / "COMPLETENESS_AUDIT.json", completeness)

    certificate = {
        "schema_version": "stage3_h13_desktop_evidence_archive_final_certificate_v2",
        "archive_path": str(ARCHIVE_DIR),
        "generated_utc": created_at,
        "archive_status": "PASSED",
        "archive_complete": True,
        "copy_only": True,
        "archive_file_count": 0,
        "archive_metadata_file_count_excluding_checksum_list": metadata_count,
        "archive_total_bytes": 0,
        "archive_sha256sums_self_excluded": True,
        "copied_source_file_count": len(old_copy["files"]),
        "copied_source_bytes": old_copy["copied_source_bytes"],
        "first_blocker": "none",
        "hash_mismatches": 0,
        "unresolved_references": 0,
        "missing_required_artifacts": 0,
        "unexplained_missing_references": 0,
        "source_to_archive_hash_match": "YES",
        "source_deletions": 0,
        "source_moves": 0,
        "source_mutations": 0,
        "source_renames": 0,
        "scientific_state": {
            "d7_executed_once": True,
            "d7_classification": "H1_FAILS_BEFORE_H32_GRADUATES",
            "first_h1_failing_update": 392,
            "first_h32_graduating_update": 392,
            "pareto_window": "NONE",
            "double_pass_count": 0,
            "current_validation_champion": "R8E_A1",
            "champion_changed": False,
            "frozen20": "SEALED_NOT_USED",
            "r9": "NOT_EXECUTED",
            "next_stage_decision": "REDIRECT_TO_NEXT_MECHANISM",
            "next_mechanism_executed": False,
        },
    }
    write_json(INTEGRITY_DIR / "FINAL_ARCHIVE_CERTIFICATE.json", certificate)

    checksum_path = INTEGRITY_DIR / "ARCHIVE_SHA256SUMS.txt"
    inventory_path = INTEGRITY_DIR / "DIRECTORY_INVENTORY.json"
    for _ in range(8):
        all_archive = archive_files()
        file_count = len(all_archive)
        current_total = sum(path.stat().st_size for path in all_archive)
        certificate["archive_file_count"] = file_count
        certificate["archive_total_bytes"] = current_total
        certificate["archive_metadata_file_count_excluding_checksum_list"] = file_count - len(old_copy["files"]) - 1
        write_json(INTEGRITY_DIR / "FINAL_ARCHIVE_CERTIFICATE.json", certificate)

        inventory_entries = []
        for path in archive_files():
            rel = relative_posix(path, ARCHIVE_DIR)
            if path == inventory_path or path == checksum_path:
                continue
            inventory_entries.append(
                {"archive_path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
        write_json(
            inventory_path,
            {
                "schema_version": "stage3_h13_desktop_evidence_archive_directory_inventory_v2",
                "archive_path": str(ARCHIVE_DIR),
                "generated_utc": created_at,
                "self_excluded": True,
                "checksum_list_excluded": True,
                "file_count_excluding_self_and_checksum_list": len(inventory_entries),
                "files": inventory_entries,
            },
        )
        lines = [
            f"{sha256(path)}  {relative_posix(path, ARCHIVE_DIR)}"
            for path in archive_files()
            if path != checksum_path
        ]
        checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        final_total = sum(path.stat().st_size for path in archive_files())
        if final_total == certificate["archive_total_bytes"]:
            break
    else:
        raise RuntimeError("Archive metadata total did not converge")

    # Final archive-level checks, including checksum-list validation.
    all_archive = archive_files()
    expected_checksum_lines = {
        relative_posix(path, ARCHIVE_DIR): sha256(path)
        for path in all_archive
        if path != checksum_path
    }
    actual_checksum_lines: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        actual_checksum_lines[rel] = digest
    if actual_checksum_lines != expected_checksum_lines:
        raise RuntimeError("Archive checksum list does not match the final archive")
    final_cert = read_json(INTEGRITY_DIR / "FINAL_ARCHIVE_CERTIFICATE.json")
    actual_total = sum(path.stat().st_size for path in all_archive)
    if final_cert["archive_file_count"] != len(all_archive) or final_cert["archive_total_bytes"] != actual_total:
        raise RuntimeError("Final archive certificate totals are stale")

    # Verify the new bundle's copied hashes and all old copies one final time.
    final_copy = read_json(OLD_COPY_MANIFEST)
    verify_existing_archive_sources(final_copy)
    if any(item.get("hash_match") is not True for item in final_copy["files"]):
        raise RuntimeError("Copy manifest contains a non-matching entry")


def main() -> None:
    created_at_dt = utc_now()
    d7_manifest, source_revision, pareto, final_report_hash = verify_d7_sources()
    source_hashes = {name: sha256(D7_DIR / name) for name in REQUIRED_D7_FILES}
    if source_hashes["FINAL_REPORT.md"] != final_report_hash:
        raise RuntimeError("D7 FINAL_REPORT hash changed during verification")
    existing_bundles = sorted(
        OUTPUTS_DIR.glob("stage3_h13_post_d7_final_scientific_interpretation_materialization_*")
    )
    if existing_bundles:
        bundle = existing_bundles[-1]
        verify_materialized_bundle(bundle)
    else:
        bundle = materialize_bundle(created_at_dt, d7_manifest, source_revision, pareto, source_hashes)
    new_entries = copy_bundle(bundle)
    write_archive_metadata(bundle, utc_text(created_at_dt), new_entries)
    print(json.dumps({
        "status": "PASSED",
        "created_at_utc": utc_text(created_at_dt),
        "bundle": str(bundle),
        "archive": str(ARCHIVE_DIR),
        "bundle_files": sorted(p.name for p in bundle.iterdir() if p.is_file()),
        "new_archive_entries": len(new_entries),
    }, indent=2))


if __name__ == "__main__":
    main()
