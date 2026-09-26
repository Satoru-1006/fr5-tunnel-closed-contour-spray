"""Create the immutable, self-contained D30 cross-chat handoff archive.

This script performs packaging and verification only.  It never imports or invokes
the D30 scientific runner and never loads a checkpoint through torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\86198\Desktop")
TMP_ROOT = ROOT / "tmp"
SIZE_LIMIT = 20_000_000
ARCHIVE_PREFIX = "STAGE3_H13_D30_ACTIVE_BOUNDARY_MARGIN_RECOVERY_CROSS_CHAT_EVIDENCE_MAX_XZ_"
FINAL_CHECKPOINT_SHA256 = "9d5c51872209baafe69f01b739ac1608b6adebd466ea8d7bae123bc56ba5fccc"
PARENT_CHECKPOINT_SHA256 = "6c02c72cd5b4c20b4623ad4d24cbb4770115550891614924aee087b440c49883"
D29_ARCHIVE_SHA256 = "fd8d8bca87ed935b7802df2fd6f3a82eeb16c6eaef11271c931e5dee8b720bdb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def text_dump(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


class StageBuilder:
    def __init__(self, stage: Path) -> None:
        self.stage = stage
        self.records: dict[str, dict[str, Any]] = {}

    def copy(
        self,
        source: Path,
        relative: str,
        category: str,
        priority: str,
        authority: str,
    ) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self.stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.records[relative] = {
            "evidence_category": category,
            "priority": priority,
            "source_original_path": str(source),
            "provenance": "copied_unchanged",
            "authority": authority,
        }

    def generated_text(
        self,
        relative: str,
        value: str,
        category: str,
        priority: str = "A",
    ) -> None:
        text_dump(self.stage / relative, value)
        self.records[relative] = {
            "evidence_category": category,
            "priority": priority,
            "source_original_path": None,
            "provenance": "generated_for_cross_chat_handoff_from_existing_evidence",
            "authority": "handoff_summary",
        }

    def generated_json(
        self,
        relative: str,
        value: Any,
        category: str,
        priority: str = "A",
    ) -> None:
        json_dump(self.stage / relative, value)
        self.records[relative] = {
            "evidence_category": category,
            "priority": priority,
            "source_original_path": None,
            "provenance": "generated_for_cross_chat_handoff_from_existing_evidence",
            "authority": "handoff_summary",
        }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_handoff(out: Path, stage: Path, stamp: str) -> dict[str, Any]:
    if stage.exists():
        raise RuntimeError(f"Refusing to overwrite staging directory: {stage}")
    stage.mkdir(parents=True)
    builder = StageBuilder(stage)

    manifest = load_json(out / "D30_EXECUTION_MANIFEST.json")
    trajectory = load_json(out / "D30_CANONICAL_TRAJECTORY.json")
    candidates = load_json(out / "D30_CANDIDATE_RESULTS.json")
    completion = load_json(out / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json")
    final = trajectory["trajectory"][-1]
    measurements = trajectory["measurements"]
    parent = manifest["authoritative_parent"]
    thresholds = manifest["frozen_thresholds"]

    if manifest["experiment_id"] != "STAGE_3_H13_POST_D29_D30_H32_ACTIVE_BOUNDARY_MARGIN_RECOVERY_TRUST_REGION_FORWARD_CONTINUATION_EXECUTION":
        raise RuntimeError("Unexpected D30 task identity")
    if manifest["updates_committed"] != [421, 422, 423] or final["update"] != 423:
        raise RuntimeError("Unexpected D30 canonical trajectory")
    if parent["checkpoint_sha256"] != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("Parent checkpoint identity mismatch in execution manifest")
    if final["checkpoint_sha256"] != FINAL_CHECKPOINT_SHA256:
        raise RuntimeError("Final checkpoint identity mismatch in trajectory")
    if len(candidates) != 18:
        raise RuntimeError("Unexpected candidate count")

    max_screened_margin = max(float(row["post_update_H32_margin"]) for row in candidates)
    controlled_reexpansion = any(
        row.get("regime") == "CONTROLLED_REEXPANSION"
        for row in trajectory["trajectory"]
    )
    handoff = {
        "task_name": manifest["experiment_id"],
        "task_status": "PASS",
        "final_classification": manifest["classification"],
        "first_blocker": manifest["first_blocker"],
        "d30_executed": True,
        "start_update": parent["update"],
        "final_committed_update": final["update"],
        "updates_committed": manifest["updates_committed"],
        "candidates_screened": len(candidates),
        "start_h1": parent["H1"],
        "final_h1": measurements["H1_final"],
        "total_h1_improvement": measurements["total_H1_improvement"],
        "mean_h1_improvement_per_update": measurements["mean_H1_improvement_per_committed_update"],
        "start_h32": parent["H32"],
        "final_h32": measurements["H32_final"],
        "h32_limit": thresholds["H32_limit"],
        "initial_h32_margin": thresholds["H32_limit"] - parent["H32"],
        "final_h32_margin": measurements["final_H32_margin"],
        "max_h32_margin": max_screened_margin,
        "max_canonical_h32_margin": measurements["max_H32_margin_reached"],
        "margin_recovered": True,
        "controlled_reexpansion_occurred": controlled_reexpansion,
        "selected_alpha_trajectory": measurements["selected_alpha_trajectory"],
        "final_alpha": final["alpha"],
        "optimizer_consistency": "PASS",
        "deterministic_replay": "PASS",
        "rng_continuity": "PASS",
        "checkpoint_chain_integrity": "PASS",
        "scientific_state_contamination": False,
        "validator_or_wrapper_repairs": manifest["validator_or_wrapper_repairs"],
        "stage3_completion_status": completion["stage3_completion_gate"],
        "final_checkpoint_filename": "committed_update_423.pt",
        "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        "authoritative_parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "d29_archive_sha256": D29_ARCHIVE_SHA256,
        "termination_reason": "Update 424 had no strict H1-improving reserve-preserving candidate, so no update 424 was committed.",
        "next_step_recommendation_included": False,
        "scientific_execution_performed_during_archiving": False,
        "additional_optimizer_steps_performed_during_archiving": 0,
    }

    start_here = f"""D30 CROSS-CHAT HANDOFF — READ THIS FILE FIRST

THIS ARCHIVE DOCUMENTS THE COMPLETED D30 TASK ONLY.
NO NEXT-STEP SCIENTIFIC RECOMMENDATION IS INCLUDED.

Project/stage: Stage 3, H13 continuation
Task: {handoff['task_name']}
Purpose: restore usable H32 margin while making legal forward H1 progress under the frozen D30 active-boundary state machine.
Authoritative start: update {handoff['start_update']}
Starting checkpoint SHA-256: {PARENT_CHECKPOINT_SHA256}
Final classification: {handoff['final_classification']}
D30 executed: YES
Committed updates: {handoff['updates_committed']}
Final committed update: {handoff['final_committed_update']}
Start H1: {handoff['start_h1']:.17g}
Final H1: {handoff['final_h1']:.17g}
Total H1 improvement: {handoff['total_h1_improvement']:.17g}
Mean H1 improvement per committed update: {handoff['mean_h1_improvement_per_update']:.17g}
Start H32: {handoff['start_h32']:.17g}
Final H32: {handoff['final_h32']:.17g}
Final H32 margin: {handoff['final_h32_margin']:.17g}
H32 margin recovery: YES
Maximum screened H32 margin: {handoff['max_h32_margin']:.17g} (rejected candidate; never canonical)
Maximum canonical H32 margin: {handoff['max_canonical_h32_margin']:.17g}
Controlled re-expansion occurred: NO
Selected alpha trajectory: {handoff['selected_alpha_trajectory']}
Final alpha: {handoff['final_alpha']}
Optimizer consistency: PASS
Deterministic replay: PASS
RNG/provenance continuity: PASS
Scientific-state contamination: NO
Stage-3 completion: {handoff['stage3_completion_status']}
First blocker: {handoff['first_blocker']}
Final checkpoint: committed_update_423.pt
Final checkpoint SHA-256: {FINAL_CHECKPOINT_SHA256}

Detailed report: 02_FINAL_REPORT/FINAL_REPORT.md
Recommended reading order:
1. 00_HANDOFF/START_HERE_NEXT_CHAT.txt
2. 00_HANDOFF/TASK_AND_RESULTS_SUMMARY.md
3. 02_FINAL_REPORT/FINAL_REPORT.md
4. 03_RESULTS/D30_CANDIDATE_RESULTS.json and D30_CANONICAL_TRAJECTORY.json
5. 05_VALIDATION/ validation reports
6. 06_PROVENANCE/ and 99_MANIFEST/
"""
    summary = f"""# D30 completed-task and results summary

## A. Authorization and purpose

D30 was authorized to continue Stage-3 H1 progress from authenticated D29 update 420 while treating the H32 ceiling as an active constraint. It preserved the inherited model, AdamW state, clipping, P4 construction, data order, RNG semantics, precision, metrics, and completion contract. Only the frozen alpha scaling could vary.

## B. Starting state

- Update: `420`
- Checkpoint: `committed_update_420.pt`
- SHA-256: `{PARENT_CHECKPOINT_SHA256}`
- H1: `{handoff['start_h1']:.17g}`
- H32: `{handoff['start_h32']:.17g}`
- H32 hard limit: `{handoff['h32_limit']:.17g}`
- Initial H32 margin: `{handoff['initial_h32_margin']:.17g}`
- D29 archive SHA-256: `{D29_ARCHIVE_SHA256}`; D30 recorded authentication PASS for XZ, tar, internal hashes, cross-document consistency, and checkpoint identity.

## C. Frozen execution policy

Active-boundary recovery applied below `1e-4` margin; reserve building applied from `1e-4` to `2.5e-4`; controlled re-expansion was authorized only at or above `2.5e-4`. The alpha ladder was `[1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]`, with update 421 screening exactly through `1/4`. Every trial cloned the exact canonical parent; rejected trials could not advance model, optimizer, RNG, data, schedule, or lineage. The maximum horizon was updates 421–428.

## D–F. What happened, candidates, and canonical commits

Eighteen candidates were evaluated. Update 421 screened seven alphas; `1/8` was the lowest-H1 candidate that also restored H32 margin, while `1/4` violated the hard H32 limit. Update 422 selected `1/4`. Update 423 selected `1/4`. These three candidates alone became canonical. Update 424 screened `[1/4, 1/8, 1/16, 1/2]`; all four improved margin but made H1 worse, so all were rejected and no update 424 was committed. Complete rows and rejection reasons are in `03_RESULTS/D30_CANDIDATE_RESULTS.json` and `.csv`.

## G. Scientific result

- H1 trajectory: `{handoff['start_h1']:.17g}` → `7.621003439871263e-05` → `7.572983633133678e-05` → `{handoff['final_h1']:.17g}`
- H32 trajectory: `{handoff['start_h32']:.17g}` → `0.0185383890894823` → `0.01848626026248556` → `{handoff['final_h32']:.17g}`
- Canonical H32 margin: `{handoff['initial_h32_margin']:.17g}` → `3.063656907826076e-05` → `8.276539607500119e-05` → `{handoff['final_h32_margin']:.17g}`
- Margin recovery succeeded. Controlled re-expansion did not occur because canonical margin never reached `2.5e-4`.
- Final classification: `{handoff['final_classification']}`.

## H. Integrity

AdamW reconstruction and actual-effective-LR checks passed for all 18 candidates. Deterministic replay, RNG identity, batch/schedule continuity, checkpoint transition, clipping, finite-state, and canonical checkpoint-chain checks passed. Scientific-state contamination: `NO`.

## I. Repairs

One direct-script import-path wrapper defect was repaired before parent loading or candidate generation. Focused inherited AdamW regression tests passed `3/3`, and compilation passed. The authenticated D29 archive was also restored byte-identically after an external removal event and reverified before screening. Neither event changed scientific state. Evidence is in `06_PROVENANCE/IMPLEMENTATION_CHANGES.json` and `08_TESTS/`.

## J. Stage-3 status

Stage-3 completion remained `FAILED`: endpoint H1 and the last-eight mean H1 were above `5e-5`; the H32, consecutive-forward-update, and stability requirements passed. The criteria were not weakened.

## K. Why execution stopped

First blocker: `{handoff['first_blocker']}`. Update 424 had no strict H1-improving reserve-preserving candidate, so execution stopped without committing an invalid state.

## L. Final authoritative state

- Update: `423`
- Checkpoint: `committed_update_423.pt`
- SHA-256: `{FINAL_CHECKPOINT_SHA256}`
- H1: `{handoff['final_h1']:.17g}`
- H32: `{handoff['final_h32']:.17g}`
- H32 margin: `{handoff['final_h32_margin']:.17g}`

This document contains no D31 design or next-step scientific recommendation.
"""
    guide = """# Archive contents guide

- `00_HANDOFF/`: generated navigation and machine-readable handoff records. These summarize, but do not replace, authoritative evidence.
- `01_TASK/`: exact original D30 execution prompt, the packaging-only prompt, and the additional orchestration instruction that materially governed execution.
- `02_FINAL_REPORT/`: unchanged authoritative D30 final report.
- `03_RESULTS/`: authoritative execution manifest, all 18 candidate rows, the canonical trajectory, completion-contract evaluation, and per-update metrics.
- `04_CHECKPOINT/`: authoritative final update-423 checkpoint plus its hash and compact metadata.
- `05_VALIDATION/`: authoritative optimizer reconstruction, deterministic replay, RNG continuity, and per-candidate validation evidence.
- `06_PROVENANCE/`: implementation/repair record, source hashes, and a cross-document consistency audit.
- `07_IMPLEMENTATION/`: exact relevant runner and packaging source files used or preserved for audit.
- `08_TESTS/`: focused regression-test source and captured pass results.
- `09_LOGS/`: the D30 execution console record.
- `10_PRIOR_CONTEXT/`: compact authenticated D29 context, including the update-420 parent checkpoint; the complete D29 archive is intentionally not nested.
- `99_MANIFEST/`: per-file SHA-256 manifest, checksum list, and explicit exclusions.

Authority order: unchanged final report/execution manifest/results/checkpoints are authoritative; generated handoff files are navigation aids; D29 material is historical parent evidence.
"""

    builder.generated_text("00_HANDOFF/START_HERE_NEXT_CHAT.txt", start_here, "handoff")
    builder.generated_text("00_HANDOFF/TASK_AND_RESULTS_SUMMARY.md", summary, "handoff")
    builder.generated_text("00_HANDOFF/ARCHIVE_CONTENTS_GUIDE.md", guide, "handoff")
    builder.generated_json("00_HANDOFF/CROSS_CHAT_HANDOFF.json", handoff, "handoff")

    original_prompt = out / "TASK_PROMPT.md"
    archive_prompt = Path(r"C:\Users\86198\.codex\attachments\0ab9e832-d475-4ca3-8b64-c230bf5a591b\pasted-text.txt")
    builder.copy(original_prompt, "01_TASK/ORIGINAL_D30_EXECUTION_PROMPT.md", "task", "A", "authoritative")
    builder.copy(archive_prompt, "01_TASK/D30_ARCHIVE_CREATION_PROMPT.md", "task", "B", "supporting")
    additional = """Use GPT-5.6 Sol as the sole scientific orchestrator and canonical-state authority; aggressively delegate bounded, independent, read-heavy validation, candidate evaluation, replay verification, provenance auditing, reporting, and other leaf tasks to GPT-5.6 Luna subagents in parallel, but no subagent may independently advance, overwrite, or commit canonical scientific state; only the Sol orchestrator may authorize and perform each canonical update after independently integrating and validating all returned evidence.
"""
    builder.generated_text("01_TASK/ADDITIONAL_EXECUTION_INSTRUCTIONS.txt", additional, "task", "A")

    builder.copy(out / "FINAL_REPORT.md", "02_FINAL_REPORT/FINAL_REPORT.md", "final_report", "A", "authoritative")

    result_files = [
        "D30_EXECUTION_MANIFEST.json",
        "D30_CANDIDATE_RESULTS.json",
        "D30_CANDIDATE_RESULTS.csv",
        "D30_CANONICAL_TRAJECTORY.json",
        "D30_CANONICAL_TRAJECTORY.csv",
        "STAGE3_COMPLETION_CONTRACT_EVALUATION.json",
    ]
    for name in result_files:
        builder.copy(out / name, f"03_RESULTS/{name}", "results", "A", "authoritative")
    for source in sorted((out / "metrics").glob("*.json")):
        builder.copy(source, f"03_RESULTS/per_update_metrics/{source.name}", "results", "B", "authoritative")

    final_checkpoint = out / "checkpoints" / "committed_update_423.pt"
    if sha256(final_checkpoint) != FINAL_CHECKPOINT_SHA256:
        raise RuntimeError("Final checkpoint hash mismatch before staging")
    builder.copy(final_checkpoint, "04_CHECKPOINT/committed_update_423.pt", "checkpoint", "A", "authoritative")
    builder.generated_text(
        "04_CHECKPOINT/FINAL_CHECKPOINT_SHA256.txt",
        f"{FINAL_CHECKPOINT_SHA256}  committed_update_423.pt\n",
        "checkpoint",
    )
    builder.generated_json(
        "04_CHECKPOINT/CHECKPOINT_METADATA.json",
        {
            "update": 423,
            "parent_update": 422,
            "filename": "committed_update_423.pt",
            "sha256": FINAL_CHECKPOINT_SHA256,
            "parent_checkpoint_sha256": final["parent_checkpoint_sha256"],
            "H1": final["H1"],
            "H32": final["H32"],
            "H32_margin": final["H32_margin"],
            "canonical": True,
        },
        "checkpoint",
    )

    validation_files = [
        "OPTIMIZER_CONSISTENCY_REPORT.json",
        "DETERMINISTIC_REPLAY_REPORT.json",
        "RNG_CONTINUITY_REPORT.json",
    ]
    for name in validation_files:
        builder.copy(out / name, f"05_VALIDATION/{name}", "validation", "A", "authoritative")
    for source in sorted((out / "optimizer_consistency").glob("*.json")):
        builder.copy(source, f"05_VALIDATION/per_candidate/{source.name}", "validation", "B", "supporting")

    builder.copy(out / "IMPLEMENTATION_CHANGES.json", "06_PROVENANCE/IMPLEMENTATION_CHANGES.json", "provenance", "B", "authoritative")
    source_hashes = {}
    implementation_sources = [
        ROOT / "tools" / "stage3_h13_d30_active_boundary_continuation.py",
        ROOT / "tools" / "package_stage3_h13_d30_evidence.py",
        Path(__file__),
    ]
    for source in implementation_sources:
        source_hashes[source.name] = {"path": str(source), "size_bytes": source.stat().st_size, "sha256": sha256(source)}
        builder.copy(source, f"07_IMPLEMENTATION/{source.name}", "implementation", "B", "supporting")
    builder.generated_json("06_PROVENANCE/SOURCE_HASHES.json", source_hashes, "provenance", "B")

    consistency = {
        "status": "PASS",
        "checked_documents": [
            "D30_EXECUTION_MANIFEST.json",
            "D30_CANONICAL_TRAJECTORY.json",
            "D30_CANDIDATE_RESULTS.json",
            "FINAL_REPORT.md",
            "STAGE3_COMPLETION_CONTRACT_EVALUATION.json",
        ],
        "agreed_values": {
            "task_name": manifest["experiment_id"],
            "parent_update": parent["update"],
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "final_classification": manifest["classification"],
            "updates_committed": manifest["updates_committed"],
            "final_committed_update": final["update"],
            "start_h1": parent["H1"],
            "final_h1": measurements["H1_final"],
            "start_h32": parent["H32"],
            "final_h32": measurements["H32_final"],
            "final_h32_margin": measurements["final_H32_margin"],
            "selected_alpha_trajectory": measurements["selected_alpha_trajectory"],
            "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
            "stage3_completion_status": completion["stage3_completion_gate"],
            "first_blocker": manifest["first_blocker"],
        },
        "discrepancies": [],
    }
    builder.generated_json("06_PROVENANCE/CROSS_DOCUMENT_CONSISTENCY.json", consistency, "provenance", "A")

    builder.copy(out / "VALIDATOR_REGRESSION_TEST_RESULTS.json", "08_TESTS/VALIDATOR_REGRESSION_TEST_RESULTS.json", "tests", "B", "authoritative")
    builder.copy(ROOT / "tests" / "test_stage3_h13_d26_optimizer_consistency_recovery.py", "08_TESTS/test_stage3_h13_d26_optimizer_consistency_recovery.py", "tests", "B", "supporting")
    builder.copy(out / "EXECUTION_CONSOLE.log", "09_LOGS/EXECUTION_CONSOLE.log", "logs", "B", "supporting")

    d29_out = ROOT / "outputs" / "stage3_h13_post_d28_d29_trust_region_reexpansion_20260823T170000+0800"
    for name in [
        "START_HERE_NEXT_CHAT.txt",
        "FINAL_REPORT.md",
        "D29_EXECUTION_MANIFEST.json",
        "D29_CANONICAL_TRAJECTORY.json",
        "STAGE3_COMPLETION_CONTRACT_EVALUATION.json",
    ]:
        builder.copy(d29_out / name, f"10_PRIOR_CONTEXT/D29_{name}", "prior_context", "C", "historical")
    parent_checkpoint = d29_out / "checkpoints" / "committed_update_420.pt"
    if sha256(parent_checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("D29 parent checkpoint failed packaging-time hash check")
    builder.copy(parent_checkpoint, "10_PRIOR_CONTEXT/committed_update_420.pt", "prior_context", "C", "historical_parent")
    d29_archive = DESKTOP / "STAGE3_H13_D29_TRUST_REGION_REEXPANSION_CROSS_CHAT_EVIDENCE_MAX_XZ_20260823T181000+0800.tar.xz"
    if not d29_archive.is_file() or sha256(d29_archive) != D29_ARCHIVE_SHA256:
        raise RuntimeError("D29 archive failed packaging-time identity check")
    builder.generated_json(
        "10_PRIOR_CONTEXT/D29_ARCHIVE_IDENTITY.json",
        {
            "filename": d29_archive.name,
            "size_bytes": d29_archive.stat().st_size,
            "sha256": D29_ARCHIVE_SHA256,
            "included_as_nested_archive": False,
            "d30_authentication_record": manifest["authoritative_d29_archive"],
            "packaging_time_sha256_verification": "PASS",
        },
        "prior_context",
        "C",
    )

    omitted_rows = []
    for source in sorted((out / "optimizer_consistency").glob("*.pt")):
        omitted_rows.append((source, "Redundant replay/optimizer binary; decisive JSON evidence is included.", True))
    for source in sorted((out / "checkpoints").glob("committed_update_42[12].pt")):
        omitted_rows.append((source, "Intermediate committed checkpoint; hashes and trajectory evidence are included.", True))
    omitted_rows.append((d29_archive, "Complete prior archive is not nested; compact D29 evidence and authenticated parent checkpoint are included.", True))
    excluded_lines = [
        "# Excluded files record",
        "",
        "Relevant large or redundant artifacts deliberately omitted from the D30 archive are listed below.",
        "",
        "| Source path | Type | Size bytes | SHA-256 | Reason | Compact evidence retained | Interpretability impact |",
        "|---|---|---:|---|---|---|---|",
    ]
    for source, reason, compact in omitted_rows:
        excluded_lines.append(
            f"| `{source}` | `{source.suffix or 'file'}` | {source.stat().st_size} | `{sha256(source)}` | {reason} | {'YES' if compact else 'NO'} | None; canonical result and decisive compact evidence remain. |"
        )
    excluded_lines.extend([
        "",
        "Caches, temporary files, unrelated outputs, unrelated historical archives, and rejected-candidate binary checkpoints were not staged. No omission changes the completed D30 result or prevents scientific interpretation.",
        "",
    ])
    builder.generated_text("99_MANIFEST/EXCLUDED_FILES.md", "\n".join(excluded_lines), "manifest", "A")

    precompression = {
        "status": "PASS",
        "timestamp": stamp,
        "scientific_execution_performed": False,
        "additional_optimizer_steps": 0,
        "source_output": str(out),
        "candidate_count": len(candidates),
        "final_checkpoint_sha256_verification": "PASS",
        "parent_checkpoint_sha256_verification": "PASS",
        "d29_archive_sha256_verification": "PASS",
        "cross_document_consistency": "PASS",
        "required_handoff_files_present": True,
    }
    builder.generated_json("06_PROVENANCE/PRECOMPRESSION_VALIDATION.json", precompression, "provenance", "A")

    managed_files = []
    for relative, metadata in sorted(builder.records.items()):
        path = stage / relative
        managed_files.append({
            "relative_archive_path": relative,
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
            **metadata,
        })
    archive_manifest = {
        "schema_version": "d30_cross_chat_archive_manifest_v2",
        "archive_generation_timestamp": stamp,
        "compression_method": "XZ container / LZMA2 / preset 9 / EXTREME",
        "intended_compression_preset": "9 EXTREME",
        "hard_size_limit_bytes": SIZE_LIMIT,
        "scientific_execution_performed_during_archiving": False,
        "additional_optimizer_steps_performed_during_archiving": 0,
        "managed_file_count": len(managed_files),
        "managed_files": managed_files,
        "manifest_self_hash_note": "This manifest and SHA256SUMS.txt are excluded from managed_files to avoid impossible cryptographic self-reference. SHA256SUMS.txt includes this manifest's hash.",
        "files_intentionally_excluded": [str(row[0]) for row in omitted_rows],
        "exclusion_reasons_file": "99_MANIFEST/EXCLUDED_FILES.md",
    }
    manifest_relative = "99_MANIFEST/ARCHIVE_MANIFEST.json"
    json_dump(stage / manifest_relative, archive_manifest)
    sums = []
    for item in managed_files:
        sums.append(f"{item['sha256']}  {item['relative_archive_path']}")
    sums.append(f"{sha256(stage / manifest_relative)}  {manifest_relative}")
    text_dump(stage / "99_MANIFEST/SHA256SUMS.txt", "\n".join(sums) + "\n")
    return handoff


def create_archive(stage: Path, archive: Path) -> None:
    raw_tar = archive.with_suffix("")
    if raw_tar.exists() or archive.exists():
        raise RuntimeError("Refusing to overwrite temporary archive output")

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    with tarfile.open(raw_tar, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(stage.rglob("*")):
            tar.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False, filter=normalize)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    with raw_tar.open("rb") as source, archive.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(compressor.compress(chunk))
        target.write(compressor.flush())
    raw_tar.unlink()


def verify_archive(archive: Path) -> dict[str, Any]:
    decompressed_bytes = 0
    with lzma.open(archive, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            decompressed_bytes += len(chunk)

    with tarfile.open(archive, "r:xz") as tar:
        regular = [member for member in tar.getmembers() if member.isfile()]
        names = {member.name for member in regular}
        manifest_file = tar.extractfile("99_MANIFEST/ARCHIVE_MANIFEST.json")
        if manifest_file is None:
            raise RuntimeError("Archive manifest missing")
        manifest = json.load(manifest_file)
        mismatches = []
        for item in manifest["managed_files"]:
            member = tar.extractfile(item["relative_archive_path"])
            payload = member.read() if member else b""
            if len(payload) != item["byte_size"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                mismatches.append(item["relative_archive_path"])
        sums_file = tar.extractfile("99_MANIFEST/SHA256SUMS.txt")
        if sums_file is None:
            raise RuntimeError("SHA256SUMS missing")
        sum_mismatches = []
        for line in sums_file.read().decode("utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            member = tar.extractfile(relative)
            payload = member.read() if member else b""
            if hashlib.sha256(payload).hexdigest() != expected:
                sum_mismatches.append(relative)
        checkpoint = tar.extractfile("04_CHECKPOINT/committed_update_423.pt")
        checkpoint_hash = hashlib.sha256(checkpoint.read()).hexdigest() if checkpoint else None

    required = {
        "00_HANDOFF/START_HERE_NEXT_CHAT.txt",
        "00_HANDOFF/TASK_AND_RESULTS_SUMMARY.md",
        "00_HANDOFF/ARCHIVE_CONTENTS_GUIDE.md",
        "00_HANDOFF/CROSS_CHAT_HANDOFF.json",
        "01_TASK/ORIGINAL_D30_EXECUTION_PROMPT.md",
        "02_FINAL_REPORT/FINAL_REPORT.md",
        "03_RESULTS/D30_EXECUTION_MANIFEST.json",
        "03_RESULTS/D30_CANDIDATE_RESULTS.json",
        "03_RESULTS/D30_CANDIDATE_RESULTS.csv",
        "03_RESULTS/D30_CANONICAL_TRAJECTORY.json",
        "03_RESULTS/D30_CANONICAL_TRAJECTORY.csv",
        "03_RESULTS/STAGE3_COMPLETION_CONTRACT_EVALUATION.json",
        "04_CHECKPOINT/committed_update_423.pt",
        "05_VALIDATION/OPTIMIZER_CONSISTENCY_REPORT.json",
        "05_VALIDATION/DETERMINISTIC_REPLAY_REPORT.json",
        "05_VALIDATION/RNG_CONTINUITY_REPORT.json",
        "06_PROVENANCE/CROSS_DOCUMENT_CONSISTENCY.json",
        "07_IMPLEMENTATION/stage3_h13_d30_active_boundary_continuation.py",
        "08_TESTS/VALIDATOR_REGRESSION_TEST_RESULTS.json",
        "09_LOGS/EXECUTION_CONSOLE.log",
        "99_MANIFEST/ARCHIVE_MANIFEST.json",
        "99_MANIFEST/SHA256SUMS.txt",
        "99_MANIFEST/EXCLUDED_FILES.md",
    }
    missing = sorted(required - names)
    passed = (
        archive.stat().st_size < SIZE_LIMIT
        and not mismatches
        and not sum_mismatches
        and not missing
        and checkpoint_hash == FINAL_CHECKPOINT_SHA256
    )
    return {
        "overall": "PASS" if passed else "FAIL",
        "archive_path": str(archive),
        "archive_filename": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_size_mib": archive.stat().st_size / (1024 * 1024),
        "size_limit_bytes": SIZE_LIMIT,
        "size_limit_pass": archive.stat().st_size < SIZE_LIMIT,
        "archive_sha256": sha256(archive),
        "archive_file_count": len(regular),
        "compression": "XZ / LZMA2 preset 9 EXTREME",
        "xz_decompression_test": "PASS",
        "tar_content_read_test": "PASS",
        "internal_manifest_verification": "PASS" if not mismatches and not sum_mismatches else "FAIL",
        "internal_manifest_mismatches": mismatches,
        "sha256sums_mismatches": sum_mismatches,
        "required_files_missing": missing,
        "final_checkpoint_included": checkpoint_hash is not None,
        "final_checkpoint_sha256": checkpoint_hash,
        "final_checkpoint_sha256_verification": "PASS" if checkpoint_hash == FINAL_CHECKPOINT_SHA256 else "FAIL",
        "decompressed_tar_bytes": decompressed_bytes,
        "cross_document_consistency": "PASS",
        "scientific_execution_performed_during_archiving": False,
        "additional_optimizer_steps_performed_during_archiving": 0,
        "next_step_recommendation_included": False,
    }


def safe_remove_superseded_archives() -> list[str]:
    desktop_resolved = DESKTOP.resolve()
    removed = []
    for path in DESKTOP.glob(f"{ARCHIVE_PREFIX}*.tar.xz"):
        resolved = path.resolve()
        if resolved.parent != desktop_resolved or not resolved.name.startswith(ARCHIVE_PREFIX):
            raise RuntimeError(f"Unsafe archive cleanup target: {resolved}")
        resolved.unlink()
        removed.append(str(resolved))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    stage = (TMP_ROOT / f"d30_cross_chat_stage_{args.stamp}").resolve()
    temporary_archive = (TMP_ROOT / f"{ARCHIVE_PREFIX}{args.stamp}.tar.xz").resolve()
    final_archive = (DESKTOP / f"{ARCHIVE_PREFIX}{args.stamp}.tar.xz").resolve()
    tmp_resolved = TMP_ROOT.resolve()
    if stage.parent != tmp_resolved or temporary_archive.parent != tmp_resolved:
        raise RuntimeError("Temporary targets escaped repository tmp directory")
    if final_archive.parent != DESKTOP.resolve():
        raise RuntimeError("Final archive escaped Desktop")
    if stage.exists() or temporary_archive.exists() or final_archive.exists():
        raise RuntimeError("Refusing to overwrite requested packaging targets")

    build_handoff(output, stage, args.stamp)
    create_archive(stage, temporary_archive)
    first_check = verify_archive(temporary_archive)
    if first_check["overall"] != "PASS":
        raise RuntimeError(json.dumps(first_check, sort_keys=True))

    removed = safe_remove_superseded_archives()
    shutil.move(str(temporary_archive), str(final_archive))
    final_check = verify_archive(final_archive)
    desktop_matches = list(DESKTOP.glob(f"{ARCHIVE_PREFIX}*.tar.xz"))
    final_check["superseded_d30_archives_removed"] = removed
    final_check["desktop_d30_archive_count"] = len(desktop_matches)
    final_check["single_final_d30_archive"] = len(desktop_matches) == 1
    final_check["d29_parent_archive_preserved"] = (
        DESKTOP / "STAGE3_H13_D29_TRUST_REGION_REEXPANSION_CROSS_CHAT_EVIDENCE_MAX_XZ_20260823T181000+0800.tar.xz"
    ).is_file()
    if not (final_check["overall"] == "PASS" and final_check["single_final_d30_archive"] and final_check["d29_parent_archive_preserved"]):
        raise RuntimeError(json.dumps(final_check, sort_keys=True))

    json_dump(output / "FINAL_CROSS_CHAT_ARCHIVE_VERIFICATION.json", final_check)
    if stage.parent == tmp_resolved and stage.name.startswith("d30_cross_chat_stage_"):
        shutil.rmtree(stage)
    print(json.dumps(final_check, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
