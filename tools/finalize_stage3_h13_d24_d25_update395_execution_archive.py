"""Build and verify the single post-execution D25 update-395 evidence archive.

This is an evidence-only utility. It does not import the training runners and
does not execute training, backward passes, optimizer steps, or candidate
branches.
"""

from __future__ import annotations

import csv
import hashlib
import json
import lzma
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\robotfucker")
DESKTOP = Path(r"C:\Users\86198\Desktop")
EXEC = ROOT / "outputs" / "stage3_h13_post_d24_d25_h32_feasible_realized_adamw_trust_region_forward_continuation_execution_20260822T003000+0800_reporting_recovery"
DESIGN = ROOT / "outputs" / "stage3_h13_post_d24_d25_h32_feasible_realized_adamw_trust_region_forward_continuation_design_review_20260821T235900+0800"
D24 = ROOT / "outputs" / "stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution_20260821T225000+0800_retry"
D23 = ROOT / "outputs" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint_20260821T190000+0800"
D22 = ROOT / "outputs" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_execution_evidence_20260821T151902+0800"
PARENT = D24 / "checkpoints" / "committed_update_394.pt"
D23_CHECKPOINT = D23 / "checkpoints" / "C5_update_392.pt"
D24_PREVIOUS = D24 / "checkpoints" / "committed_update_393.pt"
AUTH = ROOT / "outputs" / "D25_EXECUTION_AUTHORIZED_UPDATE_395_20260822T000000+0800.txt"
BUNDLE = ROOT / "outputs" / "stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry" / "post_update_384_branch_bundle.pt"
SCHEDULE_ID = ROOT / "outputs" / "stage3_h13_post_d4_canonical_batch_robustness_20260816T101500Z" / "canonical_schedule_identity.json"
SCHEDULE_STORAGE = ROOT / "outputs" / "stage3_h13_post_d4_canonical_batch_robustness_20260816T101500Z" / "canonical_schedule_manifest.json.gz"
CLASSIFICATION_CONTRACT = ROOT / "outputs" / "stage3_h13_post_d13_d14_gru_h32_temporal_conflict_accumulation_causal_intervention_design_review_20260819T183000+0800" / "classification_contract.json"
PROMPTS = {
    "D25_design_review_prompt.txt": Path(r"C:\Users\86198\.codex\attachments\1d7bb90c-192f-4242-bd16-54342a88182e\pasted-text.txt"),
    "D25_execution_authorization_prompt.txt": Path(r"C:\Users\86198\.codex\attachments\290ba71a-eb43-4f45-97d4-f76b2846968a\pasted-text.txt"),
    "D25_post_execution_archive_requirement.txt": Path(r"C:\Users\86198\.codex\attachments\2c7f0989-ee63-41a8-b435-5bcb76576c26\pasted-text.txt"),
    "D24_execution_prompt.txt": Path(r"C:\Users\86198\.codex\attachments\66c30217-9496-45da-9428-28c3e6ce69fe\pasted-text.txt"),
    "D23_execution_prompt.txt": Path(r"C:\Users\86198\.codex\attachments\5eb8e7eb-97b0-47fb-81bf-8de98a0dc3fa\pasted-text.txt"),
    "D22_design_review_prompt.txt": Path(r"C:\Users\86198\.codex\attachments\b7bfe176-0082-41a8-b37e-56f5c58e193c\pasted-text.txt"),
    "D22_execution_prompt.txt": Path(r"C:\Users\86198\.codex\attachments\648ef2d8-b5a6-4dd2-84a0-8c8e0d0c0261\pasted-text.txt"),
}
MAX_BYTES = 20 * 1024 * 1024
TOP = "STAGE3_H13_D24_D25_UPDATE395_EXECUTION_EVIDENCE"
STAMP = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
ARCHIVE_NAME = f"STAGE3_H13_D24_D25_UPDATE395_EXECUTION_EVIDENCE_MAX_XZ_{STAMP}.tar.xz"
FINAL = DESKTOP / ARCHIVE_NAME
REPORT = ROOT / "outputs" / f"D25_UPDATE395_ARCHIVE_FINALIZATION_REPORT_{STAMP}.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(stage: Path, source: Path, relative: str, records: dict, priority: str, rationale: str) -> None:
    source = source.resolve()
    require(source.is_file(), f"MISSING_SOURCE:{source}")
    require(relative not in records, f"ARCHIVE_PATH_COLLISION:{relative}")
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    records[relative] = {
        "relative_path": relative,
        "size_bytes": target.stat().st_size,
        "sha256": sha256(target),
        "priority": priority,
        "source_origin": str(source),
        "rationale": rationale,
    }


def copy_top(stage: Path, source_root: Path, target_root: str, records: dict, priority: str, rationale: str) -> None:
    for source in sorted(source_root.iterdir()):
        if source.is_file():
            copy_file(stage, source, f"{target_root}/{source.name}", records, priority, rationale)


def managed_paths(stage: Path) -> list[Path]:
    excluded = {
        "00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json",
        "00_ARCHIVE_METADATA/SHA256_ALL_FILES.json",
    }
    return [path for path in sorted(stage.rglob("*")) if path.is_file() and path.relative_to(stage).as_posix() not in excluded]


def write_summaries(stage: Path, records: dict, rows: list[dict], parent_hash: str) -> None:
    parent_h1 = 8.366622366361216e-05
    parent_h32 = 0.018494000982624222
    h32_limit = 0.01856902565856056
    summary_root = stage / "09_SUMMARIES"
    summary_root.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    rejection_lines = []
    for row in rows:
        rejection = row["rejection_reason"] or ""
        audit_rows.append({
            "alpha": float(row["alpha"]),
            "effective_lr": row["effective_lr"],
            "candidate_H1": float(row["candidate_H1"]),
            "delta_H1": float(row["delta_H1"]),
            "candidate_H32": float(row["candidate_H32"]),
            "delta_H32": float(row["delta_H32"]),
            "H32_margin_after": float(row["H32_margin_after"]),
            "finite_state_pass": row["finite_state_pass"] == "True",
            "H32_preservation_pass": row["H32_preservation_pass"] == "True",
            "H1_improvement_pass": row["H1_improvement_pass"] == "True",
            "optimizer_consistency": "FAIL" if "OPTIMIZER_STATE_CONSISTENCY_FAILURE" in rejection else "PASS",
            "candidate_valid": row["candidate_valid"] == "True",
            "selected": row["selected"] == "True",
            "rejection_reason": rejection,
            "model_state_hash": row["model_checkpoint_sha256"],
            "optimizer_state_hash": row["optimizer_state_sha256_or_equivalent"],
            "batch_identity": row["batch_identity"],
            "parameter_delta_l2": float(row["parameter_delta_l2"]),
            "parameter_delta_linf": float(row["parameter_delta_linf"]),
            "adamw_update_l2": float(row["adamw_update_l2"]),
            "gradient_l2": float(row["gradient_l2"]),
            "postclip_gradient_l2": float(row["postclip_gradient_l2"]),
        })
        rejection_lines.append(f"- α={row['alpha']}: `{rejection}`; H1={row['candidate_H1']}; H32={row['candidate_H32']}; H32 margin={row['H32_margin_after']}.")
    audit = {
        "schema_version": "stage3_h13_d25_candidate_gate_audit_from_emitted_evidence_v1",
        "scientific_execution_performed_by_this_audit": False,
        "optimizer_consistency_basis": "frozen runner rejection_reason; the frozen CANDIDATE_FIELDS schema does not persist auxiliary optimizer_step_identity_pass or adamw_decomposition_status columns",
        "rows": audit_rows,
    }
    audit_path = stage / "01_D25_EXECUTION" / "D25_CANDIDATE_GATE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    records["01_D25_EXECUTION/D25_CANDIDATE_GATE_AUDIT.json"] = {"relative_path": "01_D25_EXECUTION/D25_CANDIDATE_GATE_AUDIT.json", "size_bytes": audit_path.stat().st_size, "sha256": sha256(audit_path), "priority": "A", "source_origin": "generated from emitted D25 CSV/JSON only", "rationale": "explicit optimizer-consistency and gate audit without rerunning candidates"}

    result = f"""# D25 update-395 result summary

## Actual result

- Classification: `H32_FEASIBLE_BUT_NO_H1_IMPROVING_ALPHA`
- Status: `BLOCKED`
- First integrity blocker: `OPTIMIZER_STATE_CONSISTENCY_FAILURE`, first observed at α=0.5.
- No update-395 checkpoint or committed trajectory was created.

## Starting state

- Start update: `394`
- Target update: `395`
- Parent checkpoint SHA-256: `{parent_hash}`
- Parent H1: `{parent_h1}`
- Parent H32: `{parent_h32}`
- H32 limit: `{h32_limit}`
- Parent H32 margin: `{h32_limit - parent_h32}`

## Candidate search

- Exact evaluated sequence: `[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]`.
- Proposal: `P4`; fallback: `NONE`; canonical parent LR: `[5e-05]`.
- Effective LRs were canonical parent LR multiplied by alpha.
- Rejection records:
{chr(10).join(rejection_lines)}
- H32-feasible candidates: α=`1/32`, `1/64`, `1/128`, `1/256`.
- These H32-feasible candidates all numerically improved H1, but all failed optimizer-state consistency and therefore had `candidate_valid=false`.
- First fully acceptable alpha: none.
- Adjacent-smaller candidate under the first-valid rule: not applicable, because no valid candidate occurred.
- Selected alpha: none.

## Strongest observed tradeoff

The first H32-feasible candidate, α=`1/32`, reached H1=`8.352321361774779e-05`, H32=`0.01853788498214386`, and positive H32 margin=`3.1140676416699375e-05`, but remained uncommittable because of optimizer-state consistency failure. The lowest-H1 full step (α=`1`) violated H32 with H32=`0.02222581590520285`.

## Interpretation

The executed values support a step-size/trust-region contribution to the D24 blocker: reducing the realized step from α=`1` to α=`1/32` moved H32 back inside the frozen limit while H1 continued to decrease. However, D25 did not validate a commit-eligible P4 continuation. The remaining failure is an optimizer-state consistency conflict in the H32-feasible realized-AdamW branches; this is not evidence that P4 direction alone is impossible, nor is it sufficient to claim that step-size reduction solved the scientific problem.

## Next decision

Do not execute update 396+. A new separately authorized scientific/engineering decision is required to resolve the optimizer-consistency blocker before any future continuation; no new experiment was executed during archive finalization.
"""
    result_path = summary_root / "D25_UPDATE395_RESULT_SUMMARY.md"
    result_path.write_text(result, encoding="utf-8", newline="\n")
    records["09_SUMMARIES/D25_UPDATE395_RESULT_SUMMARY.md"] = {"relative_path": "09_SUMMARIES/D25_UPDATE395_RESULT_SUMMARY.md", "size_bytes": result_path.stat().st_size, "sha256": sha256(result_path), "priority": "A", "source_origin": "generated from finalized D25 result", "rationale": "actual result, interpretation, and next decision"}

    progress = f"""# Stage 3 H13 D22 → D25 progress summary

## D22 — causal localization

D22 replayed the frozen factorial causal-localization design through updates 385–392 only. It identified the inherited H32 structure as `H32_BY_B1_B2_C2_C1_CONDITIONAL_CHILD_STRUCTURE`, with authenticated D21 imports and no adaptive selection or threshold changes. Its key causal reports, factorial summaries, manifests, and one branch-record stream are archived under `06_D22_PROVENANCE/`.

## D23 — bounded forward optimization sprint

D23 tested the frozen candidate families from the authenticated D22 state, preserved H32 for tested candidates, but did not pass the frozen Stage-3 completion gate. Its best tested forward state was the C5 lineage at update 392, with checkpoint hash `226cd512c2f16e415c67b3c0e4045a89841a008f7ec4a09e9249e3c2309efdc1`, which became the D24 starting state.

## D24 — realized-AdamW forward continuation

D24 continued from update 392 using the authorized P4 realized-AdamW proposal and committed updates 393 and 394. Update 394 became the authenticated D25 parent with checkpoint hash `{parent_hash}`, H1=`{parent_h1}`, and H32=`{parent_h32}`. The full-step update-395 attempt exceeded the frozen H32 limit, so D24 classified the continuation as `D24_EXECUTION_BLOCKED` and motivated a trust-region continuation.

## D25 — trust-region continuation

D25 evaluated P4 from the authenticated update-394 parent with the frozen alpha ladder. Four smaller alphas preserved H32 and all numerically improved H1, but none passed optimizer-state consistency; no canonical update-395 state was committed. The current canonical update remains 394.

## Current state and unresolved blocker

- Current canonical update: `394`
- Current H1: `{parent_h1}`
- Current H32: `{parent_h32}`
- H32 limit: `{h32_limit}`
- D25 status: `BLOCKED`
- Unresolved blocker: `OPTIMIZER_STATE_CONSISTENCY_FAILURE` on H32-feasible realized-AdamW branches.
- Update 396+ has not been executed.

## What Stage 3 still requires

A new scientific decision must resolve the optimizer-consistency failure under a separately frozen and authorized protocol before any continuation. D25 success was not obtained and Stage-3 completion is not declared.
"""
    progress_path = summary_root / "STAGE3_H13_D22_TO_D25_PROGRESS_SUMMARY.md"
    progress_path.write_text(progress, encoding="utf-8", newline="\n")
    records["09_SUMMARIES/STAGE3_H13_D22_TO_D25_PROGRESS_SUMMARY.md"] = {"relative_path": "09_SUMMARIES/STAGE3_H13_D22_TO_D25_PROGRESS_SUMMARY.md", "size_bytes": progress_path.stat().st_size, "sha256": sha256(progress_path), "priority": "A", "source_origin": "generated from archived D22/D23/D24/D25 reports", "rationale": "chronological scientific lineage and current decision"}

    guide = """# Archive content guide

This archive is the single post-execution D25 update-395 evidence package. Paths are relative to the archive root.

## D25 execution

- `01_D25_EXECUTION/`: candidate CSV, blocked execution manifest/summary, authorization identity, failure report, SHA-256 manifest, and generated gate audit.
- `01_D25_EXECUTION/D25_EXECUTION_AUTHORIZED.txt`: exact authorization file used by the frozen execution gate.
- `09_SUMMARIES/D25_UPDATE395_RESULT_SUMMARY.md`: actual scientific result, interpretation, and next decision.

## D25 design

- `02_D25_DESIGN/`: complete frozen design-review artifacts, including alpha ladder, candidate contract, static validation, P4 reconstruction, optimizer validation, design manifest, final report, and design SHA manifest.
- `07_IMPLEMENTATION_AND_PROMPTS/implementation/D25_trust_region_execution.py`: exact frozen execution entrypoint.
- `07_IMPLEMENTATION_AND_PROMPTS/implementation/D25_trust_region_design_review.py`: frozen design-review implementation.

## Update-394 parent and D24 provenance

- `03_UPDATE394_PARENT/committed_update_394.pt`: authenticated parent binary.
- `04_D24_PROVENANCE/`: D24 manifests, proposal table, trajectory, final report, blocker summary, SHA manifest, and preceding update-393 checkpoint.

## D23 and D22 provenance

- `05_D23_PROVENANCE/`: D23 reports, candidate/trajectory tables, manifests, and C5 update-392 checkpoint.
- `06_D22_PROVENANCE/`: compact causal-localization reports, factorial/Möbius summaries, classification, conservation, provenance and SHA records. One duplicate branch-record stream is intentionally omitted.

## Runtime, implementation, and prompts

- `07_IMPLEMENTATION_AND_PROMPTS/`: D22–D25 implementation snapshots and preserved prompts.
- `08_RUNTIME_AND_HASH_IDENTITY/`: canonical bundle/schedule inputs, classification contract, runtime helpers, and repository execution constraints.

## Manifests and exclusions

- `00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json`: archive scope and counts.
- `00_ARCHIVE_METADATA/SHA256_ALL_FILES.json`: SHA-256 and byte-size inventory for every managed payload, excluding only the two self-referential metadata manifests named inside it.
- `00_ARCHIVE_METADATA/ARCHIVE_PROVENANCE.json`: critical identities and archive-finalization boundary.
- `00_ARCHIVE_METADATA/EXCLUSIONS.json`: explicit policy exclusions and confirmation that no file was removed for the 20 MiB size limit.

No update-395 checkpoint is included because D25 did not create one. No update 396+ evidence is present.
"""
    guide_path = summary_root / "ARCHIVE_CONTENT_GUIDE.md"
    guide_path.write_text(guide, encoding="utf-8", newline="\n")
    records["09_SUMMARIES/ARCHIVE_CONTENT_GUIDE.md"] = {"relative_path": "09_SUMMARIES/ARCHIVE_CONTENT_GUIDE.md", "size_bytes": guide_path.stat().st_size, "sha256": sha256(guide_path), "priority": "A", "source_origin": "generated from actual archive mapping", "rationale": "independent navigation guide"}


def write_metadata(stage: Path, records: dict, old_related: list[dict], parent_hash: str, d23_hash: str, d24_previous_hash: str, design_hash: str, entry_hash: str, exec_manifest_hash: str, exec_summary_hash: str, auth_hash: str) -> None:
    metadata = stage / "00_ARCHIVE_METADATA"
    metadata.mkdir(parents=True, exist_ok=True)
    provenance = {
        "archive_purpose": "single final Stage-3 H13 D24/D25 update-395 execution evidence archive",
        "archive_filename": ARCHIVE_NAME,
        "creation_timestamp": datetime.now().astimezone().isoformat(),
        "format": "tar.xz",
        "compression": "XZ / LZMA2 preset 9 + EXTREME",
        "size_limit_bytes": MAX_BYTES,
        "experiment": "STAGE_3_H13_POST_D24_D25_H32_FEASIBLE_REALIZED_ADAMW_TRUST_REGION_FORWARD_CONTINUATION_EXECUTION",
        "d25_design_manifest_sha256": design_hash,
        "d25_execution_module_sha256": entry_hash,
        "d25_execution_manifest_sha256": exec_manifest_hash,
        "d25_execution_summary_sha256": exec_summary_hash,
        "authorization_sha256": auth_hash,
        "update_394_checkpoint_sha256": parent_hash,
        "update_393_checkpoint_sha256": d24_previous_hash,
        "d23_c5_update_392_checkpoint_sha256": d23_hash,
        "target_update": 395,
        "current_canonical_update": 394,
        "update_395_checkpoint_created": "NO",
        "update_395_checkpoint_included": "NOT_CREATED",
        "scientific_execution_during_archive_finalization": {
            "training": "NO", "backward_pass": "NO", "optimizer_step": "NO",
            "additional_alpha_trials": "NO", "update_395_rerun": "NO", "update_396_plus": "NO",
        },
        "pre_existing_related_desktop_archive": old_related,
        "manifest_exclusions": ["00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json", "00_ARCHIVE_METADATA/SHA256_ALL_FILES.json"],
    }
    (metadata / "ARCHIVE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    exclusions = {
        "EXCLUDED_FOR_SIZE_LIMIT": "none",
        "POLICY_EXCLUSIONS": [
            {"path": "Desktop/pre-existing/design-evidence-archive", "reason": "pre-existing archive; not created by this task and not nested"},
            {"path": "D23/checkpoints/* except C5_update_392.pt", "reason": "large redundant historical candidate checkpoints; textual provenance retained"},
            {"path": "D22/branch_update_records.jsonl", "reason": "duplicate of branch_update_records.checkpoint.jsonl"},
            {"path": "D25_UPDATE_395_CHECKPOINT.pt", "reason": "not created because D25 was scientifically blocked"},
        ],
    }
    (metadata / "EXCLUSIONS.json").write_text(json.dumps(exclusions, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    paths = managed_paths(stage)
    file_list = [path.relative_to(stage).as_posix() for path in paths] + ["00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json", "00_ARCHIVE_METADATA/SHA256_ALL_FILES.json"]
    (metadata / "ARCHIVE_FILE_LIST.txt").write_text("\n".join(file_list) + "\n", encoding="utf-8", newline="\n")
    paths = managed_paths(stage)
    contents = {
        "archive_purpose": provenance["archive_purpose"],
        "archive_filename": ARCHIVE_NAME,
        "creation_timestamp": provenance["creation_timestamp"],
        "compression": provenance["compression"],
        "size_limit_bytes": MAX_BYTES,
        "managed_payload_file_count": len(paths),
        "managed_payload_uncompressed_bytes": sum(path.stat().st_size for path in paths),
        "d25_priority_A_evidence_included": True,
        "update_394_provenance_included": True,
        "d24_provenance_included": True,
        "d23_provenance_included": True,
        "d22_provenance_included": True,
        "update_395_checkpoint_created": "NO",
        "update_395_checkpoint_included": "NOT_CREATED",
        "excluded_for_size_limit": "none",
        "self_excluded_metadata_files": ["00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json", "00_ARCHIVE_METADATA/SHA256_ALL_FILES.json"],
    }
    (metadata / "ARCHIVE_CONTENTS_MANIFEST.json").write_text(json.dumps(contents, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    paths = managed_paths(stage)
    inventory = {
        "schema_version": "stage3_h13_d25_update395_archive_sha256_all_files_v1",
        "hash_algorithm": "SHA-256",
        "self_excluded": True,
        "excluded_from_hash_inventory": ["00_ARCHIVE_METADATA/ARCHIVE_CONTENTS_MANIFEST.json", "00_ARCHIVE_METADATA/SHA256_ALL_FILES.json"],
        "file_count": len(paths),
        "total_uncompressed_bytes": sum(path.stat().st_size for path in paths),
        "files": [{"relative_path": path.relative_to(stage).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths],
    }
    (metadata / "SHA256_ALL_FILES.json").write_text(json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def make_archive(stage: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with tarfile.open(destination, mode="w:xz", format=tarfile.PAX_FORMAT, preset=9 | lzma.PRESET_EXTREME) as archive:
        for source in sorted(stage.rglob("*")):
            if source.is_file():
                info = archive.gettarinfo(str(source), arcname=f"{TOP}/{source.relative_to(stage).as_posix()}")
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with source.open("rb") as stream:
                    archive.addfile(info, stream)


def verify_archive(path: Path, expected: dict[str, str]) -> dict:
    with lzma.open(path, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    verify_root = Path(tempfile.mkdtemp(prefix="d25_update395_archive_verify_", dir=str(ROOT / "tmp")))
    try:
        with tarfile.open(path, mode="r:xz") as archive:
            members = archive.getmembers()
            for member in members:
                parts = Path(member.name).parts
                require(not Path(member.name).is_absolute() and ".." not in parts, f"UNSAFE_TAR_MEMBER:{member.name}")
            archive.extractall(verify_root)
        root = verify_root / TOP
        inventory = read_json(root / "00_ARCHIVE_METADATA" / "SHA256_ALL_FILES.json")
        failures = []
        for item in inventory["files"]:
            target = root / item["relative_path"]
            if not target.is_file() or target.stat().st_size != int(item["size_bytes"]) or sha256(target) != item["sha256"]:
                failures.append(item["relative_path"])
        require(not failures, f"INTERNAL_MANIFEST_FAILURES:{failures}")
        critical_failures = []
        for relative, expected_hash in expected.items():
            target = root / relative
            if not target.is_file() or sha256(target) != expected_hash:
                critical_failures.append(relative)
        require(not critical_failures, f"CRITICAL_HASH_FAILURES:{critical_failures}")
        require(not (root / "01_D25_EXECUTION" / "D25_UPDATE_395_CHECKPOINT.pt").exists(), "UNEXPECTED_UPDATE_395_CHECKPOINT")
        return {
            "xz_decompression_test": "PASS",
            "tar_content_read_test": "PASS",
            "internal_manifest_verification": "PASS",
            "critical_file_hash_verification": "PASS",
            "managed_payload_file_count": inventory["file_count"],
            "tar_member_count": len(members),
            "tar_file_member_count": sum(1 for member in members if member.isfile()),
        }
    finally:
        shutil.rmtree(verify_root, ignore_errors=True)


def main() -> int:
    required = [EXEC, DESIGN, D24, D23, D22, PARENT, D23_CHECKPOINT, D24_PREVIOUS, AUTH, BUNDLE, SCHEDULE_ID, SCHEDULE_STORAGE, CLASSIFICATION_CONTRACT, *PROMPTS.values()]
    for path in required:
        require(path.exists(), f"MISSING_REQUIRED_SOURCE:{path}")
    design_hash = sha256(DESIGN / "D25_DESIGN_MANIFEST.json")
    entry_hash = sha256(ROOT / "tools" / "stage3_h13_post_d24_d25_trust_region_execution.py")
    parent_hash = sha256(PARENT)
    d23_hash = sha256(D23_CHECKPOINT)
    d24_previous_hash = sha256(D24_PREVIOUS)
    auth_hash = sha256(AUTH)
    exec_manifest_hash = sha256(EXEC / "D25_EXECUTION_MANIFEST.json")
    exec_summary_hash = sha256(EXEC / "D25_EXECUTION_SUMMARY.json")
    require(parent_hash == "f7939ac8d6c9adedbfb2ed6cf90a949e7252ba6926932053617fbe4caac97a47", "UPDATE_394_HASH_MISMATCH")
    require(d23_hash == "226cd512c2f16e415c67b3c0e4045a89841a008f7ec4a09e9249e3c2309efdc1", "D23_CHECKPOINT_HASH_MISMATCH")
    require(design_hash == "caaf7bfc8c2799671dc58148e7ebe9c4d49fc534bf89639662f3e069c6b963b0", "D25_DESIGN_HASH_MISMATCH")
    require(entry_hash == "8a4eab02e2938c59d573c6398bf240f4d0042712133e809a982e4c594ac1061f", "D25_ENTRYPOINT_HASH_MISMATCH")
    require(DESKTOP.is_dir(), f"DESKTOP_NOT_FOUND:{DESKTOP}")
    require(not FINAL.exists(), f"FINAL_ARCHIVE_ALREADY_EXISTS:{FINAL}")
    report_path = REPORT
    require(not report_path.exists(), f"FINALIZATION_REPORT_ALREADY_EXISTS:{report_path}")
    old_related = []
    for path in DESKTOP.iterdir():
        if path.is_file() and path.name.startswith("STAGE3_H13_D24_D25_REALIZED_ADAMW_TRUST_REGION_DESIGN_EVIDENCE_MAX_XZ_"):
            old_related.append({"name": path.name, "path": str(path.resolve()), "sha256": sha256(path), "status": "pre_existing_not_created_by_this_task"})

    stage = Path(tempfile.mkdtemp(prefix="d25_update395_archive_", dir=str(ROOT / "tmp")))
    candidate = ROOT / "tmp" / f"{ARCHIVE_NAME}.tmp"
    records = {}
    try:
        copy_top(stage, EXEC, "01_D25_EXECUTION", records, "A", "D25 execution evidence; no checkpoint was emitted")
        copy_file(stage, AUTH, "01_D25_EXECUTION/D25_EXECUTION_AUTHORIZED.txt", records, "A", "authorization file used by the frozen gate")
        copy_top(stage, DESIGN, "02_D25_DESIGN", records, "A", "complete frozen D25 design-review artifacts")
        copy_file(stage, PARENT, "03_UPDATE394_PARENT/committed_update_394.pt", records, "A", "authoritative update-394 parent")
        copy_top(stage, D24, "04_D24_PROVENANCE/artifacts", records, "A", "D24 direct result and provenance evidence")
        copy_file(stage, D24_PREVIOUS, "04_D24_PROVENANCE/checkpoints/committed_update_393.pt", records, "B", "D24 preceding committed state")
        copy_top(stage, D23, "05_D23_PROVENANCE/artifacts", records, "B", "D23 reports, tables, and manifests")
        copy_file(stage, D23_CHECKPOINT, "05_D23_PROVENANCE/checkpoints/C5_update_392.pt", records, "B", "authenticated D23 checkpoint")
        for source in sorted(D22.iterdir()):
            if source.is_file() and source.name != "branch_update_records.jsonl":
                copy_file(stage, source, f"06_D22_PROVENANCE/{source.name}", records, "C", "D22 causal-localization provenance")
        implementation = {
            "D25_trust_region_execution.py": ROOT / "tools" / "stage3_h13_post_d24_d25_trust_region_execution.py",
            "D25_trust_region_design_review.py": ROOT / "tools" / "stage3_h13_post_d24_d25_trust_region_design_review.py",
            "D24_forward_optimization_execution.py": ROOT / "tools" / "stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution.py",
            "D23_forward_optimization_execution.py": ROOT / "tools" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint.py",
            "D22_execution_replay.py": ROOT / "tools" / "stage3_h13_post_d21_d22_execution_replay.py",
            "D22_design_review.py": ROOT / "tools" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_design_review.py",
            "D17_runtime.py": ROOT / "tools" / "stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay.py",
            "D16_optimizer_pipeline.py": ROOT / "tools" / "stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay.py",
            "AGENTS.md": ROOT / "AGENTS.md",
        }
        for name, source in implementation.items():
            target = "08_RUNTIME_AND_HASH_IDENTITY/AGENTS.md" if name == "AGENTS.md" else f"07_IMPLEMENTATION_AND_PROMPTS/implementation/{name}"
            copy_file(stage, source, target, records, "A" if name.startswith("D25") or name == "AGENTS.md" else "B", "implementation/protocol identity")
        for name, source in PROMPTS.items():
            copy_file(stage, source, f"07_IMPLEMENTATION_AND_PROMPTS/prompts/{name}", records, "A" if name.startswith("D25") else "B", "preserved governing prompt")
        runtime_inputs = {
            "post_update_384_branch_bundle.pt": BUNDLE,
            "canonical_schedule_identity.json": SCHEDULE_ID,
            "canonical_schedule_manifest.json.gz": SCHEDULE_STORAGE,
            "classification_contract.json": CLASSIFICATION_CONTRACT,
        }
        for name, source in runtime_inputs.items():
            copy_file(stage, source, f"08_RUNTIME_AND_HASH_IDENTITY/canonical_inputs/{name}", records, "B", "authenticated canonical runtime input")

        rows = list(csv.DictReader((EXEC / "D25_CANDIDATE_RESULTS.csv").open(encoding="utf-8", newline="")))
        write_summaries(stage, records, rows, parent_hash)
        write_metadata(stage, records, old_related, parent_hash, d23_hash, d24_previous_hash, design_hash, entry_hash, exec_manifest_hash, exec_summary_hash, auth_hash)

        make_archive(stage, candidate)
        require(candidate.stat().st_size <= MAX_BYTES, f"ARCHIVE_SIZE_LIMIT_EXCEEDED:{candidate.stat().st_size}")
        expected = {
            "02_D25_DESIGN/D25_DESIGN_MANIFEST.json": design_hash,
            "07_IMPLEMENTATION_AND_PROMPTS/implementation/D25_trust_region_execution.py": entry_hash,
            "03_UPDATE394_PARENT/committed_update_394.pt": parent_hash,
            "01_D25_EXECUTION/D25_EXECUTION_MANIFEST.json": exec_manifest_hash,
            "01_D25_EXECUTION/D25_EXECUTION_SUMMARY.json": exec_summary_hash,
            "01_D25_EXECUTION/D25_CANDIDATE_RESULTS.csv": sha256(EXEC / "D25_CANDIDATE_RESULTS.csv"),
        }
        verify_archive(candidate, expected)
        shutil.copy2(candidate, FINAL)
        verification = verify_archive(FINAL, expected)
        require(FINAL.stat().st_size <= MAX_BYTES, "FINAL_ARCHIVE_SIZE_LIMIT_EXCEEDED")
        task_archives = [path for path in DESKTOP.iterdir() if path.is_file() and path.name.startswith("STAGE3_H13_D24_D25_UPDATE395_EXECUTION_EVIDENCE_MAX_XZ_") and path.suffix.lower() == ".xz"]
        require(len(task_archives) == 1 and task_archives[0].resolve() == FINAL.resolve(), f"ONLY_ONE_TASK_ARCHIVE_FAILED:{task_archives}")
        other_task_archives = [path for path in DESKTOP.iterdir() if path.is_file() and path.name.startswith("STAGE3_H13_D24_D25_UPDATE395_EXECUTION_EVIDENCE_MAX_XZ_") and path.suffix.lower() != ".xz"]
        require(not other_task_archives, f"ALTERNATE_TASK_ARCHIVE_PRESENT:{other_task_archives}")
        result = {
            "FINAL_EVIDENCE_ARCHIVE": str(FINAL.resolve()),
            "FORMAT": "tar.xz",
            "COMPRESSION": "XZ / LZMA2 preset 9 + EXTREME",
            "ARCHIVE_SIZE_BYTES": FINAL.stat().st_size,
            "ARCHIVE_SIZE_MIB": FINAL.stat().st_size / (1024 * 1024),
            "FILE_COUNT": {"managed_payload_count": verification["managed_payload_file_count"], "tar_member_count": verification["tar_member_count"], "tar_file_member_count": verification["tar_file_member_count"]},
            "ARCHIVE_SHA256": sha256(FINAL),
            "XZ_DECOMPRESSION_TEST": verification["xz_decompression_test"],
            "TAR_CONTENT_READ_TEST": verification["tar_content_read_test"],
            "INTERNAL_MANIFEST_VERIFICATION": verification["internal_manifest_verification"],
            "CRITICAL_FILE_HASH_VERIFICATION": verification["critical_file_hash_verification"],
            "D25_PRIORITY_A_EVIDENCE_INCLUDED": "YES",
            "UPDATE_395_CHECKPOINT_CREATED": "NO",
            "UPDATE_395_CHECKPOINT_INCLUDED": "NOT_CREATED",
            "UPDATE_395_CHECKPOINT_HASH_VERIFICATION": "NOT_APPLICABLE",
            "UPDATE_394_PROVENANCE_INCLUDED": "YES",
            "UPDATE_394_CHECKPOINT_HASH_VERIFICATION": "PASS",
            "D24_PROVENANCE_INCLUDED": "YES",
            "D23_PROVENANCE_INCLUDED": "YES",
            "D22_PROVENANCE_INCLUDED": "YES",
            "RESULT_SUMMARY_INCLUDED": "YES",
            "D22_TO_D25_PROGRESS_SUMMARY_INCLUDED": "YES",
            "EXCLUDED_FOR_SIZE_LIMIT": "none",
            "POLICY_EXCLUSIONS": ["pre-existing design archive not nested", "redundant D23 candidate checkpoints omitted", "duplicate D22 branch record stream omitted", "no D25 checkpoint existed"],
            "ONLY_ONE_FINAL_ARCHIVE_CREATED": "YES",
            "PRE_EXISTING_RELATED_DESKTOP_ARCHIVES_NOT_CREATED_BY_THIS_TASK": old_related,
            "TRAINING_EXECUTED_DURING_ARCHIVE_FINALIZATION": "NO",
            "BACKWARD_PASS_EXECUTED_DURING_ARCHIVE_FINALIZATION": "NO",
            "OPTIMIZER_STEP_EXECUTED_DURING_ARCHIVE_FINALIZATION": "NO",
            "UPDATE_395_RERUN_DURING_ARCHIVE_FINALIZATION": "NO",
            "UPDATE_396_PLUS_EXECUTED": "NO",
            "D25_DESIGN_MANIFEST_SHA256": design_hash,
            "D25_EXECUTION_MODULE_SHA256": entry_hash,
            "D25_EXECUTION_MANIFEST_SHA256": exec_manifest_hash,
            "D25_EXECUTION_SUMMARY_SHA256": exec_summary_hash,
            "AUTHORIZATION_SHA256": auth_hash,
            "AUTHORITATIVE_UPDATE_394_SHA256": parent_hash,
            "FINALIZATION_REPORT_PATH": str(REPORT.resolve()),
        }
        REPORT.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
        return 0
    finally:
        if candidate.exists():
            candidate.unlink()
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
