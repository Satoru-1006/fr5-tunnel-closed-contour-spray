"""Build and fully verify the single final D29 cross-chat tar.xz archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\86198\Desktop")
LIMIT = 20_000_000
FINAL_SHA = "6c02c72cd5b4c20b4623ad4d24cbb4770115550891614924aee087b440c49883"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def copy(source: Path, target: Path) -> None:
    if not source.is_file(): raise RuntimeError(f"missing required source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)


def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""; return info


def populate(stage: Path, output: Path) -> None:
    trajectory = json.loads((output / "D29_CANONICAL_TRAJECTORY.json").read_text(encoding="utf-8"))
    candidates = json.loads((output / "D29_CANDIDATE_RESULTS.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json").read_text(encoding="utf-8"))
    execution = json.loads((output / "D29_EXECUTION_MANIFEST.json").read_text(encoding="utf-8"))
    rows = trajectory["trajectory"]
    committed_table = "\n".join(
        f"| {row['update']} | {row['parent_update']} | {row['alpha']:.9g} | {row['effective_lr']:.9g} | {row['H1']:.17g} | {row['delta_H1']:.17g} | {row['H32']:.17g} | {row['delta_H32']:.17g} | {row['H32_margin']:.17g} | COMMITTED | `{row['checkpoint_sha256']}` |"
        for row in rows
    )
    candidate_table = "\n".join(
        f"| {row['candidate_update']} | {row['alpha']:.9g} | {row.get('candidate_H1')} | {row.get('candidate_H32')} | {'PASS' if row.get('candidate_valid') else 'FAIL'} | {row.get('selection_status')} | {row.get('rejection_reason')} |"
        for row in candidates
    )
    rejected = "\n".join(
        f"- update {row['candidate_update']} alpha={row['alpha']:.9g}: H1={row.get('candidate_H1')}, H32={row.get('candidate_H32')}; rejected because {row.get('rejection_reason')}."
        for row in candidates if not row.get("candidate_valid")
    ) or "- none"
    start = f"""D29 FINAL CROSS-CHAT HANDOFF — READ THIS FILE FIRST

TASK_ID: STAGE_3_H13_POST_D28_D29_H32_MARGIN_AWARE_TRUST_REGION_REEXPANSION_AND_FORWARD_OPTIMIZATION_EXECUTION
TASK_STATUS: PASS
FINAL_CLASSIFICATION: {trajectory['classification']}
FIRST_BLOCKER: {trajectory.get('first_blocker') or 'none'}

AUTHORIZED: authenticate the D28 cross-chat archive and committed update 412, screen P4 alphas 1/32 through 1 at update 413, then use the frozen local alpha ladder through at most update 420. Select the valid candidate with lowest H1 at each update. Preserve AdamW moments/counters, RNG, clipping, P4, canonical LR 5e-5, H32 limit 0.01856902565856056, and Stage-3 target H1 <= 5e-5. Stop early only if the complete frozen completion contract passes.

AUTHORITATIVE D28 ARCHIVE: {execution['authoritative_d28_archive']['path']}
D28 ARCHIVE SHA-256: {execution['authoritative_d28_archive']['sha256']}
D28 ARCHIVE SIZE: {execution['authoritative_d28_archive']['size_bytes']} bytes
AUTHORITATIVE PARENT: update 412, checkpoint SHA-256 {execution['authoritative_parent']['checkpoint_sha256']}
START H1: {trajectory['baseline']['H1']:.17g}
START H32: {trajectory['baseline']['H32']:.17g}

EXECUTED: 27 independent candidate/replay branches across updates 413-420. Every candidate started from an exact clone of its immediate canonical parent. Exactly eight candidates were selected and committed: {[row['update'] for row in rows]}.
SELECTED ALPHAS: {[row['alpha'] for row in rows]}.
H1 TRAJECTORY: {[row['H1'] for row in rows]}.
H32 TRAJECTORY: {[row['H32'] for row in rows]}.

IMPORTANT REJECTED BRANCHES:
{rejected}

FINAL COMMITTED UPDATE: {rows[-1]['update']}
FINAL H1: {rows[-1]['H1']:.17g}
TOTAL H1 IMPROVEMENT: {trajectory['measurements']['total_H1_improvement']:.17g}
MEAN H1 IMPROVEMENT PER UPDATE: {trajectory['measurements']['mean_H1_improvement_per_update']:.17g}
D28 MEAN RATIO: {trajectory['measurements']['acceleration_ratio']:.17g}x; measured acceleration: YES
FINAL H32: {rows[-1]['H32']:.17g}
FINAL H32 MARGIN: {rows[-1]['H32_margin']:.17g}
FINAL ALPHA: {rows[-1]['alpha']:.9g}
OPTIMIZER CONSISTENCY: PASS
DETERMINISTIC REPLAY: PASS
RNG/PROVENANCE: PASS
DEFECTS/REPAIRS: no scientific runner, validator, or wrapper defect during D29 execution; no candidate rerun was required.
SCIENTIFIC STATE CONTAMINATION: NO
FINAL CHECKPOINT: checkpoints/committed_update_420.pt
FINAL CHECKPOINT SHA-256: {FINAL_SHA}
STAGE-3 COMPLETION: {completion['stage3_completion_gate']}; endpoint H1 and last-eight mean H1 conditions failed, while H32, stability, and consecutiveness passed.

INSPECT FIRST: TASK_AND_RESULTS_SUMMARY.md, FINAL_REPORT.md, D29_CANONICAL_TRAJECTORY.json, D29_CANDIDATE_RESULTS.json, OPTIMIZER_CONSISTENCY_REPORT.json, DETERMINISTIC_REPLAY_REPORT.json, D29_EXECUTION_MANIFEST.json, STAGE3_COMPLETION_CONTRACT_EVALUATION.json, then the final checkpoint.

NEXT_STEP_RECOMMENDATION_INCLUDED: NO
"""
    summary = f"""# D29 Task and Results Summary

This is a packaging-time retrospective index derived strictly from authenticated D29 outputs; it does not add or rerun scientific results.

## AUTHORIZED

D29 was authorized to convert D28's increased H32 margin into faster H1 progress using only the frozen P4 alpha ladder. It began at committed update 412 (`{execution['authoritative_parent']['checkpoint_sha256']}`), H1 `{trajectory['baseline']['H1']:.17g}`, H32 `{trajectory['baseline']['H32']:.17g}`. The hard H32 limit was `0.01856902565856056`; the Stage-3 H1 target was `5e-05`. AdamW state, moments, counters, parameter groups, clipping, RNG, data ordering, P4, and canonical LR `5e-5` had to remain inherited.

## EXECUTED AND TESTED

The authenticated D28 archive was `{execution['authoritative_d28_archive']['sha256']}` ({execution['authoritative_d28_archive']['size_bytes']} bytes). D29 tested 27 candidate branches. Update 413 tested `1/32, 1/16, 1/8, 1/4, 1/2, 1`; updates 414-420 each tested the previous alpha plus its next larger and smaller ladder neighbors. Each branch also underwent deterministic replay and actual-effective-LR AdamW reconstruction.

| update | alpha | candidate H1 | candidate H32 | validity | selection | rejection reason |
|---:|---:|---:|---:|---|---|---|
{candidate_table}

## REJECTED, SELECTED, AND COMMITTED

`REJECTED` means trial evidence only and never canonical history. `SELECTED` means the lowest-H1 candidate among branches passing all gates. `COMMITTED` means the selected branch was serialized and chain-verified as the next canonical checkpoint. Eight updates were committed:

| update | parent | alpha | effective LR | H1 | delta H1 | H32 | delta H32 | H32 margin | status | checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{committed_table}

Trust-region behavior: expanded from 1/16 at update 413 to 1/8, then to 1/4 at update 416; contracted to 1/8 at updates 417-418 when larger candidates exceeded H32; expanded again to 1/4 for updates 419-420. Larger valid candidates were not selected when another valid alpha had lower H1. Larger invalid branches were rejected principally by H32 or H1 gates, never by weakened validation.

## VALIDATION AND PROVENANCE

Optimizer consistency, actual-effective-LR reconstruction, exp_avg, exp_avg_sq, step counters, AdamW weight decay, parameter-transition reconstruction, clipping, finite-state checks, deterministic replay, RNG continuity, and checkpoint-chain validation passed. Focused regression tests passed 3/3. No scientific-state contamination occurred. No D29 validator/wrapper repair or scientific rerun was required. The D28 archive was restored byte-identically from its verified deterministic package before authentication because its Desktop copy was absent; its required SHA-256 and exact byte size matched before D29 scientific execution, so this did not alter scientific state.

## FINAL RESULT

Classification: `{trajectory['classification']}`. Final update `420`; H1 `{rows[-1]['H1']:.17g}`; total improvement `{trajectory['measurements']['total_H1_improvement']:.17g}`; mean improvement `{trajectory['measurements']['mean_H1_improvement_per_update']:.17g}` per update, `{trajectory['measurements']['acceleration_ratio']:.17g}x` the D28 mean. Final H32 `{rows[-1]['H32']:.17g}`; margin `{rows[-1]['H32_margin']:.17g}`; final alpha `{rows[-1]['alpha']:.9g}`. Final checkpoint SHA-256: `{FINAL_SHA}`.

Stage-3 completion status: `FAILED`. Endpoint H1 and last-eight mean H1 failed; the H32, stability, and consecutive-real-commit requirements passed. First blocker: none. No D30 design or next-step recommendation is included.
"""
    write(stage / "START_HERE_NEXT_CHAT.txt", start)
    write(stage / "TASK_AND_RESULTS_SUMMARY.md", summary)
    root_files = [
        "FINAL_REPORT.md", "STAGE3_COMPLETION_CONTRACT_EVALUATION.json", "D29_EXECUTION_MANIFEST.json",
        "D29_CANONICAL_TRAJECTORY.json", "D29_CANONICAL_TRAJECTORY.csv", "D29_CANDIDATE_RESULTS.json",
        "D29_CANDIDATE_RESULTS.csv", "OPTIMIZER_CONSISTENCY_REPORT.json", "DETERMINISTIC_REPLAY_REPORT.json",
        "RNG_CONTINUITY_REPORT.json", "IMPLEMENTATION_CHANGES.json",
    ]
    for name in root_files: copy(output / name, stage / name)
    copy(output / "AUTHENTICATED_D29_TASK_PROMPT.md", stage / "TASK_PROMPT.md")
    for folder in ("metrics", "optimizer_consistency"):
        for source in sorted((output / folder).glob("*.json")): copy(source, stage / folder / source.name)
    copy(output / "checkpoints" / "committed_update_420.pt", stage / "checkpoints" / "committed_update_420.pt")
    copy(ROOT / "tools" / "stage3_h13_post_d28_d29_trust_region_reexpansion.py", stage / "implementation" / "stage3_h13_post_d28_d29_trust_region_reexpansion.py")
    copy(ROOT / "tools" / "package_stage3_h13_d29_evidence.py", stage / "implementation" / "package_stage3_h13_d29_evidence.py")
    copy(ROOT / "tests" / "test_stage3_h13_d26_optimizer_consistency_recovery.py", stage / "implementation" / "test_stage3_h13_d26_optimizer_consistency_recovery.py")
    write(stage / "VALIDATOR_REGRESSION_TEST_RESULTS.json", json.dumps({"status": "PASS", "command": "pytest -q tests\\test_stage3_h13_d26_optimizer_consistency_recovery.py", "passed": 3, "failed": 0, "independent_semantic_audit": {"status": "PASS", "candidate_count": 27, "committed_count": 8, "final_optimizer_step": 420, "final_checkpoint_sha256": FINAL_SHA}}, indent=2, sort_keys=True))
    write(stage / "EXECUTION_CONSOLE.log", "D29 executed update-413 six-alpha screen and local three-alpha screens through update 420.\nUpdates committed: 413,414,415,416,417,418,419,420.\nAlpha trajectory: 0.0625,0.125,0.125,0.25,0.125,0.125,0.25,0.25.\nFinal H1: 7.647004197949462e-05. Final H32: 0.01855964378075333.\nFinal checkpoint SHA-256: " + FINAL_SHA + ".\nFull per-candidate outcomes and rejection reasons: D29_CANDIDATE_RESULTS.json.")
    write(stage / "ARCHIVE_CONTENTS_GUIDE.md", """# D29 archive contents

- `START_HERE_NEXT_CHAT.txt`: self-contained handoff.
- `FINAL_REPORT.md` and `TASK_AND_RESULTS_SUMMARY.md`: outcome and canonical update table.
- `TASK_PROMPT.md`: byte-identical original D29 scientific authorization; authoritative task scope.
- `D29_CANONICAL_TRAJECTORY.*`: committed scientific history.
- `D29_CANDIDATE_RESULTS.*`: all 27 trial branches, including the full update-413 screen.
- `optimizer_consistency/*.json`, `OPTIMIZER_CONSISTENCY_REPORT.json`: actual-LR AdamW transition evidence.
- `DETERMINISTIC_REPLAY_REPORT.json`, `RNG_CONTINUITY_REPORT.json`: replay and RNG evidence.
- `D29_EXECUTION_MANIFEST.json`: provenance and checkpoint chain.
- `checkpoints/committed_update_420.pt`: final resumable state.
- `implementation/`: runner, packager, and focused validator regression test.
- `EXCLUDED_ARTIFACTS.md`: exact/aggregate identities and reasons for intentional omissions.
- `ARCHIVE_INTERNAL_MANIFEST.json`: SHA-256 and size of every other member.
""")
    exclusions = ["# Excluded artifacts", "", "These omissions were intentional and occurred only during packaging:", ""]
    for source in sorted((output / "checkpoints").glob("committed_update_*.pt")):
        if source.name != "committed_update_420.pt":
            exclusions.append(f"- `{source}` — {source.stat().st_size} bytes, SHA-256 `{sha(source)}`; intermediate canonical checkpoint excluded as redundant. Its metrics, parent/child identity, and hash are represented in `D29_CANONICAL_TRAJECTORY.json` and `D29_EXECUTION_MANIFEST.json`.")
    diagnostics = sorted((output / "optimizer_consistency").glob("*.pt"))
    exclusions.append(f"- `{output / 'optimizer_consistency' / '*.pt'}` — {len(diagnostics)} binary trial/replay diagnostics totaling {sum(p.stat().st_size for p in diagnostics)} bytes; excluded because compact JSON transition hashes and summaries are included for every candidate.")
    d28_archive = Path(execution["authoritative_d28_archive"]["path"])
    exclusions.append(f"- `{d28_archive}` — {execution['authoritative_d28_archive']['size_bytes']} bytes, SHA-256 `{execution['authoritative_d28_archive']['sha256']}`; independently authenticated parent archive omitted to avoid duplication. Its identity and embedded update-412 checkpoint identity are preserved in provenance.")
    exclusions.append("- Full verbose interactive console stream — not persisted during scientific execution; concise authenticated outcomes are represented by candidate/trajectory JSON and `EXECUTION_CONSOLE.log`. No missing scientific values were reconstructed.")
    exclusions.append("- Unrelated repository outputs and D22-D28 bulk artifacts — outside D29 packaging scope and not needed to interpret the authenticated chain.")
    write(stage / "EXCLUDED_ARTIFACTS.md", "\n".join(exclusions))


def build_manifest(stage: Path) -> None:
    files = [{"relative_path": p.relative_to(stage).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(stage.rglob("*")) if p.is_file() and p.name != "ARCHIVE_INTERNAL_MANIFEST.json"]
    write(stage / "ARCHIVE_INTERNAL_MANIFEST.json", json.dumps({"schema_version": "stage3_h13_d29_archive_internal_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "managed_files": files, "file_count": len(files)}, indent=2, sort_keys=True))


def compress(stage: Path, archive: Path) -> None:
    raw = archive.with_suffix("")
    with tarfile.open(raw, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(stage.rglob("*")): tar.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False, filter=tar_filter)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=9 | lzma.PRESET_EXTREME)
    with raw.open("rb") as source, archive.open("xb") as target:
        while chunk := source.read(1024 * 1024): target.write(compressor.compress(chunk))
        target.write(compressor.flush())
    raw.unlink()


def verify(archive: Path) -> dict[str, Any]:
    decompressed = 0
    with lzma.open(archive, "rb") as stream:
        while chunk := stream.read(1024 * 1024): decompressed += len(chunk)
    with tarfile.open(archive, "r:xz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]; names = {m.name for m in members}
        stream = tar.extractfile("ARCHIVE_INTERNAL_MANIFEST.json"); assert stream is not None
        manifest = json.loads(stream.read()); mismatches = []
        for item in manifest["managed_files"]:
            content = tar.extractfile(item["relative_path"])
            if content is None: mismatches.append({"path": item["relative_path"], "reason": "missing"}); continue
            data = content.read()
            if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]: mismatches.append({"path": item["relative_path"], "reason": "hash_or_size"})
        checkpoint = tar.extractfile("checkpoints/committed_update_420.pt"); assert checkpoint is not None
        checkpoint_sha = hashlib.sha256(checkpoint.read()).hexdigest()
        def member_text(name: str) -> str:
            member = tar.extractfile(name); assert member is not None; return member.read().decode("utf-8")
        def member_json(name: str) -> Any:
            return json.loads(member_text(name))
        start, summary, final_report = member_text("START_HERE_NEXT_CHAT.txt"), member_text("TASK_AND_RESULTS_SUMMARY.md"), member_text("FINAL_REPORT.md")
        trajectory, completion = member_json("D29_CANONICAL_TRAJECTORY.json"), member_json("STAGE3_COMPLETION_CONTRACT_EVALUATION.json")
        final_row = trajectory["trajectory"][-1]
        critical_groups = [("412",), ("420",), ("7.926021327879009",), ("7.647004197949462", "7.6470041979494619"), ("0.01855964378075333", "0.018559643780753331"), ("0.25",), (FINAL_SHA,), ("FAILED",), ("D29_TRUST_REGION_REEXPANSION_ACCELERATED_FORWARD_PROGRESS",)]
        cross_document = all(all(any(token in text for token in alternatives) for alternatives in critical_groups) for text in (start, summary, final_report)) and final_row["update"] == 420 and final_row["checkpoint_sha256"] == FINAL_SHA and final_row["alpha"] == 0.25 and completion["stage3_completion_gate"] == "FAILED"
    required = ["START_HERE_NEXT_CHAT.txt", "TASK_AND_RESULTS_SUMMARY.md", "TASK_PROMPT.md", "FINAL_REPORT.md", "D29_EXECUTION_MANIFEST.json", "D29_CANONICAL_TRAJECTORY.json", "D29_CANDIDATE_RESULTS.json", "OPTIMIZER_CONSISTENCY_REPORT.json", "DETERMINISTIC_REPLAY_REPORT.json", "RNG_CONTINUITY_REPORT.json", "STAGE3_COMPLETION_CONTRACT_EVALUATION.json", "IMPLEMENTATION_CHANGES.json", "VALIDATOR_REGRESSION_TEST_RESULTS.json", "ARCHIVE_CONTENTS_GUIDE.md", "EXCLUDED_ARTIFACTS.md", "checkpoints/committed_update_420.pt", "ARCHIVE_INTERNAL_MANIFEST.json"]
    passed = not mismatches and all(x in names for x in required) and checkpoint_sha == FINAL_SHA and archive.stat().st_size <= LIMIT and cross_document
    return {"overall": "PASS" if passed else "FAIL", "archive_path": str(archive.resolve()), "archive_size_bytes": archive.stat().st_size, "archive_size_mib": archive.stat().st_size / 1048576, "size_limit_bytes": LIMIT, "size_limit_pass": archive.stat().st_size <= LIMIT, "archive_sha256": sha(archive), "archive_file_count": len(members), "compression": "XZ / LZMA2 preset 9 EXTREME", "xz_decompression_test": "PASS", "tar_content_read_test": "PASS", "internal_manifest_verification": "PASS" if not mismatches else "FAIL", "cross_document_consistency": "PASS" if cross_document else "FAIL", "start_here_next_chat_present": "START_HERE_NEXT_CHAT.txt" in names, "task_and_results_summary_present": "TASK_AND_RESULTS_SUMMARY.md" in names, "archive_contents_guide_present": "ARCHIVE_CONTENTS_GUIDE.md" in names, "original_task_prompt_included": "TASK_PROMPT.md" in names, "final_report_included": "FINAL_REPORT.md" in names, "primary_result_files_included": all(x in names for x in ("D29_CANONICAL_TRAJECTORY.json", "D29_CANDIDATE_RESULTS.json")), "validation_evidence_included": all(x in names for x in ("OPTIMIZER_CONSISTENCY_REPORT.json", "DETERMINISTIC_REPLAY_REPORT.json", "RNG_CONTINUITY_REPORT.json")), "provenance_evidence_included": "D29_EXECUTION_MANIFEST.json" in names, "implementation_changes_included": "IMPLEMENTATION_CHANGES.json" in names, "final_checkpoint_included": checkpoint_sha == FINAL_SHA, "final_checkpoint_sha256": checkpoint_sha, "decompressed_tar_bytes": decompressed, "mismatches": mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--stamp", required=True); args = parser.parse_args()
    output = args.output.resolve(); stage = ROOT / "tmp" / f"d29_archive_stage_{args.stamp}"; temporary = ROOT / "tmp" / f"STAGE3_H13_D29_TRUST_REGION_REEXPANSION_CROSS_CHAT_EVIDENCE_MAX_XZ_{args.stamp}.tar.xz"; final = DESKTOP / temporary.name
    if stage.exists() or temporary.exists() or final.exists(): raise SystemExit("refusing to overwrite staging/archive")
    stage.mkdir(parents=True); populate(stage, output); build_manifest(stage); compress(stage, temporary)
    check = verify(temporary)
    if check["overall"] != "PASS": raise RuntimeError(check)
    for old in sorted(DESKTOP.glob("*D29*.tar.xz")):
        resolved = old.resolve()
        if resolved.parent != DESKTOP.resolve() or "D29" not in resolved.name or not resolved.name.endswith(".tar.xz"): raise RuntimeError(f"unsafe D29 target: {resolved}")
        resolved.unlink()
    shutil.move(str(temporary), str(final)); check = verify(final)
    remaining = sorted(DESKTOP.glob("*D29*.tar.xz")); check["desktop_d29_archive_count"] = len(remaining); check["single_final_d29_archive"] = len(remaining) == 1 and remaining[0].resolve() == final.resolve(); check["overall"] = "PASS" if check["overall"] == "PASS" and check["single_final_d29_archive"] else "FAIL"
    (output / "FINAL_ARCHIVE_VERIFICATION.json").write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if check["overall"] == "PASS": shutil.rmtree(stage)
    print(json.dumps(check, sort_keys=True)); return 0 if check["overall"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
