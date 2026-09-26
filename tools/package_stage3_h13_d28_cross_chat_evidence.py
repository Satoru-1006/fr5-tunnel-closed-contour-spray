"""Create the final D28 cross-chat forensic evidence archive only."""

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
PARENT_SHA = "d59a011c4689aefb75c094a82d934fa1569e822d94288698954577401baf7a9b"
FINAL_SHA = "f877703c07b39fcd2777ad3c393d4f8532f90ced88490f7c8409a1de63e22dae"
D27_ARCHIVE_SHA = "63b4b8e26b92ee26823f7eb88d889fba5aaa1466c0894bb2506a797b8cde1e0b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"Required source missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_result(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trajectory = json.loads((output / "D28_FORWARD_TRAJECTORY.json").read_text(encoding="utf-8"))
    execution = json.loads((output / "D28_EXECUTION_MANIFEST.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json").read_text(encoding="utf-8"))
    rows = trajectory["trajectory"]
    assert [row["update"] for row in rows] == list(range(407, 413))
    assert execution["updates_committed"] == list(range(407, 413))
    assert trajectory["classification"] == execution["classification"] == "D28_FORWARD_PROGRESS_CONFIRMED"
    assert trajectory["measurements"]["H1_final"] == rows[-1]["H1"] == 7.926021327879009e-05
    assert trajectory["measurements"]["H32_final"] == rows[-1]["H32"] == 0.01751783891085687
    assert rows[-1]["checkpoint_sha256"] == FINAL_SHA
    assert completion["stage3_completion_gate"] == "FAILED"
    return trajectory, execution, completion


def start_here(output: Path, trajectory: dict[str, Any], completion: dict[str, Any]) -> str:
    rows = trajectory["trajectory"]
    table = "\n".join(
        f"{row['update']} | {row['H1']:.17g} | {row['H32']:.17g} | {-row['delta_H1']:.17g} | {row['H32_margin_after']:.17g} | COMMITTED | PASS"
        for row in rows
    )
    return f"""D28 FINAL CROSS-CHAT HANDOFF — READ THIS FILE FIRST

A. TASK_IDENTITY
PROJECT_STAGE: STAGE_3_H13
TASK_ID: STAGE_3_H13_POST_D27_D28_FORWARD_CONTINUATION_EXECUTION
TASK_NAME: D28 Forward Continuation Execution
EXECUTION_TIME: 2026-08-23 15:34:26 +08:00 (output timestamp)
OUTPUT_DIRECTORY: {output}
AUTHORITATIVE_PREDECESSOR: D27 forward continuation evidence archive
D27_ARCHIVE_PATH: C:\\Users\\86198\\Desktop\\STAGE3_H13_D27_FORWARD_CONTINUATION_EVIDENCE_MAX_XZ_20260822T094500+0800.tar.xz
D27_ARCHIVE_SHA256: {D27_ARCHIVE_SHA}
STARTING_CHECKPOINT: D27 committed_update_406.pt
STARTING_CHECKPOINT_SHA256: {PARENT_SHA}

B. WHAT_THIS_TASK_WAS
REQUESTED: Continue the authenticated D25 -> D26 -> D27 optimizer-consistent Stage-3 H13 trajectory through exactly updates 407-412, as one parent-child chain. The scientific objective was to measure whether H1 improvement continued, plateaued, became limited by H32, or satisfied the inherited Stage-3 completion contract.

Frozen policy: P4; alpha=1/32; canonical learning rate 5e-5; actual optimizer learning rate 1.5625e-6; inherited AdamW parameters, moments, step counters, clipping, RNG, and determinism settings. Every child had to pass H1 improvement, H32 <= 0.01856902565856056, finite-state, actual-LR AdamW reconstruction, parameter-transition, step-counter, RNG, and deterministic-replay gates before commit.

Prohibited changes included optimizer reset, LR/alpha/policy search, architecture changes, historical D22-D27 mutation, new causal screens, or execution beyond update 412.

C. WHAT_CODEX_ACTUALLY_DID
ACTUALLY EXECUTED: Authenticated the D27 archive identity and repository copy of update 406; located the frozen D24 completion contract; created the narrow D28 runner tools/stage3_h13_post_d27_d28_forward_continuation.py; ran the inherited optimizer-consistency pytest suite; executed the chained updates 407, 408, 409, 410, 411, and 412; ran each candidate twice from the same parent for deterministic replay; reconstructed AdamW transitions using the actual LR; committed all six valid children; verified every checkpoint hash, parent link, payload position, and per-parameter optimizer step; evaluated the eight-update completion window (405-412); generated reports/manifests; and built this archive with XZ/LZMA2 preset 9 EXTREME.

Entry point used:
python tools\\stage3_h13_post_d27_d28_forward_continuation.py --output {output}

Regression command:
pytest -q tests\\test_stage3_h13_d26_optimizer_consistency_recovery.py
Result: 3 passed, 0 failed.

D. FINAL_RESULT
TASK_STATUS: PASS
FINAL_CLASSIFICATION: D28_FORWARD_PROGRESS_CONFIRMED
START_H1: 7.9619566197397247e-05
FINAL_H1: 7.9260213278790093e-05
H1_IMPROVEMENT_FROM_406: 3.5935291860715406e-07
H1_PERCENT_IMPROVEMENT_FROM_406: 0.4513374485314657
START_H32: 0.017801222825239323
FINAL_H32: 0.01751783891085687
H32_CHANGE_FROM_406: -0.00028338391438245333
FINAL_H32_MARGIN: 0.0010511867477036897
OPTIMIZER_CONSISTENCY: PASS
DETERMINISTIC_REPLAY: PASS
REGRESSION_TESTS: PASS (3/3)
PROVENANCE: PASS
STAGE3_COMPLETION_CONTRACT_FOUND: YES
STAGE3_COMPLETION_STATUS: FAILED
UNSATISFIED_CONDITIONS: endpoint H1 <= 5e-5; last-eight mean H1 <= 5e-5
LAST_8_MEAN_H1: {completion['last_8_mean_H1']:.17g}
The H32, stability, and consecutive-real-forward completion conditions passed.

E. TRAJECTORY_OR_RESULT_SUMMARY
update | H1 | H32 | H1 gain | H32 margin | state | validation
{table}

All six H1 gains were positive. H32 decreased at every D28 update, so the feasibility margin increased. No frozen plateau threshold existed; none was invented.

F. PROBLEMS_FOUND
1. A preliminary read-only PowerShell archive-inspection expression had a syntax error. Root cause: invalid inline native-command/status expression. Scientific state affected: NO. Previously committed results affected: NO.
2. The external `xz` executable was unavailable during preliminary inspection. The canonical D27 archive identity still matched exactly; standards-compliant Python lzma/tar handling was used for archive integrity and final packaging. Scientific state affected: NO.
3. The first standalone post-run checkpoint-verifier invocation had PowerShell/Python quoting failure. It failed before evaluation. The unchanged verifier was rerun over standard input and all assertions passed. Scientific state affected: NO.
4. Independent cross-document archive QA found that the first retrospective draft stated the correct metrics and conclusions but omitted the literal canonical classification string. Scientific state affected: NO. The retrospective was corrected and the archive rebuilt before final handoff.

G. REPAIRS_MADE
VALIDATION/REPORTING REPAIR: Corrected the read-only PowerShell command structure; no scientific code or result changed.
VALIDATION/REPORTING REPAIR: Reinvoked the unchanged semantic verifier through standard input; all six chain/hash/step checks passed.
PACKAGING IMPLEMENTATION: Added tools/package_stage3_h13_d28_evidence.py and tools/package_stage3_h13_d28_cross_chat_evidence.py to create normal maximum-compression tar.xz archives without an external xz binary.
HANDOFF QA REPAIR: Added the explicit `D28_FORWARD_PROGRESS_CONFIRMED` classification to TASK_AND_RESULTS_SUMMARY.md after the independent consistency assertion exposed the omission; rebuilt and fully reverified the archive.
SCIENTIFIC EXECUTION CHANGE: NONE. No D28 update required rerun, because no defect contaminated an uncommitted or committed scientific state.

H. SCIENTIFIC_STATE_INTEGRITY
SCIENTIFIC_STATE_CONTAMINATION: NO
The scientific runner completed without a candidate, optimizer, RNG, replay, checkpoint-chain, or finite-state defect. The encountered issues were read-only shell/packaging wrappers outside scientific-state transitions. Every committed update passed independent replay and optimizer-transition checks.

I. AUTHORITATIVE_FINAL_STATE
CANONICAL_FINAL_CHECKPOINT: 07_CHECKPOINTS/committed_update_412.pt
ORIGINAL_PATH: {output / 'checkpoints' / 'committed_update_412.pt'}
FINAL_UPDATE: 412
CHECKPOINT_SHA256: {FINAL_SHA}
PARENT_UPDATE_406_SHA256: {PARENT_SHA}
FINAL_H1: 7.9260213278790093e-05
FINAL_H32: 0.01751783891085687
OPTIMIZER_STATE: authenticated AdamW continuity through step 412; PASS
RNG_AND_DETERMINISM: continuity and deterministic replay PASS
BEST_CHECKPOINT: same as latest committed checkpoint, update 412

J. IMPORTANT_FILE_MAP
START_HERE_NEXT_CHAT.txt — this authoritative cross-chat orientation.
TASK_AND_RESULTS_SUMMARY.md — detailed retrospective of the completed D28 task.
TASK_PROMPT.md — byte-identical original D28 execution authorization.
CODEX_FINAL_RESPONSE.md — the completion response reported after D28.
02_FINAL_REPORTS/FINAL_REPORT.md — authoritative human-readable scientific result.
03_RESULTS/D28_FORWARD_TRAJECTORY.json — canonical machine-readable per-update trajectory.
04_VALIDATION/OPTIMIZER_CONTINUITY_REPORT.json — per-update AdamW state/transition evidence.
04_VALIDATION/DETERMINISTIC_REPLAY_REPORT.json — candidate-versus-replay identity evidence.
04_VALIDATION/VALIDATOR_REGRESSION_TEST_RESULTS.json — pytest and semantic-verifier results.
02_FINAL_REPORTS/STAGE3_COMPLETION_CONTRACT_EVALUATION.json — frozen completion-gate evaluation.
05_PROVENANCE/D28_EXECUTION_MANIFEST.json — authorization, parent, policy, commit, and classification provenance.
05_PROVENANCE/CHECKPOINT_HASH_MANIFEST.json — output file and checkpoint hashes.
06_IMPLEMENTATION/ — D28 runner, reused validators, packagers, tests, and change explanation.
07_CHECKPOINTS/committed_update_412.pt — canonical final resumable scientific state.
08_LOGS/EXECUTION_CONSOLE.log — concise execute/commit console trace.
ARCHIVE_MANIFEST_SHA256.txt and ARCHIVE_INTERNAL_MANIFEST.json — archive member identities.

K. ARCHIVE_STRUCTURE
Root: cross-chat handoff, task prompt, reported response, contents guide, exclusions, and archive manifests.
01_TASK_DEFINITION: authorization identity/support.
02_FINAL_REPORTS: final report, classification, and completion evaluation.
03_RESULTS: trajectory, candidate table, and per-update metric records.
04_VALIDATION: optimizer/replay/regression evidence and per-update transition summaries.
05_PROVENANCE: execution, checkpoint, provenance, D27 parent-handoff, and implementation manifests.
06_IMPLEMENTATION: code and directly relevant regression test.
07_CHECKPOINTS: canonical final checkpoint plus identity record.
08_LOGS: concise execution and validation traces.

L. EVIDENCE_CONFIDENCE
PASS: D27 archive SHA-256 and size authentication; update-406 parent SHA-256; parent model/optimizer/policy/step identity; all six actual-LR AdamW reconstructions; first/second moments; weight decay and parameter-transition residuals; per-parameter steps; RNG continuity; deterministic replay; finite-state and H32 gates; six checkpoint parent/hash links; 3/3 regression tests; semantic checkpoint verifier; internal archive SHA-256 manifest; XZ decompression and TAR readability.

M. WHAT_WAS_NOT_DONE
No update 413 or later was executed. No Stage-3 completion was claimed. No plateau threshold was invented. No D22-D27 artifact or checkpoint was modified. No optimizer, moment, RNG, LR, alpha, P-policy, architecture, threshold, or clipping configuration was changed. No causal-screen, hyperparameter search, D29, Stage 4, or other future experiment was performed. Intermediate checkpoints and large duplicate optimizer diagnostic PT files were excluded from this archive as redundant; their decisive JSON evidence is included.

N. NO_NEXT_STEP_PREDICTION
NEXT_STEP_RECOMMENDATION: NOT INCLUDED BY DESIGN
THIS_ARCHIVE_DOCUMENTS THE COMPLETED TASK ONLY.
"""


def task_summary(output: Path, trajectory: dict[str, Any], completion: dict[str, Any]) -> str:
    rows = trajectory["trajectory"]
    table = "\n".join(f"| {r['update']} | {r['parent_update']} | {r['H1']:.17g} | {r['delta_H1']:.17g} | {r['H32']:.17g} | {r['delta_H32']:.17g} | {r['H32_margin_after']:.17g} |" for r in rows)
    return f"""# D28 Task and Results Summary

## 1. Background entering D28

D27 ended at authenticated committed update 406 with H1 `7.9619566197397247e-05`, H32 `0.017801222825239323`, and checkpoint SHA-256 `{PARENT_SHA}`. D27 had preserved the P4, alpha=1/32 optimizer-consistent regime and repaired a historical validation-only NumPy RNG-array comparison defect. D28 inherited the repaired validator and the full committed AdamW/RNG state.

## 2. Completed task

D28 was an execution-only continuation through exactly updates 407-412. Its purpose was to collect six real forward observations while keeping the optimizer, policy, learning rates, clipping, deterministic execution, H32 feasibility limit, and scientific validation gates frozen. It was not a search or redesign task.

## 3. Starting state and methodology

The D27 archive identity was authenticated as `{D27_ARCHIVE_SHA}` and the local update-406 checkpoint matched `{PARENT_SHA}`. The runner loaded the serialized model, AdamW state and moments, step 406 counters, canonical LR `5e-5`, actual LR `1.5625e-6`, and RNG state. Each update used the preceding committed child as its only parent.

For each update, the runner executed the P4 candidate and then replayed it from the same parent/RNG. Commit required H1 improvement, H32 feasibility, finite values, exact actual-LR identity, AdamW moment/decomposition consistency, step identity, parameter-transition residual consistency, RNG identity, and deterministic replay identity.

## 4. What was actually run

Updates 407, 408, 409, 410, 411, and 412 were each executed, validated, and committed. No update outside that range was run. The inherited optimizer-consistency regression suite passed 3/3. A separate checkpoint verifier loaded all six checkpoints and validated their hashes, parent links, payload positions, and per-parameter optimizer steps.

## 5. Numerical results

Final classification: `D28_FORWARD_PROGRESS_CONFIRMED`.

| Update | Parent | H1 | delta H1 | H32 | delta H32 | H32 margin |
|---:|---:|---:|---:|---:|---:|---:|
{table}

Endpoint H1 was `7.9260213278790093e-05`, an absolute improvement of `3.5935291860715406e-07` (`0.4513374485314657%`) from update 406. Endpoint H32 was `0.01751783891085687`, a decrease of `0.00028338391438245333`; its remaining feasibility margin was `0.0010511867477036897`.

All six H1 gains were positive. Their magnitudes were `1.3914900723816838e-07`, `7.599622597661963e-08`, `3.944506576746129e-08`, `2.9570125428064666e-08`, `4.2106458524013876e-08`, and `3.3086035672826226e-08`. D28 did not label a plateau because the inherited artifacts contained no frozen numerical plateau criterion.

## 6. Completion-contract result

The frozen contract was found in the D24 runner. It requires an endpoint H1 and last-eight mean H1 at or below `5e-5`, all eight H32 values at or below `0.01856902565856056`, stable valid optimizer states, and eight consecutive real-forward updates. For window 405-412, H32, stability, and real-forward continuity passed. Endpoint H1 and the last-eight mean H1 (`{completion['last_8_mean_H1']:.17g}`) failed. Stage 3 therefore was not declared complete.

## 7. Problems, repairs, and anomalies

The scientific runner encountered no scientific-state defect. Three non-scientific wrapper/environment issues were recorded: a PowerShell expression syntax error during preliminary read-only inspection, absence of an external `xz` executable, and shell quoting failure in the first standalone checkpoint-verifier invocation. The inspection command was corrected, the unchanged verifier was rerun via standard input and passed, and standard Python lzma/tar support was used for standards-compliant XZ/LZMA2 packaging. Independent archive QA also caught that the first retrospective draft omitted the literal canonical classification string; the summary was corrected and the archive rebuilt. No update was rerun and no committed state was altered.

## 8. Validation and trust basis

Parent authentication, actual-LR optimizer reconstruction, first/second moments, weight decay, parameter residuals, optimizer steps, clipping instrumentation, finite-state checks, H32 feasibility, RNG identity, deterministic replay, checkpoint-chain validation, 3/3 regression tests, and post-run semantic checkpoint verification all passed. The archive separately verifies every included member against its SHA-256 manifest and passes XZ/TAR integrity checks.

## 9. Authoritative final state

The canonical final state is `07_CHECKPOINTS/committed_update_412.pt`, copied byte-for-byte from `{output / 'checkpoints' / 'committed_update_412.pt'}`. Its SHA-256 is `{FINAL_SHA}`. It is the latest and best D28 committed state, with serialized AdamW and RNG continuity through update 412.

## 10. What D28 established

D28 established that the frozen P4/alpha=1/32 optimizer-consistent regime produced six further strictly H1-improving commits while increasing H32 feasibility margin. It also established that the inherited Stage-3 completion contract was not satisfied at update 412. These are measured results; this retrospective contains no proposal or prediction for a later task.

NEXT_STEP_RECOMMENDATION: NOT INCLUDED BY DESIGN
THIS_ARCHIVE_DOCUMENTS THE COMPLETED TASK ONLY.
"""


def contents_guide() -> str:
    return """# Archive Contents Guide

## Root handoff layer

- `START_HERE_NEXT_CHAT.txt`: primary cross-chat orientation, exact task/result/state summary, evidence map, and integrity statement.
- `TASK_AND_RESULTS_SUMMARY.md`: detailed completed-task retrospective with the update table and validation basis.
- `TASK_PROMPT.md`: byte-identical D28 execution authorization supplied by the user.
- `CODEX_FINAL_RESPONSE.md`: completion response reported after scientific execution.
- `ARCHIVE_MANIFEST_SHA256.txt` / `ARCHIVE_INTERNAL_MANIFEST.json`: size, SHA-256, source, role, and authority for archive files; self-excluded.
- `EXCLUDED_ARTIFACTS.md`: explains deliberate high-bulk exclusions.

## 01_TASK_DEFINITION

- Authorization identity record connecting the copied prompt to its original attachment and SHA-256.

## 02_FINAL_REPORTS

- Authoritative D28 final report, final classification JSON, and frozen Stage-3 completion-contract evaluation.

## 03_RESULTS

- Canonical trajectory JSON/CSV, candidate-results CSV, and six per-update metric/record JSON files.

## 04_VALIDATION

- Optimizer continuity, deterministic replay, regression-test results, and six detailed optimizer-transition JSON summaries. These are the compact decisive validation artifacts; redundant multi-megabyte diagnostic PT copies are omitted.

## 05_PROVENANCE

- D28 execution, checkpoint-hash, provenance, and implementation manifests, plus compact authenticated D27 predecessor handoff files.

## 06_IMPLEMENTATION

- The D28 runner, reused D27/D26 validator runners, archive packagers, directly relevant pytest file, and implementation-change explanations.

## 07_CHECKPOINTS

- Byte-identical canonical final update-412 checkpoint and a human/machine-readable identity record.

## 08_LOGS

- Concise successful execution trace, regression/semantic validation output, and wrapper-repair record.
"""


def codex_response() -> str:
    return f"""# Codex D28 Completion Response — Scientific Result Portion

The scientific completion portion below preserves the reported D28 values and classification. The earlier response also identified the now-superseded compact archive and contained a future-facing action line. Those packaging-identity lines are superseded by this archive, and the future-facing line is intentionally not reproduced because the current archive instruction explicitly prohibits next-step recommendations. No scientific result was rewritten.

TASK_STATUS: PASS
FINAL_CLASSIFICATION: D28_FORWARD_PROGRESS_CONFIRMED
FIRST_BLOCKER: none
D28_EXECUTED: YES
UPDATES_EXECUTED: 407, 408, 409, 410, 411, 412
UPDATES_COMMITTED: 407, 408, 409, 410, 411, 412
START_CHECKPOINT_SHA256: {PARENT_SHA}
FINAL_CHECKPOINT_SHA256: {FINAL_SHA}
START_H1: 7.961956619739725e-05
FINAL_H1: 7.926021327879009e-05
H1_IMPROVEMENT_FROM_406: 3.593529186071541e-07 (0.451337%)
START_H32: 0.017801222825239323
FINAL_H32: 0.01751783891085687
FINAL_H32_MARGIN: 0.0010511867477036897
OPTIMIZER_CONSISTENCY: PASS
DETERMINISTIC_REPLAY: PASS
SCIENTIFIC_STATE_CONTAMINATION: NO
VALIDATOR_OR_WRAPPER_REPAIRS: Two read-only shell syntax/quoting repairs; both reran successfully and did not affect scientific state
STAGE3_COMPLETION_CONTRACT_FOUND: YES
STAGE3_COMPLETION_STATUS: FAILED — endpoint H1 and last-eight mean H1 remain above 5e-05

D28 established six strictly H1-improving, H32-feasible commits. H1 progress continued but at small, variable gains; H32 is not limiting because its margin increased substantially. No frozen plateau threshold exists, so no plateau was invented.
"""


def populate(stage: Path, output: Path, prompt: Path, trajectory: dict[str, Any], execution: dict[str, Any], completion: dict[str, Any]) -> None:
    write_text(stage / "START_HERE_NEXT_CHAT.txt", start_here(output, trajectory, completion))
    write_text(stage / "TASK_AND_RESULTS_SUMMARY.md", task_summary(output, trajectory, completion))
    copy_file(prompt, stage / "TASK_PROMPT.md")
    write_text(stage / "CODEX_FINAL_RESPONSE.md", codex_response())
    write_text(stage / "ARCHIVE_CONTENTS_GUIDE.md", contents_guide())
    write_text(stage / "EXCLUDED_ARTIFACTS.md", """# Excluded Artifacts

- Intermediate committed checkpoints 407-411: redundant for cross-chat handoff; identities remain in the trajectory and checkpoint manifests.
- Twelve optimizer diagnostic/replay `.pt` files: large redundant serialization; complete per-update JSON transition/replay evidence is included.
- The predecessor update-406 binary: its original path, exact SHA-256, model/optimizer/RNG identities, and D27 manifests are included; the final update-412 checkpoint contains the resumable D28 state.
- Verbose repository-wide files, caches, images, and unrelated historical stages: outside this completed task.
- The earlier compact D28 archive: superseded and removed from the Desktop so only this canonical cross-chat archive remains.
""")

    write_json(stage / "01_TASK_DEFINITION" / "AUTHORIZATION_IDENTITY.json", {"original_path": str(prompt.resolve()), "archive_copy": "TASK_PROMPT.md", "sha256": sha256_file(prompt), "size_bytes": prompt.stat().st_size, "authoritative": True})
    packaging_prompt = Path(r"C:\Users\86198\.codex\attachments\9d3f0809-00e1-4e08-bd06-26d755edb5a3\pasted-text.txt")
    copy_file(packaging_prompt, stage / "01_TASK_DEFINITION" / "FINAL_ARCHIVE_CREATION_INSTRUCTIONS.md")
    write_json(stage / "01_TASK_DEFINITION" / "ARCHIVE_INSTRUCTION_IDENTITY.json", {"original_path": str(packaging_prompt.resolve()), "archive_copy": "01_TASK_DEFINITION/FINAL_ARCHIVE_CREATION_INSTRUCTIONS.md", "sha256": sha256_file(packaging_prompt), "size_bytes": packaging_prompt.stat().st_size, "authoritative": True})

    for name in ("FINAL_REPORT.md", "FINAL_CLASSIFICATION.json", "STAGE3_COMPLETION_CONTRACT_EVALUATION.json"):
        copy_file(output / name, stage / "02_FINAL_REPORTS" / name)
    for name in ("D28_FORWARD_TRAJECTORY.json", "D28_FORWARD_TRAJECTORY.csv", "D28_CANDIDATE_RESULTS.csv"):
        copy_file(output / name, stage / "03_RESULTS" / name)
    for source in sorted((output / "metrics").glob("*.json")):
        copy_file(source, stage / "03_RESULTS" / "metrics" / source.name)

    for name in ("OPTIMIZER_CONTINUITY_REPORT.json", "DETERMINISTIC_REPLAY_REPORT.json", "VALIDATOR_REGRESSION_TEST_RESULTS.json"):
        copy_file(output / name, stage / "04_VALIDATION" / name)
    for source in sorted((output / "optimizer_consistency").glob("*.json")):
        copy_file(source, stage / "04_VALIDATION" / "optimizer_consistency" / source.name)

    for name in ("D28_EXECUTION_MANIFEST.json", "CHECKPOINT_HASH_MANIFEST.json", "PROVENANCE_MANIFEST.json", "IMPLEMENTATION_CHANGES.json"):
        copy_file(output / name, stage / "05_PROVENANCE" / name)
    d27 = ROOT / "outputs" / "stage3_h13_post_d26_d27_forward_continuation_20260822T094500+0800"
    for name in ("START_HERE_NEXT_CHAT.txt", "FINAL_REPORT.md", "D27_EXECUTION_MANIFEST.json", "D27_FORWARD_TRAJECTORY.json", "OPTIMIZER_CONTINUITY_REPORT.json", "CHECKPOINT_HASH_MANIFEST.json"):
        copy_file(d27 / name, stage / "05_PROVENANCE" / "D27_PREDECESSOR" / name)

    implementation_sources = (
        ROOT / "tools" / "stage3_h13_post_d27_d28_forward_continuation.py",
        ROOT / "tools" / "stage3_h13_post_d26_d27_forward_continuation.py",
        ROOT / "tools" / "stage3_h13_d26_progress_first_optimizer_consistency_recovery.py",
        ROOT / "tools" / "package_stage3_h13_d28_evidence.py",
        Path(__file__).resolve(),
        ROOT / "tests" / "test_stage3_h13_d26_optimizer_consistency_recovery.py",
    )
    for source in implementation_sources:
        copy_file(source, stage / "06_IMPLEMENTATION" / source.name)
    copy_file(output / "IMPLEMENTATION_CHANGES.json", stage / "06_IMPLEMENTATION" / "IMPLEMENTATION_CHANGES.json")
    write_text(stage / "06_IMPLEMENTATION" / "IMPLEMENTATION_CHANGES.md", """# Implementation Changes

## `tools/stage3_h13_post_d27_d28_forward_continuation.py`

Purpose: minimally extend the authenticated D27 continuation infrastructure to authenticate update 406 and execute updates 407-412. Before: the D27 runner authenticated update 400 and ran 401-406. After: D28 authenticates update 406 and runs 407-412 with the same P4/alpha/LR/AdamW/RNG/replay gates. Scientific policy changed: NO. Regression: inherited optimizer-consistency suite 3/3 PASS; six-checkpoint semantic verifier PASS.

## Archive packagers

Purpose: create standards-compliant XZ/LZMA2 preset 9 EXTREME evidence packages using Python lzma because an external xz executable was unavailable. Scientific algorithm or result changed: NO. Validation: XZ decompression, TAR readability, internal SHA-256, required-root-file, topology, count, and size checks.

## Reused validation code

The D27 and D26 runner/validator sources and the directly relevant pytest file are included unchanged as supporting implementation evidence.
""")

    checkpoint = output / "checkpoints" / "committed_update_412.pt"
    copy_file(checkpoint, stage / "07_CHECKPOINTS" / checkpoint.name)
    write_json(stage / "07_CHECKPOINTS" / "CHECKPOINT_IDENTITY.json", {"filename": checkpoint.name, "originating_path": str(checkpoint.resolve()), "update": 412, "scientific_role": "canonical final/latest/best D28 resumable committed state", "sha256": sha256_file(checkpoint), "size_bytes": checkpoint.stat().st_size, "parent_update": 411, "D28_start_parent_update": 406, "D28_start_parent_sha256": PARENT_SHA, "included_reason": "Canonical final checkpoint fits comfortably below the archive limit."})

    copy_file(output / "EXECUTION_CONSOLE.log", stage / "08_LOGS" / "EXECUTION_CONSOLE.log")
    write_text(stage / "08_LOGS" / "VALIDATION_OUTPUT.txt", """pytest -q tests\\test_stage3_h13_d26_optimizer_consistency_recovery.py
... [100%]
3 passed in 3.04s

Post-run semantic verifier:
status=PASS
updates_checked=407,408,409,410,411,412
checks=checkpoint SHA-256, parent hash link, payload parent/update position, completed optimizer step, all per-parameter optimizer steps, completion-contract logic
""")
    write_text(stage / "08_LOGS" / "WRAPPER_REPAIRS.md", """# Wrapper Repairs

1. Read-only PowerShell inspection expression syntax failed before scientific execution. The command structure was corrected and authentication rerun. Scientific state affected: NO.
2. External `xz` command was unavailable. Standard Python `lzma` with XZ format, CRC64, preset 9 + EXTREME was used. Scientific state affected: NO.
3. First standalone checkpoint-verifier command had shell quoting failure before evaluation. The unchanged verifier was passed through standard input and all assertions passed. Scientific state affected: NO.
4. First cross-chat retrospective draft omitted the literal canonical classification string. Independent consistency QA failed closed, the explicit classification was added, and the archive was rebuilt. Scientific state affected: NO.

No scientific execution repair, checkpoint rewrite, or D28 update rerun occurred.
""")


def role_for(relative: str) -> tuple[str, str]:
    if relative in {"START_HERE_NEXT_CHAT.txt", "TASK_AND_RESULTS_SUMMARY.md", "ARCHIVE_CONTENTS_GUIDE.md"}:
        return "cross-chat handoff", "authoritative summary"
    if relative == "TASK_PROMPT.md" or relative.startswith("01_TASK_DEFINITION/"):
        return "task definition", "authoritative"
    if relative.startswith("02_FINAL_REPORTS/") or relative == "CODEX_FINAL_RESPONSE.md":
        return "final report", "authoritative"
    if relative.startswith("03_RESULTS/"):
        return "scientific result", "authoritative"
    if relative.startswith("04_VALIDATION/"):
        return "validation", "authoritative/supporting"
    if relative.startswith("05_PROVENANCE/"):
        return "provenance", "authoritative/supporting"
    if relative.startswith("06_IMPLEMENTATION/"):
        return "implementation", "supporting"
    if relative.startswith("07_CHECKPOINTS/"):
        return "checkpoint", "authoritative"
    if relative.startswith("08_LOGS/"):
        return "log", "supporting"
    return "archive documentation", "supporting"


def build_manifest(stage: Path, source_map: dict[str, str]) -> dict[str, Any]:
    entries = []
    for path in sorted(stage.rglob("*")):
        relative = path.relative_to(stage).as_posix()
        if path.is_file() and path.name not in {"ARCHIVE_INTERNAL_MANIFEST.json", "ARCHIVE_MANIFEST_SHA256.txt"}:
            role, authority = role_for(relative)
            entries.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "source_path": source_map.get(relative, "generated_for_cross_chat_archive"), "role": role, "authority": authority})
    manifest = {"schema_version": "stage3_h13_d28_final_cross_chat_archive_manifest_v1", "hash_algorithm": "SHA-256", "SELF_EXCLUDED": "YES", "excluded_from_self_hash": ["ARCHIVE_INTERNAL_MANIFEST.json", "ARCHIVE_MANIFEST_SHA256.txt"], "managed_file_count": len(entries), "managed_files": entries}
    write_json(stage / "ARCHIVE_INTERNAL_MANIFEST.json", manifest)
    lines = ["SELF_EXCLUDED: YES", "HASH_ALGORITHM: SHA-256", "FORMAT: sha256<TAB>size_bytes<TAB>role<TAB>authority<TAB>source_path<TAB>relative_archive_path"]
    lines.extend(f"{e['sha256']}\t{e['size_bytes']}\t{e['role']}\t{e['authority']}\t{e['source_path']}\t{e['relative_path']}" for e in entries)
    write_text(stage / "ARCHIVE_MANIFEST_SHA256.txt", "\n".join(lines))
    return manifest


def deterministic_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644 if info.isfile() else 0o755
    return info


def compress(stage: Path, temporary_archive: Path) -> None:
    raw_tar = stage.parent / f"{stage.name}.tar"
    if raw_tar.exists() or temporary_archive.exists():
        raise RuntimeError("Temporary TAR/archive already exists")
    with tarfile.open(raw_tar, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(stage.rglob("*")):
            tar.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False, filter=deterministic_filter)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=9 | lzma.PRESET_EXTREME)
    with raw_tar.open("rb") as source, temporary_archive.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(compressor.compress(chunk))
        target.write(compressor.flush())
    raw_tar.unlink()


def verify(archive: Path) -> dict[str, Any]:
    decompressed = 0
    with lzma.open(archive, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            decompressed += len(chunk)
    with tarfile.open(archive, "r:xz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        names = {member.name for member in members}
        manifest_file = tar.extractfile("ARCHIVE_INTERNAL_MANIFEST.json")
        if manifest_file is None:
            raise RuntimeError("Internal manifest missing")
        manifest = json.loads(manifest_file.read().decode("utf-8"))
        mismatches = []
        for entry in manifest["managed_files"]:
            stream = tar.extractfile(entry["relative_path"])
            if stream is None:
                mismatches.append({"path": entry["relative_path"], "reason": "missing"})
                continue
            data = stream.read()
            if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                mismatches.append({"path": entry["relative_path"], "reason": "size_or_hash"})
        expected = {entry["relative_path"] for entry in manifest["managed_files"]} | {"ARCHIVE_INTERNAL_MANIFEST.json", "ARCHIVE_MANIFEST_SHA256.txt"}
        topology = sorted(names.symmetric_difference(expected))
        def member_bytes(name: str) -> bytes:
            stream = tar.extractfile(name)
            if stream is None:
                raise RuntimeError(f"Required cross-check member missing: {name}")
            return stream.read()
        start = member_bytes("START_HERE_NEXT_CHAT.txt").decode("utf-8")
        summary = member_bytes("TASK_AND_RESULTS_SUMMARY.md").decode("utf-8")
        final_report = member_bytes("02_FINAL_REPORTS/FINAL_REPORT.md").decode("utf-8")
        trajectory = json.loads(member_bytes("03_RESULTS/D28_FORWARD_TRAJECTORY.json"))
        classification = json.loads(member_bytes("02_FINAL_REPORTS/FINAL_CLASSIFICATION.json"))
        checkpoint_identity = json.loads(member_bytes("07_CHECKPOINTS/CHECKPOINT_IDENTITY.json"))
        embedded_checkpoint_sha = hashlib.sha256(member_bytes("07_CHECKPOINTS/committed_update_412.pt")).hexdigest()
        cross_document_consistency = all("D28_FORWARD_PROGRESS_CONFIRMED" in text and "7.926021327879009" in text and FINAL_SHA in text for text in (start, summary, final_report)) and start.rstrip().endswith("THIS_ARCHIVE_DOCUMENTS THE COMPLETED TASK ONLY.") and summary.rstrip().endswith("THIS_ARCHIVE_DOCUMENTS THE COMPLETED TASK ONLY.") and trajectory["measurements"]["updates_committed"] == [407, 408, 409, 410, 411, 412] and trajectory["measurements"]["H1_final"] == 7.926021327879009e-05 and trajectory["measurements"]["H32_final"] == 0.01751783891085687 and classification["classification"] == "D28_FORWARD_PROGRESS_CONFIRMED" and embedded_checkpoint_sha == checkpoint_identity["sha256"] == FINAL_SHA
    size = archive.stat().st_size
    required = {name: name in names for name in ("START_HERE_NEXT_CHAT.txt", "TASK_AND_RESULTS_SUMMARY.md", "ARCHIVE_CONTENTS_GUIDE.md", "TASK_PROMPT.md", "CODEX_FINAL_RESPONSE.md", "02_FINAL_REPORTS/FINAL_REPORT.md", "03_RESULTS/D28_FORWARD_TRAJECTORY.json", "04_VALIDATION/OPTIMIZER_CONTINUITY_REPORT.json", "05_PROVENANCE/D28_EXECUTION_MANIFEST.json", "06_IMPLEMENTATION/IMPLEMENTATION_CHANGES.md", "07_CHECKPOINTS/committed_update_412.pt")}
    passed = not mismatches and not topology and all(required.values()) and size <= LIMIT and cross_document_consistency
    return {"overall": "PASS" if passed else "FAIL", "archive_path": str(archive.resolve()), "archive_size_bytes": size, "archive_size_mib": size / (1024 * 1024), "size_limit_bytes": LIMIT, "size_limit_pass": size <= LIMIT, "archive_sha256": sha256_file(archive), "archive_file_count": len(members), "compression": "XZ / LZMA2 preset 9 EXTREME", "xz_decompression_test": "PASS", "tar_content_read_test": "PASS", "internal_manifest_verification": "PASS" if not mismatches and not topology else "FAIL", "cross_document_consistency": "PASS" if cross_document_consistency else "FAIL", "embedded_checkpoint_sha256": embedded_checkpoint_sha, "decompressed_tar_bytes": decompressed, "required_files": required, "mismatches": mismatches, "topology_difference": topology}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()
    output, prompt = args.output.resolve(), args.prompt.resolve()
    trajectory, execution, completion = load_result(output)
    stage = ROOT / "tmp" / f"stage3_h13_d28_cross_chat_stage_{args.stamp}"
    temporary_archive = ROOT / "tmp" / f"STAGE3_H13_D28_FORWARD_CONTINUATION_CROSS_CHAT_EVIDENCE_MAX_XZ_{args.stamp}.tar.xz"
    final_archive = DESKTOP / temporary_archive.name
    if stage.exists() or temporary_archive.exists() or final_archive.exists():
        raise SystemExit("Refusing to overwrite staging or archive path")
    stage.mkdir(parents=True)
    populate(stage, output, prompt, trajectory, execution, completion)
    source_map: dict[str, str] = {}
    build_manifest(stage, source_map)
    compress(stage, temporary_archive)
    pre_move = verify(temporary_archive)
    if pre_move["overall"] != "PASS":
        raise RuntimeError(f"Temporary archive verification failed: {pre_move}")

    competing = sorted(DESKTOP.glob("*D28*.tar.xz"))
    for old in competing:
        resolved = old.resolve()
        if resolved.parent != DESKTOP.resolve() or not resolved.name.endswith(".tar.xz") or "D28" not in resolved.name:
            raise RuntimeError(f"Unsafe competing archive target: {resolved}")
        resolved.unlink()
    shutil.move(str(temporary_archive), str(final_archive))
    verification = verify(final_archive)
    remaining = sorted(DESKTOP.glob("*D28*.tar.xz"))
    verification["desktop_d28_archive_count"] = len(remaining)
    verification["single_canonical_archive"] = len(remaining) == 1 and remaining[0].resolve() == final_archive.resolve()
    verification["overall"] = "PASS" if verification["overall"] == "PASS" and verification["single_canonical_archive"] else "FAIL"
    write_json(output / "FINAL_CROSS_CHAT_ARCHIVE_VERIFICATION.json", verification)
    if verification["overall"] == "PASS":
        shutil.rmtree(stage)
    print(json.dumps(verification, sort_keys=True))
    return 0 if verification["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
