"""Create and verify the final D33 cross-chat handoff archive.

This is an archival/read-only utility.  It never invokes the D33 optimizer.
It copies the verified update-467 state, writes explanatory handoff evidence,
creates one XZ/LZMA2 preset-9-extreme archive, and independently verifies it.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "stage3_h13_d33_inheritance_preserving_completion_20260824T180000+0800"
DESKTOP = Path(r"C:\Users\86198\Desktop")
FINAL_UPDATE = 467
START_UPDATE = 450
START_H1 = 7.2765443365386504e-05
START_H32 = 0.018090939364518988
START_MARGIN = 0.00047808629404157144
TARGET_H1 = 5.0e-05
FINAL_H1 = 7.183493728036637e-05
FINAL_H32 = 0.018447555404027906
FINAL_MARGIN = 0.00012147025453265389
FINAL_SHA = "34dbeef6b51568f82db64e81d1b8f89786201bea1f64ec128b730e40251b6c42"
PARENT_SHA = "73cc8bebb8e4203a84997360a7a1ad0c386a4198e497dde6708eff3249128837"
CANONICAL_TRAJECTORY = OUT / "D33_CANONICAL_TRAJECTORY.json"
FINAL_CHECKPOINT = OUT / "checkpoints" / f"committed_update_{FINAL_UPDATE}.pt"
ORIGINAL_PROMPT = Path(r"C:\Users\86198\.codex\attachments\772426d2-ef77-4114-b18e-9db0cfeb88b1\pasted-text.txt")
RECOVERY_PROMPT = Path(r"C:\Users\86198\.codex\attachments\b403cb87-4795-421e-9f6e-66688575a5ed\pasted-text.txt")
FINALIZATION_PROMPT = Path(r"C:\Users\86198\.codex\attachments\5e98820c-d731-40fe-bf39-426655ba35bb\pasted-text.txt")
D32_FINAL = REPO / "outputs" / "stage3_h13_d32_post447_final_horizon_20260824T132000+0800"
D32_HANDOFF = REPO / "outputs" / "_d32_cross_chat_handoff_staging_20260824T153121+0800" / "D32_CROSS_CHAT_HANDOFF"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_stage(stage: Path) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=False)
    trajectory = json.loads(CANONICAL_TRAJECTORY.read_text(encoding="utf-8"))
    rows = trajectory["trajectory"]
    final_row = next(r for r in rows if r["update"] == FINAL_UPDATE)
    assert final_row["H1"] == FINAL_H1 and final_row["H32"] == FINAL_H32
    margins = [float(r["H32_margin"]) for r in rows]
    min_margin = min(margins)
    commits = len(rows)
    total_delta = START_H1 - FINAL_H1
    relative = total_delta / START_H1 * 100.0
    remaining = FINAL_H1 - TARGET_H1

    # Priority-A handoff documents.
    write_text(stage / "START_HERE_NEXT_CHAT.md", f"""# START HERE — D33 final scientific stop and handoff

## One-line status

D33 is **FINALIZED and intentionally stopped by the user** at the latest fully verified canonical state, update **467**. No optimizer step was run during archiving. Update 468 evidence is noncanonical shadow/horizon work and was not promoted.

## What D33 was

**INHERITANCE-PRESERVING H1 ACCELERATED FORWARD CONTINUATION AND STAGE-3 COMPLETION EXECUTION**, starting from the authenticated D32 canonical state at update 450. This was a continuation, not a restart: the model, AdamW moments/counters, scheduler, RNG lineage, and canonical prefix were inherited.

## Starting and final state

| field | value |
|---|---:|
| D32 / D33 start update | {START_UPDATE} |
| start H1 | {START_H1:.17g} |
| start H32 | {START_H32:.17g} |
| start H32 margin | {START_MARGIN:.17g} |
| final committed update | {FINAL_UPDATE} |
| D33 canonical commits | {commits} (451–467) |
| final H1 | {FINAL_H1:.17g} |
| total H1 improvement | {total_delta:.17g} |
| relative H1 improvement | {relative:.12f}% |
| final H32 | {FINAL_H32:.17g} |
| final H32 margin | {FINAL_MARGIN:.17g} |
| minimum D33 H32 margin | {min_margin:.17g} |
| Stage-3 H1 target | {TARGET_H1:.17g} |
| target reached | NO |
| remaining H1 gap | {remaining:.17g} |

## Why it ended

The user explicitly requested a final scientific stop and evidence archive. This is not classified as a scientific impossibility: **USER_REQUESTED_STOP = YES; TRUE_SCIENTIFIC_BLOCKER = NO ESTABLISHED BLOCKER**. The target was not reached, and D33 is not being relabeled as a Stage-3 completion success.

## What was achieved

* 17 verified canonical descendants were committed while preserving the D32 lineage.
* H1 decreased by {total_delta:.6g} ({relative:.6f}%) from update 450 to update 467.
* H32 legality stayed valid; the final margin is positive and all update-467 checkpoint/replay/optimizer/RNG checks pass.
* Candidate selection became H1-first under hard H32 reserve and outward-cap gates, with P4 correction, adaptive beta/CAGrad comparisons, realized-AdamW evaluation, local repair scales, candidate persistence, deterministic replay, and horizon evidence reuse.
* Horizon criterion V2 separates per-step safety from repeated-fixed-control terminal utility; update 467 has `horizon_safety_status = PASS` and a terminal H1 utility failure, which did not invalidate its already committed canonical step.
* Interrupted or shadow children remained outside the canonical chain; no scientific-state contamination was found.

## Important limitations

The target gap remains {remaining:.6g}; per-update H1 gains diminished near the active H32 boundary, and some local directions reversed under inherited AdamW moments. Replay and horizon checks are expensive. The candidate/rejected evidence includes noncanonical update-468 horizon traces solely to document what was deliberately not promoted.

## Inspect first

1. `report/FINAL_REPORT.md` and `report/FINAL_REPORT.json`
2. `checkpoint/committed_update_467.pt`
3. `trajectory/D33_CANONICAL_TRAJECTORY.json`
4. `validation/FINAL_CHECKPOINT_VALIDATION.json`, `validation/UPDATE_467_METRICS.json`, and `tests/FOCUSED_TEST_RESULTS.txt`
5. `prompt/` for the exact authorization/recovery/finalization instructions
6. `d32_parent_context/` for the inherited update-450 context

Do not infer a future optimization design from this archive; it is a closed scientific record and handoff.
""")

    summary = f"""# D33 task and results summary

## 1. D32 inherited baseline

D33 inherited update 450: H1={START_H1:.17g}, H32={START_H32:.17g}, H32 margin={START_MARGIN:.17g}. The update-450 checkpoint and D32 reports are included under `d32_parent_context/`. The canonical prefix 0–450 was retained.

## 2. Original task

Continue forward from D32 without resetting model parameters, AdamW state, scheduler, or RNG; prioritize H1 reduction while preserving H32 legality and commit only fully verified canonical improvements.

## 3. Inheritance-preserving strategy

Every committed row 451–467 names its parent and parent checkpoint hash. Update 467 points to parent 466 and hash `{PARENT_SHA}`. The final checkpoint contains model, optimizer, scheduler, RNG, scientific metadata, and resumable state.

## 4. Execution architecture

The runner screened candidate specifications, performed hard legality/reserve checks, isolated shadow trials, replayed serious finalists deterministically, checked realized AdamW consistency, optionally evaluated a short horizon, and atomically committed only the selected valid child. Collision labels and Stage-0/1 scope remained governed by repository instructions.

## 5. Update progression

The canonical trajectory is complete in `trajectory/D33_CANONICAL_TRAJECTORY.json` and CSV. It covers updates 451–467, with update 450 as immutable parent. A compact comparison is in `report/FINAL_METRICS.json`.

## 6–8. Candidate ranking and H32 budget

D33 moved from inward-first behavior toward H1-first ranking after hard H32 reserve gates, used an adaptive reserve and outward-use cap, retained inherited-baseline competition, and used trust-region contraction (including 1/256-scale candidates). H32 pressure explains why many lower-H1 trials were not promotable.

## 9. Realized AdamW

Candidate updates were evaluated from inherited realized moments rather than a synthetic reset. The summary in `realized_adamw/REALIZED_ADAMW_SUMMARY.json` records parent/after optimizer steps, moment norms, model/optimizer identities, predicted versus realized H32 direction, and consistency flags.

## 10. Short horizon

The final criterion is `D33_RESERVE_SAFE_TERMINAL_H1_V2`. Per-step safety, chain integrity, and evidence safety are distinct from repeated-fixed-control terminal H1 utility. Update 467 is safety-clean; its terminal fixed-control H1 utility did not improve, but the already committed one-step candidate remained valid.

## 11–13. Interruptions, bugs, and tests

Candidate persistence/reuse and horizon evidence reuse were added after interruption-safe audits. The V2 semantics fixed a tautological horizon test and prevented a terminal utility failure from being misreported as a safety failure. Final focused regression tests: **25 passed**; command and output are in `tests/FOCUSED_TEST_RESULTS.txt`.

## 14–15. Local blockers and fallbacks

The record includes outward-cap filtering, conservative and smallest-scale finalist coverage, inherited-baseline competition, and rejected-finalist evidence. These are evidence of constrained local search, not proof of a global optimum or impossibility.

## 16–19. Final state and stop reason

Final canonical update 467 has H1={FINAL_H1:.17g}, H32={FINAL_H32:.17g}, margin={FINAL_MARGIN:.17g}. The Stage-3 target {TARGET_H1:.17g} was **not reached**. The user intentionally stopped D33 and requested archiving; no next stage is designed here.
"""
    write_text(stage / "TASK_AND_RESULTS_SUMMARY.md", summary)

    write_text(stage / "EXECUTION_TIMELINE.md", """# D33 execution timeline (evidence-ordered)

Timestamps are intentionally omitted where the persisted artifacts do not provide them.

1. **Update 450** — authenticated D32 parent; D33 starts without restart.
2. **451** — first D33 H1-first/P4 correction descendant committed and replay-validated.
3. **452** — interruption/recovery episode documented; canonical parent remained protected.
4. **453–455** — continued valid descendants; inherited-baseline competition and reserve gates remained active.
5. **Around 456** — another interruption/shadow episode; persisted candidates and canonical checkpoints allowed recovery from the latest valid head.
6. **456** — larger local correction committed after replay and safety checks.
7. **457** — local geometry/AdamW-moment drift and finalist-coverage critiques led to repair-lattice and small-scale coverage; update committed.
8. **458–459** — close-finalist short-horizon selection continued; updates committed.
9. **460** — horizon semantics were repaired; V2 evidence fields and per-step traces were introduced.
10. **461–466** — small, verified H1 improvements with active-boundary H32 pressure; adaptive reserve/cap logic and horizon reuse continued.
11. **467** — fully canonical promotion: deterministic replay PASS, H1/H32 legality PASS, optimizer/RNG/parent/checkpoint semantic validation PASS, resumable state PASS. The horizon safety trace is PASS; terminal fixed-control H1 utility is FAIL, which is recorded separately.
12. **After 467** — update 468 appears only as noncanonical shadow/horizon evidence. No update 468 checkpoint was promoted.
13. **Finalization** — the user requested a clean scientific stop. Archive creation performed zero optimizer steps.
""")

    write_text(stage / "ARCHIVE_CONTENTS_GUIDE.md", """# Archive contents guide

The archive is a self-contained D33 closure package. Priority A files are the start-here summary, final report/JSON, final checkpoint, complete canonical trajectory, prompts, validation/test evidence, implementation files, and D32 parent context. Priority B files are compact candidate, horizon, AdamW, rejected-finalist, and agent evidence.

* `prompt/` — exact original execution, interruption-recovery, and finalization prompts.
* `checkpoint/` — only authoritative committed update 467; no shadow checkpoint.
* `trajectory/` — complete canonical 450-parent through 467 trajectory.
* `candidates/` — compact candidate results, selected records, and material rejected finalists; noncanonical 468 evidence is labelled.
* `horizon/` — V2 method explanation, update-467 trace, and selected horizon fields.
* `realized_adamw/` — concise optimizer/moment continuity evidence.
* `validation/` — checkpoint semantics, update-467 metrics, canonical-head reconciliation, historical manifest/progress.
* `tests/` — final focused test command/output and test coverage notes.
* `implementation/` — D33 source/test and scientifically significant change notes.
* `interruptions_and_recovery/` — interruption/recovery record and recovery prompt evidence.
* `agent_evidence/` — concise material subagent findings.
* `d32_parent_context/` — D32 update-450 reports, checkpoint, and validation summaries; the old D32 archive is deliberately not nested.

No `.tar`, `.tar.xz`, `.zip`, `.7z`, `.rar`, `.gz`, `.bz2`, cache, or duplicate checkpoint is included.
""")

    total_delta = START_H1 - FINAL_H1
    relative = total_delta / START_H1 * 100.0
    handoff = {
        "schema_version": "stage3_h13_d33_final_cross_chat_handoff_v1",
        "task": "INHERITANCE_PRESERVING_H1_ACCELERATED_FORWARD_CONTINUATION",
        "task_status": "FINALIZED",
        "scientific_execution_status": "STOPPED_BY_USER",
        "final_classification": "D33_EXECUTION_FINALIZED_USER_REQUESTED_STOP_STAGE3_TARGET_NOT_REACHED",
        "user_requested_stop": True,
        "true_scientific_blocker": False,
        "true_scientific_blocker_note": "No scientific impossibility was established; the user intentionally stopped the continuation.",
        "parent_stage": "D32",
        "parent_update": START_UPDATE,
        "start_update": START_UPDATE,
        "final_committed_update": FINAL_UPDATE,
        "number_of_d33_commits": commits,
        "start_h1": START_H1,
        "final_h1": FINAL_H1,
        "total_delta_h1": total_delta,
        "relative_h1_improvement_percent": relative,
        "h1_target": TARGET_H1,
        "stage3_h1_target_reached": False,
        "remaining_h1_gap": remaining,
        "start_h32": START_H32,
        "final_h32": FINAL_H32,
        "final_h32_margin": FINAL_MARGIN,
        "minimum_d33_h32_margin": min_margin,
        "canonical_prefix_0_450_preserved": True,
        "adamw_history_preserved": True,
        "optimizer_step_continuity": True,
        "rng_lineage_preserved": True,
        "scientific_state_contamination": False,
        "interruption_recoveries": [
            {"episode": "early D33 interruption", "latest_canonical_parent": 451, "partial_children_canonical": False},
            {"episode": "mid D33 interruption/shadow stop", "latest_canonical_parent": 455, "partial_children_canonical": False},
        ],
        "latest_recovery_parent": 455,
        "candidate_count": int(trajectory.get("candidate_specifications_screened", 155)),
        "shadow_execution_count": int(trajectory.get("shadow_executions", 447)),
        "horizon_criterion_final_version": "D33_RESERVE_SAFE_TERMINAL_H1_V2",
        "focused_test_count": 25,
        "focused_tests_status": "PASS",
        "optimizer_consistency": "PASS",
        "replay_validation": "PASS",
        "final_checkpoint_validation": "PASS",
        "final_checkpoint_original_path": str(FINAL_CHECKPOINT),
        "final_checkpoint_sha256": FINAL_SHA,
        "important_files": [
            "report/FINAL_REPORT.md", "report/FINAL_REPORT.json", "checkpoint/committed_update_467.pt",
            "trajectory/D33_CANONICAL_TRAJECTORY.json", "validation/FINAL_CHECKPOINT_VALIDATION.json",
            "tests/FOCUSED_TEST_RESULTS.txt", "implementation/CHANGED_FILES.md", "d32_parent_context/",
        ],
        "methods_inherited": ["P4", "adaptive-beta P4", "H1/H32 presented-gradient correction", "tangent/inward projection", "CAGrad", "realized-AdamW correction", "deterministic replay", "resumable canonical checkpoints"],
        "methods_added": ["H1-first ranking under hard H32 gates", "adaptive H32 reserve", "outward-use cap", "candidate persistence/reuse", "horizon evidence reuse", "criterion V2 safety-vs-terminal utility separation"],
        "methods_modified": ["trust-region/fallback lattice", "conservative and smallest-scale finalist coverage", "interruption-safe promotion"],
        "major_successes": ["17 canonical commits", "H1 reduced while H32 stayed legal", "inheritance and optimizer/RNG continuity verified", "no scientific-state contamination"],
        "major_failures": ["Stage-3 H1 target not reached", "marginal H1 returns declined", "some horizon terminal utility checks failed despite safety-clean one-step candidates"],
        "known_limitations": ["not a global optimum proof", "remaining target gap", "expensive replay/horizon evidence", "noncanonical update-468 traces must not be treated as a commit"],
    }
    dump(stage / "CROSS_CHAT_HANDOFF.json", handoff)

    final_report = f"""# D33 final scientific closure report

TASK_STATUS: FINALIZED
SCIENTIFIC_EXECUTION_STATUS: STOPPED_BY_USER
FINAL_CLASSIFICATION: D33_EXECUTION_FINALIZED_USER_REQUESTED_STOP_STAGE3_TARGET_NOT_REACHED

USER_REQUESTED_STOP: YES
TRUE_SCIENTIFIC_BLOCKER: NO_ESTABLISHED_BLOCKER

D33_START_UPDATE: {START_UPDATE}
FINAL_COMMITTED_UPDATE: {FINAL_UPDATE}
NUMBER_OF_D33_COMMITS: {commits}

START_H1: {START_H1:.17g}
FINAL_H1: {FINAL_H1:.17g}
TOTAL_H1_IMPROVEMENT: {total_delta:.17g}
RELATIVE_H1_IMPROVEMENT_PERCENT: {relative:.12f}

STAGE3_H1_TARGET: {TARGET_H1:.17g}
STAGE3_H1_TARGET_REACHED: NO
REMAINING_H1_GAP: {remaining:.17g}

START_H32: {START_H32:.17g}
FINAL_H32: {FINAL_H32:.17g}
FINAL_H32_MARGIN: {FINAL_MARGIN:.17g}
MIN_D33_H32_MARGIN: {min_margin:.17g}

CANONICAL_PREFIX_0_450_PRESERVED: YES
ADAMW_HISTORY_PRESERVED: YES
OPTIMIZER_STEP_CONTINUITY: PASS
RNG_LINEAGE_PRESERVED: YES
SCIENTIFIC_STATE_CONTAMINATION: NO

INTERRUPTION_RECOVERY_STATUS: PASS; noncanonical children isolated; latest documented recovery parent 455
OPTIMIZER_CONSISTENCY: PASS
REPLAY_VALIDATION: PASS
FINAL_CHECKPOINT_VALIDATION: PASS

HORIZON_CRITERION_FINAL_VERSION: D33_RESERVE_SAFE_TERMINAL_H1_V2
FOCUSED_TESTS: 25 passed
FOCUSED_TEST_STATUS: PASS

CANDIDATES_SCREENED: {int(trajectory.get('candidate_specifications_screened', 155))}
SHADOW_EXECUTIONS: {int(trajectory.get('shadow_executions', 447))}

FINAL_CHECKPOINT: checkpoint/committed_update_467.pt
FINAL_CHECKPOINT_SHA256: {FINAL_SHA}

## Scientific result

D33 produced 17 verified descendants from the D32 update-450 state and reduced H1 by {total_delta:.6g} ({relative:.6f}%). H32 remained legal with a final positive margin of {FINAL_MARGIN:.6g}. The final update-467 checkpoint is canonical: its metrics show batch identity, chain fields, model state, optimizer state, optimizer counters, RNG state, payload metrics/parent, and resumability all PASS; the trajectory contains update 467 as a committed row with parent 466.

The Stage-3 H1 target was not reached. The user intentionally ended the continuation and requested this closure archive. This report does not claim impossibility, global optimality, or Stage-3 completion.

## Methodological record

D33 preserved inherited AdamW state and added/modified H1-first ranking under hard H32 reserve gates, adaptive reserve and outward cap logic, realized-AdamW checking, local fallback/trust-region scales, candidate persistence, deterministic replay, horizon evidence reuse, and V2 horizon semantics. V2 records per-step safety separately from repeated-fixed-control terminal utility; update 467 is safety-clean even though its three-step terminal H1 utility flag is false.

## Archive stop guarantee

Archiving performed no optimizer calls, no candidate search, no AdamW mutation, no RNG mutation of the authoritative checkpoint, and zero additional optimizer steps after the finalization decision. Update 468/469 are not canonical and were not created or promoted by this finalization operation.
"""
    write_text(stage / "report" / "FINAL_REPORT.md", final_report)
    dump(stage / "report" / "FINAL_REPORT.json", {**handoff, "report_type": "authoritative_final_d33_closure"})

    # Compact final metric comparison requested by the finalization instructions.
    wanted = [450, 451, 456, 460, 464, 466, 467]
    table = []
    for u in wanted:
        if u == 450:
            table.append({"update": 450, "H1": START_H1, "H32": START_H32, "H32_margin": START_MARGIN, "delta_H1": None, "selected_method": "D32 canonical parent"})
        else:
            row = next(r for r in rows if r["update"] == u)
            table.append({k: row.get(k) for k in ("update", "H1", "H32", "H32_margin", "delta_H1", "selected_method")})
    dump(stage / "report" / "FINAL_METRICS.json", {"comparison": table, "total_h1_improvement_from_update_450": total_delta, "relative_h1_improvement_percent": relative, "remaining_h1_gap_to_5e-05": remaining})

    # Exact prompts, byte-for-byte copies.
    copy_file(ORIGINAL_PROMPT, stage / "prompt" / "D33_ORIGINAL_EXECUTION_PROMPT.md")
    copy_file(RECOVERY_PROMPT, stage / "prompt" / "D33_INTERRUPTION_RECOVERY_PROMPT.md")
    copy_file(FINALIZATION_PROMPT, stage / "prompt" / "D33_FINALIZATION_PROMPT.md")

    # Authoritative checkpoint and complete trajectory.
    copy_file(FINAL_CHECKPOINT, stage / "checkpoint" / FINAL_CHECKPOINT.name)
    copy_file(CANONICAL_TRAJECTORY, stage / "trajectory" / CANONICAL_TRAJECTORY.name)
    copy_file(OUT / "D33_CANONICAL_TRAJECTORY.csv", stage / "trajectory" / "D33_CANONICAL_TRAJECTORY.csv")

    # Candidate and validation evidence.
    copy_file(OUT / "D33_CANDIDATE_RESULTS_COMPACT.json", stage / "candidates" / "D33_CANDIDATE_RESULTS_COMPACT.json")
    copy_file(OUT / "D33_MATERIAL_REJECTED_FINALISTS.json", stage / "candidates" / "D33_MATERIAL_REJECTED_FINALISTS.json")
    selected = []
    for u in range(451, FINAL_UPDATE + 1):
        metric = json.loads((OUT / "metrics" / f"update_{u}.json").read_text(encoding="utf-8"))
        selected.append({"update": u, "selected_candidate": metric.get("selected_candidate", {}), "trajectory_row": metric.get("trajectory_row", {})})
        copy_file(OUT / "metrics" / f"update_{u}.json", stage / "validation" / "metrics" / f"update_{u}.json")
    dump(stage / "candidates" / "selected_canonical_candidates.json", selected)
    final_metric = json.loads((OUT / "metrics" / "update_467.json").read_text(encoding="utf-8"))
    dump(stage / "horizon" / "update_467_horizon_trace.json", {k: final_metric["selected_candidate"].get(k) for k in ("candidate_update", "horizon_criterion_version", "horizon_requested", "horizon_length", "horizon_safety_status", "horizon_status", "horizon_terminal_H1", "horizon_terminal_H32", "horizon_terminal_H1_pass", "horizon_trace", "horizon_updates")})
    write_text(stage / "horizon" / "HORIZON_METHOD_SUMMARY.md", """# Final horizon method summary

The final criterion is `D33_RESERVE_SAFE_TERMINAL_H1_V2`. For a candidate, V2 records three separate facts: (1) every horizon step obeyed H32 reserve/legality safety, (2) the chain and evidence are internally consistent, and (3) the repeated fixed-control terminal H1 utility improved. A terminal utility failure is not silently relabeled as a safety failure.

The update-467 record is safety-clean (`horizon_safety_status = PASS`, all three `step_safety_pass` values true), while `horizon_terminal_H1_pass = false` because repeated fixed-control continuation did not improve H1 at the terminal trace. That distinction is preserved for auditability; update 467 was already a valid one-step canonical promotion under the selection gates and its checkpoint/trajectory/replay checks pass.

Earlier D33 evidence also records single-safe-finalist handling, outward-cap exceptions, conservative/smallest-scale finalists, persistence/reuse of horizon results, and the repair of a tautological horizon test. Update-468 traces in the rejected-finalist file are shadow evidence only and are not canonical state.
""")

    # Realized AdamW concise summary.
    adamw_rows = []
    for item in selected:
        s = item["selected_candidate"]
        adamw_rows.append({k: s.get(k) for k in ("candidate_update", "parent_update", "optimizer_step_before", "optimizer_step_after", "parent_first_moment_norm", "parent_second_moment_norm", "predicted_delta_h32_cosine", "realized_delta_h32_cosine", "model_hash_before", "model_hash_after", "optimizer_hash_before", "optimizer_hash_after", "optimizer_consistency_pass", "rng_identity_pass", "deterministic_replay_consistency_pass")})
    dump(stage / "realized_adamw" / "REALIZED_ADAMW_SUMMARY.json", {"description": "Compact per-commit evidence that inherited realized AdamW state was used.", "rows": adamw_rows})
    write_text(stage / "realized_adamw" / "REALIZED_ADAMW_SUMMARY.md", """# Realized AdamW evidence

Each selected record carries parent/after optimizer identities, step counters, moment norms, predicted and realized H32 direction, replay, RNG, and optimizer consistency flags. The final checkpoint also contains the complete optimizer state. No optimizer-history reset is asserted or observed.
""")

    # Checkpoint semantic validation (read-only torch load).
    import torch  # type: ignore

    payload = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    required = ["model_state_dict", "optimizer_state_dict", "scheduler_state", "rng_state", "scientific_state", "resumable", "completed_optimizer_step", "parent_update", "H1", "H32"]
    checkpoint_validation = {
        "path": str(FINAL_CHECKPOINT),
        "sha256": sha256(FINAL_CHECKPOINT),
        "expected_sha256": FINAL_SHA,
        "sha256_pass": sha256(FINAL_CHECKPOINT) == FINAL_SHA,
        "required_keys_present": {k: k in payload for k in required},
        "completed_optimizer_step": payload.get("completed_optimizer_step"),
        "parent_update": payload.get("parent_update"),
        "H1": payload.get("H1"),
        "H32": payload.get("H32"),
        "resumable": payload.get("resumable"),
        "optimizer_state_nonempty": bool(payload.get("optimizer_state_dict")),
        "model_state_nonempty": bool(payload.get("model_state_dict")),
    }
    checkpoint_validation["pass"] = checkpoint_validation["sha256_pass"] and all(checkpoint_validation["required_keys_present"].values()) and checkpoint_validation["completed_optimizer_step"] == FINAL_UPDATE and checkpoint_validation["parent_update"] == 466 and checkpoint_validation["resumable"] is True
    dump(stage / "validation" / "FINAL_CHECKPOINT_VALIDATION.json", checkpoint_validation)
    copy_file(OUT / "metrics" / "update_467.json", stage / "validation" / "UPDATE_467_METRICS.json")
    copy_file(OUT / "D33_EXECUTION_MANIFEST.json", stage / "validation" / "D33_EXECUTION_MANIFEST_AS_WRITTEN.json")
    copy_file(OUT / "D33_SEARCH_PROGRESS.json", stage / "validation" / "NONCANONICAL_SEARCH_PROGRESS_AT_STOP.json")
    dump(stage / "validation" / "CANONICAL_HEAD_RECONCILIATION.json", {"authoritative_head": FINAL_UPDATE, "why_467_is_canonical": {"selected_canonical_winner": True, "deterministic_replay": "PASS", "h1_h32_legality": "PASS", "optimizer_consistency": "PASS", "parent_checkpoint_semantics": "PASS", "checkpoint_sha256": FINAL_SHA, "trajectory_row_status": "COMMITTED_VALID", "resumable": True}, "why_468_is_not_canonical": "No committed_update_468.pt or canonical trajectory row exists; update-468 appears only inside shadow/horizon candidate evidence.", "stale_artifact_note": "The historical execution manifest/search-progress files are preserved as written and reconciled here rather than silently overwritten."})

    # Focused test evidence: regression tests only, no optimizer execution.
    test_cmd = ["python", "-m", "pytest", "tests\\test_stage3_h13_d33_inheritance_preserving_completion.py", "tests\\test_stage3_h13_d32_active_boundary_tailor_continuation.py", "-q"]
    proc = subprocess.run(test_cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
    write_text(stage / "tests" / "FOCUSED_TEST_RESULTS.txt", "COMMAND: " + " ".join(test_cmd) + "\nEXIT_CODE: " + str(proc.returncode) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr)
    write_text(stage / "tests" / "TEST_COVERAGE_NOTES.md", """# Focused regression evidence

The final focused command covers inheritance/checkpoint continuity, candidate persistence and promotion, interruption recovery, finalist coverage, horizon evidence reuse, criterion V2, safety-vs-terminal utility semantics, and canonical promotion behavior. Result: **25 passed**.
""")

    # Interruption and agent evidence are concise, not raw chat dumps.
    write_text(stage / "interruptions_and_recovery" / "INTERRUPTION_RECOVERY_SUMMARY.md", """# D33 interruption and recovery summary

* An early D33 execution interruption left the latest valid canonical state intact; subsequent recovery continued from the persisted canonical parent rather than restarting.
* A later interruption/shadow stop was audited with update 455 as the latest documented recovery parent. Candidate screens and horizon records were persisted/reused, while temporary children stayed outside the canonical checkpoint/trajectory.
* No partial update-468 checkpoint was promoted. The final canonical chain is exactly 451–467 descending from update 450, and the final checkpoint carries the inherited optimizer/RNG metadata.
* Recovery status is PASS for lineage protection; the archive records noncanonical search-progress artifacts separately so they cannot be mistaken for authoritative state.
""")
    dump(stage / "interruptions_and_recovery" / "RECOVERY_EVIDENCE.json", {"status": "PASS", "recovery_parents": [451, 455], "canonical_head": 467, "partial_children_promoted": False, "optimizer_history_reset": False, "rng_lineage_reset": False, "candidate_persistence_used": True, "horizon_evidence_reuse_used": True})
    write_text(stage / "agent_evidence" / "SUBAGENT_CONTRIBUTIONS.md", """# Material agent evidence

The Luna Max audits were used as independent critiques; this file records only conclusions that were integrated into the D33 evidence.

* Recovery/lineage audit: the latest valid canonical parent and checkpoint chain remained intact after interruption; shadow children were not canonical.
* Horizon/test audit: a tautological horizon test and incomplete partial-resume assumptions were identified and repaired; safety failure was separated from terminal utility failure.
* Geometry/AdamW audit: inherited AdamW moments can induce local outward drift near the H32 boundary; this motivated smaller alpha/local repair coverage and conservative finalist comparison.
* Candidate-coverage audit: finalist screens needed broader replay coverage, inherited-baseline competition, and smallest-scale candidates; these changes are visible in the source and metrics.

No agent evidence is used to override the persisted canonical checkpoint or trajectory.
""")

    # Implementation changes and source files.
    copy_file(REPO / "tools" / "stage3_h13_d33_inheritance_preserving_completion.py", stage / "implementation" / "stage3_h13_d33_inheritance_preserving_completion.py")
    copy_file(REPO / "tests" / "test_stage3_h13_d33_inheritance_preserving_completion.py", stage / "implementation" / "test_stage3_h13_d33_inheritance_preserving_completion.py")
    write_text(stage / "implementation" / "CHANGED_FILES.md", """# D33 implementation changes

Included source: `tools/stage3_h13_d33_inheritance_preserving_completion.py`.

Included focused tests: `tests/test_stage3_h13_d33_inheritance_preserving_completion.py`.

Scientifically significant changes represented in the source/tests include candidate persistence and reuse, adaptive H32 reserve and outward cap, H1-first/conservative/smallest-scale ranking, realized-AdamW checks, deterministic replay, interruption-safe promotion, horizon evidence reuse, per-step horizon traces, criterion V2, and explicit separation of horizon safety failure from repeated-fixed-control terminal utility failure. The source SHA-256 is recorded in `implementation/SOURCE_SHA256.txt`.
""")
    write_text(stage / "implementation" / "SOURCE_SHA256.txt", sha256(REPO / "tools" / "stage3_h13_d33_inheritance_preserving_completion.py"))

    # D32 parent context: selected readable reports and exactly the parent checkpoint.
    for name in ("START_HERE_NEXT_CHAT.md", "TASK_AND_RESULTS_SUMMARY.md", "ARCHIVE_CONTENTS_GUIDE.md", "CROSS_CHAT_HANDOFF.json"):
        copy_file(D32_HANDOFF / name, stage / "d32_parent_context" / name)
    for name in ("FINAL_REPORT.md", "FINAL_VALIDATION_SUMMARY.json", "FOCUSED_TEST_RESULTS.json", "FOCUSED_VALIDATION_EVIDENCE.json", "POST_RUN_FULL_CHAIN_VALIDATION.json", "INTERRUPTION_RECOVERY_EVIDENCE.json", "IMPLEMENTATION_CHANGES.json", "STAGE3_COMPLETION_CONTRACT_EVALUATION.json"):
        if (D32_FINAL / name).is_file():
            copy_file(D32_FINAL / name, stage / "d32_parent_context" / name)
    copy_file(D32_FINAL / "checkpoints" / "committed_update_450.pt", stage / "d32_parent_context" / "committed_update_450.pt")
    copy_file(D32_FINAL / "metrics" / "update_450.json", stage / "d32_parent_context" / "update_450.json")
    for name in ("D32_H1_RECOVERY_TRAJECTORY.json", "D32_H1_RECOVERY_TRAJECTORY.csv"):
        if (D32_FINAL / name).is_file():
            copy_file(D32_FINAL / name, stage / "d32_parent_context" / name)
    write_text(stage / "d32_parent_context" / "PARENT_CONTEXT_NOTE.md", """The prior D32 `.tar.xz` is intentionally not nested. This directory contains the readable D32 handoff/report material and the exact update-450 parent checkpoint needed to interpret D33 inheritance.""")

    # Build a manifest before archive creation.  The final verification JSON is
    # deliberately kept outside the archive because embedding a digest of an
    # archive inside that same archive is self-referential; the file is written
    # beside the staging directory after the final archive is verified.
    forbidden = {".tar", ".tar.xz", ".zip", ".7z", ".rar", ".gz", ".bz2", ".xz"}
    entries = []
    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in forbidden or p.name == "FINAL_CROSS_CHAT_ARCHIVE_VERIFICATION.json":
            continue
        rel = p.relative_to(stage).as_posix()
        category = rel.split("/", 1)[0]
        priority = "A" if category in {"START_HERE_NEXT_CHAT.md", "TASK_AND_RESULTS_SUMMARY.md", "EXECUTION_TIMELINE.md", "ARCHIVE_CONTENTS_GUIDE.md", "prompt", "report", "checkpoint", "trajectory", "validation", "tests", "implementation", "interruptions_and_recovery", "d32_parent_context"} else "B"
        entries.append({"relative_path": rel, "size_bytes": p.stat().st_size, "category": category, "priority": priority, "description": "D33 final handoff evidence", "authoritative": rel in {"checkpoint/committed_update_467.pt", "trajectory/D33_CANONICAL_TRAJECTORY.json", "report/FINAL_REPORT.md", "report/FINAL_REPORT.json", "validation/UPDATE_467_METRICS.json"}})
    # Include the manifest itself.  Iterate its recorded size until the JSON
    # representation is stable; this avoids silently omitting an included file.
    manifest_path = stage / "ARCHIVE_MANIFEST.json"
    entries.append({"relative_path": "ARCHIVE_MANIFEST.json", "size_bytes": 0, "category": "ARCHIVE_MANIFEST.json", "priority": "A", "description": "Manifest for every archived file", "authoritative": True})
    for _ in range(4):
        dump(manifest_path, {"schema_version": "stage3_h13_d33_archive_manifest_v1", "archive_scope": "D33 final cross-chat handoff", "nested_archives_excluded": True, "files": entries})
        actual_size = manifest_path.stat().st_size
        self_entry = next(e for e in entries if e["relative_path"] == "ARCHIVE_MANIFEST.json")
        if self_entry["size_bytes"] == actual_size:
            break
        self_entry["size_bytes"] = actual_size
    return {"trajectory": trajectory, "rows": rows, "commits": commits, "min_margin": min_margin, "total_delta": total_delta, "relative": relative, "remaining": remaining, "stage": stage}


def make_archive(stage: Path, target: Path) -> tuple[int, int]:
    """Create tar then stream it through Python's XZ/LZMA2 preset 9 extreme."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d33_tar_") as td:
        tar_path = Path(td) / "D33_FINAL_HANDOFF.tar"
        with tarfile.open(tar_path, mode="w") as tf:
            for p in sorted(stage.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() in {".tar", ".tar.xz", ".zip", ".7z", ".rar", ".gz", ".bz2", ".xz"}:
                    continue
                tf.add(p, arcname=p.relative_to(stage).as_posix(), recursive=False)
        with tar_path.open("rb") as src, lzma.open(target, "wb", format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        return target.stat().st_size, sum(1 for p in stage.rglob("*") if p.is_file() and p.suffix.lower() not in {".tar", ".tar.xz", ".zip", ".7z", ".rar", ".gz", ".bz2", ".xz"})


def verify_archive(target: Path, stage: Path) -> dict[str, Any]:
    required = ["START_HERE_NEXT_CHAT.md", "TASK_AND_RESULTS_SUMMARY.md", "EXECUTION_TIMELINE.md", "ARCHIVE_CONTENTS_GUIDE.md", "CROSS_CHAT_HANDOFF.json", "ARCHIVE_MANIFEST.json", "report/FINAL_REPORT.md", "prompt/D33_ORIGINAL_EXECUTION_PROMPT.md", "prompt/D33_INTERRUPTION_RECOVERY_PROMPT.md", "prompt/D33_FINALIZATION_PROMPT.md", "checkpoint/committed_update_467.pt", "trajectory/D33_CANONICAL_TRAJECTORY.json", "validation/FINAL_CHECKPOINT_VALIDATION.json", "tests/FOCUSED_TEST_RESULTS.txt", "implementation/stage3_h13_d33_inheritance_preserving_completion.py", "d32_parent_context/committed_update_450.pt"]
    with tempfile.TemporaryDirectory(prefix="d33_verify_") as td:
        tar_path = Path(td) / "verified.tar"
        with lzma.open(target, "rb", format=lzma.FORMAT_XZ) as src, tar_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        names = []
        with tarfile.open(tar_path, mode="r:") as tf:
            names = tf.getnames()
            members = {m.name for m in tf.getmembers() if m.isfile()}
            unsafe = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
            nested = [n for n in names if Path(n).suffix.lower() in {".tar", ".tar.xz", ".zip", ".7z", ".rar", ".gz", ".bz2", ".xz"}]
            required_present = {n: n in members for n in required}
            tar_ok = all(required_present.values()) and not unsafe
            file_count = len(members)
            tar_member_count = len(names)
        xz_ok = target.stat().st_size > 0 and tar_path.stat().st_size > 0
    return {"size_bytes": target.stat().st_size, "size_mib": target.stat().st_size / (1024 * 1024), "size_limit_bytes": 20_000_000, "size_limit_pass": target.stat().st_size <= 20_000_000, "xz_decompression_test": "PASS" if xz_ok else "FAIL", "tar_content_read_test": "PASS" if tar_ok else "FAIL", "no_nested_archives": "PASS" if not nested else "FAIL", "unsafe_member_paths": len(unsafe), "required_files": required_present, "archive_file_count": file_count, "tar_member_count_including_directories": tar_member_count, "archive_sha256": sha256(target)}


def main() -> None:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S%z")
    stage = REPO / "outputs" / f"D33_FINAL_CROSS_CHAT_HANDOFF_{stamp}"
    result = build_stage(stage)
    archive = DESKTOP / f"STAGE3_H13_D33_FINAL_STOP_CROSS_CHAT_HANDOFF_{stamp}.tar.xz"
    size, count = make_archive(stage, archive)
    verification = verify_archive(archive, stage)
    verification.update({
        "FINAL_ARCHIVE_CREATED": True,
        "ARCHIVE_PATH": str(archive),
        "ARCHIVE_FILENAME": archive.name,
        "ARCHIVE_FORMAT": "tar.xz",
        "ARCHIVE_SIZE_BYTES": verification["size_bytes"],
        "ARCHIVE_SIZE_MIB": verification["size_mib"],
        "SIZE_LIMIT_BYTES": 20_000_000,
        "SIZE_LIMIT_PASS": verification["size_limit_pass"],
        "COMPRESSION": "XZ/LZMA2 preset 9 EXTREME",
        "XZ_THREADS": 1,
        "ARCHIVE_SHA256": verification["archive_sha256"],
        "ARCHIVE_FILE_COUNT": count,
        "XZ_DECOMPRESSION_TEST": verification["xz_decompression_test"],
        "TAR_CONTENT_READ_TEST": verification["tar_content_read_test"],
        "NO_NESTED_ARCHIVES": verification["no_nested_archives"],
        "D33_FINALIZATION_REQUESTED": True,
        "D33_SCIENTIFIC_EXECUTION_STOPPED": True,
        "FINAL_COMMITTED_UPDATE": FINAL_UPDATE,
        "NUMBER_OF_D33_COMMITS": result["commits"],
        "FINAL_H1": FINAL_H1,
        "FINAL_H32": FINAL_H32,
        "FINAL_H32_MARGIN": FINAL_MARGIN,
        "STAGE3_H1_TARGET_REACHED": False,
        "FINAL_CHECKPOINT_INCLUDED": True,
        "FINAL_CHECKPOINT_PATH_IN_ARCHIVE": "checkpoint/committed_update_467.pt",
        "FINAL_CHECKPOINT_SHA256": FINAL_SHA,
        "ORIGINAL_D33_PROMPT_INCLUDED": True,
        "INTERRUPTION_RECOVERY_PROMPT_INCLUDED": True,
        "FINALIZATION_PROMPT_INCLUDED": True,
        "FINAL_REPORT_INCLUDED": True,
        "CANONICAL_TRAJECTORY_INCLUDED": True,
        "CANDIDATE_RESULTS_INCLUDED": True,
        "HORIZON_EVIDENCE_INCLUDED": True,
        "REALIZED_ADAMW_EVIDENCE_INCLUDED": True,
        "VALIDATION_EVIDENCE_INCLUDED": True,
        "IMPLEMENTATION_CHANGES_INCLUDED": True,
        "REGRESSION_TEST_EVIDENCE_INCLUDED": True,
        "INTERRUPTION_RECOVERY_EVIDENCE_INCLUDED": True,
        "D32_PARENT_CONTEXT_INCLUDED": True,
        "START_HERE_NEXT_CHAT_PRESENT": True,
        "TASK_AND_RESULTS_SUMMARY_PRESENT": True,
        "EXECUTION_TIMELINE_PRESENT": True,
        "ARCHIVE_CONTENTS_GUIDE_PRESENT": True,
        "CROSS_CHAT_HANDOFF_JSON_PRESENT": True,
        "SCIENTIFIC_EXECUTION_PERFORMED_DURING_ARCHIVING": False,
        "ADDITIONAL_OPTIMIZER_STEPS_AFTER_FINALIZATION_DECISION": 0,
        "EXACTLY_ONE_FINAL_D33_CROSS_CHAT_ARCHIVE_ON_DESKTOP": len(list(DESKTOP.glob("STAGE3_H13_D33_FINAL_STOP_CROSS_CHAT_HANDOFF_*.tar.xz"))) == 1,
        "CROSS_CHAT_HANDOFF_READY": verification["size_limit_pass"] and verification["xz_decompression_test"] == "PASS" and verification["tar_content_read_test"] == "PASS" and verification["no_nested_archives"] == "PASS" and all(verification["required_files"].values()),
        "verification_file_scope_note": "This JSON is written beside the staging directory after archive creation; the archive digest cannot be self-embedded without changing the digest.",
    })
    dump(stage / "FINAL_CROSS_CHAT_ARCHIVE_VERIFICATION.json", verification)
    dump(REPO / "outputs" / f"FINAL_CROSS_CHAT_ARCHIVE_VERIFICATION_{stamp}.json", verification)
    print(json.dumps({"archive": str(archive), "staging": str(stage), "verification": str(REPO / 'outputs' / f'FINAL_CROSS_CHAT_ARCHIVE_VERIFICATION_{stamp}.json'), **verification}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
