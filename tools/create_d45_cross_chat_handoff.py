"""Build the single D45 Stage 3 final-closure cross-chat handoff archive.

This is intentionally a release-context packager, not a repository backup.  It
copies the authoritative release, the small set of evidence needed to audit it,
the exact supplied prompt bodies, and the D45 source changes.  Large raw run
tables and unrelated dirty-worktree material are excluded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs" / "STAGE3_FINAL_RELEASE"
CHECKPOINT = (
    ROOT
    / "outputs"
    / "stage3_h13_d35_permanent_champion"
    / "checkpoints"
    / "committed_update_480.pt"
)
D41_RUN = (
    ROOT
    / "outputs"
    / "stage3_h13_d41_offline_robot_certification"
    / "run_20260827T152326Z"
)
ATTACHMENTS = [
    Path(r"C:\Users\86198\.codex\attachments\6f029c26-2848-4ec5-8b7e-48a9f10d51ef\pasted-text.txt"),
    Path(r"C:\Users\86198\.codex\attachments\9526e262-6257-45a1-b234-c906d91775db\pasted-text.txt"),
    Path(r"C:\Users\86198\.codex\attachments\88acbba2-9adf-45b1-b4f7-b4d6e6dd8507\pasted-text.txt"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip("\n") + "\n")


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_release(package_root: Path) -> None:
    destination = package_root / "final_release"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(RELEASE, destination)
    copy_file(CHECKPOINT, destination / "canonical_checkpoint" / CHECKPOINT.name)


def load_release_json(name: str) -> dict:
    return json.loads(text(RELEASE / name))


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def prompt_archive(package_root: Path, agents: str) -> None:
    sections: list[str] = [
        "# User prompts preserved for the D45 cross-chat handoff",
        "",
        "This file contains user-authored prompt material available to this run. "
        "The pasted request bodies below are copied verbatim from the supplied "
        "attachment files. System, developer, tool, and hidden reasoning text is "
        "not included.",
        "",
    ]

    for index, attachment in enumerate(ATTACHMENTS[:2], start=1):
        if not attachment.is_file():
            sections.extend(
                [
                    f"## User Prompt {index:02d} — D45 closure request attachment",
                    "",
                    f"Attachment path: `{attachment}`",
                    "",
                    "The attachment was not accessible when this archive was built; no text was fabricated.",
                    "",
                ]
            )
            continue
        body = text(attachment)
        sections.extend(
            [
                f"## User Prompt {index:02d} — D45 closure request attachment",
                "",
                f"Attachment path: `{attachment}`",
                f"Attachment bytes: `{attachment.stat().st_size}`",
                f"Attachment SHA-256: `{sha256(attachment)}`",
                "",
                "### File-paste message wrapper as accessible in the run",
                "",
                "# Files pasted by the user:",
                f"## Pasted attachment: `{attachment}`",
                "Pasted text contains the user's request.",
                "",
                "## My request:",
                "",
                "### Verbatim pasted request body",
                "",
                body.rstrip("\n"),
                "",
            ]
        )

    sections.extend(
        [
            "## User Prompt 03 — repository execution rules supplied in this run",
            "",
            "The user supplied the following repository rules. The same text is also "
            "preserved as `evidence/repository_state/AGENTS.md`.",
            "",
            "```text",
            agents.rstrip("\n"),
            "```",
            "",
        ]
    )

    attachment = ATTACHMENTS[2]
    if attachment.is_file():
        body = text(attachment)
        sections.extend(
            [
                "## User Prompt 04 — final closure archive request attachment",
                "",
                f"Attachment path: `{attachment}`",
                f"Attachment bytes: `{attachment.stat().st_size}`",
                f"Attachment SHA-256: `{sha256(attachment)}`",
                "",
                "### File-paste message wrapper as accessible in the run",
                "",
                "# Files pasted by the user:",
                f"## Pasted attachment: `{attachment}`",
                "Pasted text contains the user's request.",
                "",
                "## My request:",
                "",
                "### Verbatim pasted request body",
                "",
                body.rstrip("\n"),
                "",
            ]
        )
    else:
        sections.extend(
            [
                "## User Prompt 04 — final closure archive request attachment",
                "",
                f"Attachment path: `{attachment}`",
                "",
                "The attachment was not accessible when this archive was built; no text was fabricated.",
                "",
            ]
        )

    write_text(package_root / "USER_PROMPTS.md", "\n".join(sections))


def write_documents(package_root: Path) -> None:
    baseline = load_release_json("PRE_D45_STAGE3_GOLDEN_BASELINE.json")
    metrics = load_release_json("canonical_metrics_h1_h32.json")
    model = load_release_json("authoritative_model_definition.json")
    robot = load_release_json("robot_validation_status.json")
    adversarial = load_release_json("adversarial_checks.json")
    manifest = load_release_json("release_manifest.json")

    checkpoint_hash = metrics["checkpoint_sha256"]
    h1 = metrics["H1"]
    h32 = metrics["H32"]
    h32_limit = metrics["H32_limit"]
    h32_margin = metrics["H32_margin"]

    write_text(
        package_root / "START_HERE_D45_HANDOFF.md",
        f"""# Start here: D45 Stage 3 final closure handoff

## Verdict

`TASK_STATUS: PASS`

`STAGE_3_FINAL_CLOSURE: PASS`

`STAGE_3_STATUS: FROZEN_AND_CLOSED`

`STAGE4_INPUT_READY: YES`

D45 sealed Stage 3 as one deterministic, regression-safe offline release. The
purpose of closure was to preserve the validated H32 acceptance floor and the
robot-motion chain instead of continuing H1 micro-optimization after the H1
stretch target remained non-blocking and D44 produced no canonical promotion.
The operating doctrine was **aggressive execution, conservative acceptance**:
explore shadows freely, but mutate production state only when every protected
metric, identity, and robot gate survives.

## What a new ChatGPT should read

1. `USER_PROMPTS.md` — supplied user request bodies and repository rules.
2. `D45_FINAL_CLOSURE_REPORT.md` — the concise closure decision and gate table.
3. `FINAL_CANONICAL_STATE.md` and `H1_H32_METRIC_VERIFICATION.md` — the exact
   production state and metric evidence.
4. `END_TO_END_REPLAY_REPORT.md` and `FINAL_ROBOT_MOTION_VALIDATION.md` — the
   executed replay chain and robot evidence.
5. `D45_DEFECT_REPAIR_LEDGER.md` and
   `D45_REGRESSION_AND_BASELINE_REPORT.md` — what was repaired and how it was
   shown non-regressive.
6. `STAGE3_HISTORICAL_ACCOUNTING.md`,
   `STAGE3_OPEN_ISSUES_AND_TECHNICAL_DEBT.md`, and `STAGE4_INPUT_CONTRACT.md`.
7. `final_release/` and `changed_files/` for machine-readable release files and
   the exact D45 source changes.

## Authoritative state at closure

- Canonical checkpoint: `{manifest['canonical_checkpoint']}`
- SHA-256: `{checkpoint_hash}`
- Completed optimizer update: `{manifest['canonical_update']}`
- H1: `{h1}`
- H32: `{h32}`
- H32 limit: `{h32_limit}`
- H32 margin: `{h32_margin}`
- Authoritative input: `{manifest['authoritative_input_root']}`; 181-point ON-state open-arch pair
- Release entrypoint: `{manifest['authoritative_entrypoint']}`
- Robot status: `PASS`; D41 strict replay: `PASS`; robustness: `37/37 PASS`
- Collision label: `adaptive_discrete_interpolation`; CCD: `not_available`; clearance: JSON `null`

The checkpoint is included at `final_release/canonical_checkpoint/{CHECKPOINT.name}`
for independent hash and loader verification. The release is CPU-authenticated;
cross-platform bitwise recreation is not claimed.

## Executed chain

The closure chain was: authenticated H10/H11 TRAIN preprocessing → D42
evaluation using the D17 compact causal evaluator → retained D39 stable prior →
D40 locked `routeA_DLS_position_dominant_v4` → D41 native offline MoveIt2 replay
with PlanningScene, FK, dynamics, Ruckig, and post-Ruckig collision checks →
D45 deterministic replay, adversarial checks, and release sealing.

D36 was rerun and its previously missing persisted summary was added at the
existing reporting boundary. The fresh D41 run was
`{robot['D41']['run_directory']}`.

## Why Stage 3 is closed

H32 is under its protected limit with positive margin; the canonical identity is
unchanged; D44 accepted zero candidates; and the complete offline robot chain
passed. H1 remains above its `5e-05` stretch target, but that target was
explicitly deferred as non-blocking. Continuing to optimize H1 would risk the
validated H32 floor and the robot baseline without a current acceptance case.

## Three accepted D45 repairs

1. Persist the successful D36 evaluation summary (`D36_SUMMARY.json`).
2. Exclude D41's explicitly named nominal row from the 37-case robustness
   denominator; the fresh run remains 37/37 PASS.
3. Add the fail-closed `scripts/run_stage3_final_release.py` closure entrypoint
   and its release gate implementation.

All three were reporting/release-completeness repairs. They did not change the
canonical model, optimizer state, H1/H32 values, input graph, or robot result.

## Next rational step

Begin Stage 4 from `outputs/STAGE3_FINAL_RELEASE` only. Inherit the update-480
checkpoint, authenticated preprocessing, the locked D40 route, and D41 native
offline evidence. Stage 4 should address system-level motion stability,
smoothness, accuracy, reproducibility, reliability, and robustness; it should
not silently promote D43/D44 shadows, legacy 720-point material, or the deferred
H1 stretch objective.
""",
    )

    write_text(
        package_root / "D45_FINAL_CLOSURE_REPORT.md",
        text(RELEASE / "STAGE3_CLOSURE_REPORT.md"),
    )

    write_text(
        package_root / "STAGE3_FINAL_RELEASE.md",
        f"""# Stage 3 final release definition

## Release identity

- Name: `{manifest['release_name']}`
- Status: `{manifest['release_status']}`
- Schema: `{manifest['schema_version']}`
- Authoritative entrypoint: `{manifest['authoritative_entrypoint']}`
- Standard replay: `{manifest['standard_command']}`
- Full fresh robot replay: `{manifest['full_fresh_robot_command']}`

## Production state

The only Stage 3 production checkpoint is `{manifest['canonical_checkpoint']}`
with SHA-256 `{manifest['canonical_checkpoint_sha256']}` at update
`{manifest['canonical_update']}`. It is the D35 permanent champion and is
unchanged by D43, D44, or D45.

The release evaluates the 181-point ON-state open-arch pair under
`{manifest['authoritative_input_root']}` using authenticated H10/H11 TRAIN
statistics. Its metric path is `{manifest['evaluation_path']}` and its robot path
is `{manifest['robot_motion_path']}`.

## Explicit exclusions and semantics

OFF states, reorientation, approach, retreat, GNN, PPO, LSTM, Transformer, and
closed-contour transitions are outside this release. Collision results are
labelled `{manifest['collision_method']}` and are not strict continuous
collision detection. Bullet CCD is `{manifest['ccd']}` and clearance is JSON
`null`; no clearance is inferred from a collision-free state.

D43/D44 outputs remain research-only shadows. The legacy 720-point outputs and
legacy spray-off/reorientation reports are excluded. Stage 4 was not run.

The complete machine-readable manifest is preserved as
`final_release/release_manifest.json`; the source release directory is also
preserved under `final_release/`.
""",
    )

    write_text(
        package_root / "FINAL_CANONICAL_STATE.md",
        f"""# Final canonical Stage 3 state

## Checkpoint identity

- Path: `{manifest['canonical_checkpoint']}`
- Archive copy: `final_release/canonical_checkpoint/{CHECKPOINT.name}`
- SHA-256: `{checkpoint_hash}`
- Completed optimizer update: `{manifest['canonical_update']}`
- Model class: `{model['model_class']}`
- Trainable parameters: `{model['trainable_parameter_count']}`
- Optimizer state: protected and retained
- Source revision recorded by the release: `{model['runtime']['source_revision']}`
- Source revision match: `{model['runtime']['source_revision_match']}`

## Input identity

The authoritative pair is the 181-point ON-state open-arch input under
`outputs/internal_wiper_moveit_inputs/`. The protected baseline records these
SHA-256 values:

| Input | SHA-256 |
|---|---|
| `open_arch_tcp_poses_base_link.csv` | `{baseline['authoritative_inputs']['input_files'][0]['sha256']}` |
| `open_arch_seed_joints.csv` | `{baseline['authoritative_inputs']['input_files'][1]['sha256']}` |

Both files are copied under `evidence/authoritative_inputs/`.

## Canonical metrics and robot baseline

- H1: `{h1}`
- H32: `{h32}`
- H32 limit: `{h32_limit}`
- H32 margin: `{h32_margin}`
- D44 canonical promotions: `0`
- D44 canonical mutation: `NO`
- Retained robot baseline: D39 stable prior → D40
  `routeA_DLS_position_dominant_v4` → D41 native offline certification
- Robot certification: `PASS`

The final release intentionally reports `ccd: not_available` and
`clearance: null` under the repository collision contract. The raw D41 run is
not reinterpreted here.
""",
    )

    write_text(
        package_root / "END_TO_END_REPLAY_REPORT.md",
        f"""# D45 end-to-end replay report

## Result

`END_TO_END_REPLAY_STATUS: PASS`

`CLEAN_REPLAY_STATUS: PASS`

The final release gate authenticated the 181-point input pair, loaded the
update-480 checkpoint, ran the H1–H32 evaluator twice, compared the complete
metric payload and endpoints deterministically, validated the locked D40
robot baseline, validated the fresh D41 strict replay, ran adversarial checks,
and wrote the frozen release manifest.

## Executed components

1. Preflight used authenticated H10/H11 TRAIN statistics; future-reference
   joint rows read: `0`.
2. Metric evaluation used D42's evaluator and the D17 compact causal evaluator.
3. The D36 system evaluation was rerun and completed with `5` checkpoints,
   `230400` ledger episodes, `46080` scenario episodes, `225` summaries, and
   `180` paired comparisons. Its persisted result is
   `evidence/replay/D36_SUMMARY.json`.
4. The fresh D41 native run was
   `{robot['D41']['run_directory']}` and its strict replay directory is
   archived selectively under `evidence/replay/d41_strict_replay/`.
5. Release-level adversarial checks all passed: canonical identity unchanged,
   corrupted checkpoint rejection, input shape/finiteness, invalid tensor
   shape rejection, NaN/Inf policy, and shadow non-promotion.

## Reproduction commands

From the repository root:

```text
python scripts/run_stage3_final_release.py --replay
python scripts/run_stage3_final_release.py --replay --run-robot
```

The second command is an offline native replay and does not imply hardware
deployment. The archive contains the exact release code, checkpoint, input
pair, and compact evidence needed to inspect the result.

## Evidence pointers

- `final_release/release_manifest.json`
- `final_release/canonical_metrics_h1_h32.json`
- `final_release/robot_validation_status.json`
- `final_release/adversarial_checks.json`
- `evidence/replay/D36_SUMMARY.json`
- `evidence/replay/d41_strict_replay/`
- `evidence/replay/logs/`
""",
    )

    write_text(
        package_root / "D45_DEFECT_REPAIR_LEDGER.md",
        """# D45 defect and repair ledger

All three discovered material defects were release-completeness/reporting
defects. None changed the canonical model, optimizer behavior, input graph,
robot trajectory, or acceptance values.

| ID | Symptom | Root cause | Scope | Repair | Non-regression evidence | Final status |
|---|---|---|---|---|---|---|
| D45-R1 | D36 returned PASS but no persisted summary existed | The runner stopped after successful evaluation without writing the versioned summary at the existing reporting boundary | D36 reporting only | Persist `D36_SUMMARY.json` | D36 rerun: 5 checkpoints, 230400 ledger episodes, 46080 scenario episodes, 225 summaries, 180 paired comparisons; focused tests green | Accepted; no metric/model change |
| D45-R2 | D45 robot gate counted the nominal D41 control in the robustness denominator | Native robustness output contains one explicit `nominal_d40` row plus 37 robustness cases | D45 release gate only | Filter the explicitly named nominal row before applying the 37/37 requirement | Fresh native run remains 37/37 PASS; no D41 artifact changed | Accepted; no robot-result change |
| D45-R3 | No single executable closure entrypoint covered the final chain | Evidence was distributed across D35/D40/D41/D44 | Release orchestration only | Add fail-closed `tools/stage3_d45_finalize_release.py` and `scripts/run_stage3_final_release.py` | Canonical SHA/H1/H32 unchanged; repeated replay deterministic; D41 strict replay PASS; adversarial checks PASS | Accepted; final release sealed |

`NUMBER_OF_MATERIAL_DEFECTS_FOUND: 3`

`NUMBER_OF_MATERIAL_DEFECTS_FIXED: 3`

`UNRESOLVED_CORRECTNESS_BLOCKERS: NONE`
""",
    )

    write_text(
        package_root / "D45_REGRESSION_AND_BASELINE_REPORT.md",
        f"""# D45 regression and protected-baseline report

## Protected rolling baseline

The read-only pre-D45 baseline is copied at
`final_release/PRE_D45_STAGE3_GOLDEN_BASELINE.json`. It protects the update-480
checkpoint identity, the 181-point input pair, the collision contract, D44's
zero canonical promotions, and the retained D39/D40/D41 robot baselines.

## Accepted sequence

| Stage | Accepted state | Mutation to production checkpoint |
|---|---|---|
| Pre-D45 | D35 update 480, H1 `{h1}`, H32 `{h32}`, protected input hashes | None |
| D36 repair | Persisted successful system-evaluation summary | None |
| D45 gate repair | Correct 37-case denominator by excluding named nominal row | None |
| Final release | Fail-closed replay wrapper and manifest | None |

## Regression evidence

- Canonical checkpoint SHA-256 stayed `{checkpoint_hash}`.
- Canonical update stayed `480`.
- H1 stayed `{h1}`.
- H32 stayed `{h32}` and remained below `{h32_limit}`.
- D44 canonical promotions stayed `0`; canonical mutation stayed `NO`.
- D40 remained locked and D41 remained `PASS`.
- Fresh D41 strict replay remained `PASS` with `37/37 PASS` robustness.
- Adversarial checks passed, including corrupted-checkpoint rejection and
  shadow non-promotion.
- Focused Stage 3 suite: `41 passed in 14.53s`.
- Final D45/D36 gate suite: `6 passed in 9.65s`.
- Python compilation and `git diff --check` completed successfully; the only
  diff-check output was pre-existing line-ending warnings.

`REGRESSIONS_INTRODUCED_BY_ACCEPTED_REPAIRS: 0`.

The surrounding worktree was already dirty. Unrelated user changes were
preserved and are intentionally excluded from this context archive.
""",
    )

    write_text(
        package_root / "H1_H32_METRIC_VERIFICATION.md",
        f"""# H1–H32 metric verification

## Endpoint acceptance

| Metric | Final value | Gate | Result |
|---|---:|---:|---|
| H1 | `{h1}` | stretch target `5e-05` | Unmet, explicitly deferred non-blocking |
| H32 | `{h32}` | limit `{h32_limit}` | PASS; margin `{h32_margin}` |

H1–H32 are validation-window mean endpoint joint-position errors by horizon.
The evaluator is `{metrics['evaluator']}`; the split is `{metrics['evaluation_split']}`;
free-running evaluation is `{metrics['free_running']}`; future-reference joint
rows read is `{metrics['future_reference_joint_rows_read']}`.

## Determinism

The final release ran the metric replay twice and required exact equality of the
repeated metric payload and endpoints. `deterministic_replay: PASS`.

The full horizon vector, schema, checkpoint identity, and aggregation definition
are preserved in `final_release/canonical_metrics_h1_h32.json`.

## Closure interpretation

H32 is the protected acceptance floor and is satisfied with a positive margin.
H1's `5e-05` value is a stretch objective, not a Stage 3 blocker. D44 explored
response-surface candidates but accepted zero promotions; therefore the
production checkpoint remains update 480. H1 optimization is deferred to Stage
4 under an explicit non-regression contract.
""",
    )

    write_text(
        package_root / "FINAL_ROBOT_MOTION_VALIDATION.md",
        f"""# Final robot-motion validation

`FINAL_ROBOT_CERTIFICATION_STATUS: PASS`

## Retained production path

`D39 retained stable prior → D40 routeA_DLS_position_dominant_v4 → D41 native offline replay`

D40 is locked and D41's fresh run is
`{robot['D41']['run_directory']}`. The strict replay executed MoveIt2
PlanningScene, FK, Ruckig, dynamics, and post-Ruckig collision checks. The
release also required runtime evidence containing the loaded robot model and
PlanningScene monitor markers.

## Gate evidence

- D40 retention: `PASS`; D40 quality status: `pass`.
- D41 strict replay: `PASS`.
- Native robustness: `37/37 PASS` after excluding the explicitly named nominal
  control from the robustness denominator.
- PlanningScene: `executed`.
- FK: `executed`.
- Dynamics: `executed`.
- Ruckig: `executed`.
- Post-Ruckig collision: `executed`.
- Point count: `181`.
- Future joint reads: `0`.
- Collision label: `adaptive_discrete_interpolation`.
- CCD: `not_available`.
- Clearance in the release contract: JSON `null`.

The concise release validation record is
`final_release/robot_validation_status.json`. Selected strict replay outputs,
native logs, and the non-nominal case summary are under
`evidence/replay/`. The archive does not promote candidate-specific D43/D44
robot validity.

This is an offline certification result. It is not a physical hardware
deployment; the existing virtual-design tool-TCP metadata must be replaced by
measured hardware calibration before deployment.
""",
    )

    write_text(
        package_root / "STAGE3_HISTORICAL_ACCOUNTING.md",
        """# Stage 3 historical accounting

## Proven and retained

- D35 permanent champion, update-480 canonical checkpoint.
- D39 stable prior as the retained robot input.
- D40 locked `routeA_DLS_position_dominant_v4` task-space baseline.
- D41 native offline MoveIt2/PlanningScene/FK/Ruckig/dynamics/post-Ruckig
  certification and fresh strict replay.

## Superseded but not erased

- D37 and D38 incomplete robot routes.
- D39 pre-D40 partial system status.

These remain historical evidence in the repository but are superseded as the
production route by the locked D40/D41 path.

## Research-only assets

D43/D44 response surfaces, shadow candidates, unsafe candidates, and
non-promoted ledgers are research assets. D44 canonical promotions were `0` and
the canonical checkpoint was not mutated.

## Invalidated conclusions

`HISTORICAL_RESULTS_INVALIDATED: NONE`.

## Excluded material

The legacy 720-point outputs and legacy spray-off/reorientation reports are not
part of the Stage 0/1 graph. Stage 3 covers only the 181-point ON-state
open-arch pair. OFF states, reorientation, GNN, PPO, LSTM, Transformer,
approach, retreat, and closed-contour transitions are outside scope.

## Deferred

Stage 4 inherits system-level stability, smoothness, accuracy, reproducibility,
reliability, and robustness work. The H1 stretch target is deferred and is not
a Stage 3 correctness blocker.
""",
    )

    write_text(
        package_root / "STAGE3_OPEN_ISSUES_AND_TECHNICAL_DEBT.md",
        text(RELEASE / "OPEN_TECHNICAL_DEBT.md")
        + "\n## Closure status\n\n"
        + "`BLOCKING_ISSUES: NONE`\n\n"
        + "The open items are non-blocking for the frozen Stage 3 release. Stage 4 "
        + "must explicitly plan them rather than treating the release as a hardware "
        + "deployment or silently substituting historical shadows.\n",
    )

    write_text(
        package_root / "STAGE4_INPUT_CONTRACT.md",
        text(RELEASE / "STAGE4_INPUT_CONTRACT.md")
        + "\n## Protected floor inherited by Stage 4\n\n"
        + f"- Canonical checkpoint hash: `{checkpoint_hash}`\n"
        + f"- H1: `{h1}`; H32: `{h32}`; H32 limit: `{h32_limit}`\n"
        + "- Production robot baseline: D40 locked and D41-certified offline\n"
        + "- Any candidate must preserve the 181-point scope, authenticated preprocessing, "
        + "the collision label, and the explicit `not_available`/`null` CCD-clearance contract.\n",
    )

    source_files = [
        ROOT / "AGENTS.md",
        ROOT / "tools" / "stage3_d45_finalize_release.py",
        ROOT / "scripts" / "run_stage3_final_release.py",
        ROOT / "tools" / "stage3_h13_d36_system_evaluation.py",
        ROOT / "tests" / "test_stage3_d45_release.py",
    ]
    for source in source_files:
        relative = source.relative_to(ROOT)
        copy_file(source, package_root / "changed_files" / relative)

    source_hashes = {
        str(source.relative_to(ROOT)).replace("\\", "/"): sha256(source)
        for source in source_files
    }
    write_text(
        package_root / "CODE_CHANGE_SUMMARY.md",
        """# D45 code-change summary

The archive includes the exact source for the D45 release gate, its executable
wrapper, the D36 summary persistence repair, the focused D45 tests, and the
repository rules that constrained the work. These are the only D45 source files
copied into `changed_files/`; unrelated dirty-worktree files are not presented
as D45 changes.

| File | Role |
|---|---|
| `tools/stage3_d45_finalize_release.py` | Fail-closed D45 release verification, deterministic replay, robot gate, adversarial checks, and manifest/report generation |
| `scripts/run_stage3_final_release.py` | Single executable replay/release entrypoint |
| `tools/stage3_h13_d36_system_evaluation.py` | D36 successful-run summary persistence repair |
| `tests/test_stage3_d45_release.py` | D45 release-gate regression tests |
| `AGENTS.md` | User-supplied repository execution contract preserved for handoff |

All listed file contents are copied verbatim under `changed_files/`. The
repository was not reset, cleaned, or committed during packaging.
"""
        + "\n## SHA-256 of copied source files\n\n"
        + "```json\n"
        + json.dumps(source_hashes, indent=2, sort_keys=True)
        + "\n```\n",
    )

    write_text(
        package_root / "evidence" / "repository_state" / "repository_state.md",
        """# Repository state at D45 archive creation

The worktree was already materially dirty before this archive task. Existing
user changes were preserved. No reset, clean, checkout, restore, or broad
overwrite operation was performed. No commit was created for this archive.

The archive includes only the focused D45 source paths under `changed_files/`
and a compact status/stat record. It does not claim the entire worktree is clean.
""",
    )

    write_text(
        package_root / "evidence" / "repository_state" / "git_status_short.txt",
        """D45 relevant paths were untracked/new in the pre-existing dirty worktree:
?? tools/stage3_d45_finalize_release.py
?? scripts/run_stage3_final_release.py
?? tests/test_stage3_d45_release.py
?? tools/stage3_h13_d36_system_evaluation.py (pre-existing untracked path, D45-modified)
?? AGENTS.md (user-supplied repository contract; preserved)

The complete unrelated status was intentionally not copied into this archive.
""",
    )
    write_text(
        package_root / "evidence" / "repository_state" / "git_diff_stat.txt",
        """Relevant D45 files did not have tracked-index diffs because they were
new/untracked in the pre-existing worktree. Full source copies are in
`changed_files/`. The pre-existing tracked worktree had 32 modified files and
the repository had hundreds of unrelated untracked artifacts; none were reset
or cleaned.
""",
    )
    copy_file(ROOT / "AGENTS.md", package_root / "evidence" / "repository_state" / "AGENTS.md")

    # Authoritative input pair is small and makes the archive independently useful.
    input_root = ROOT / "outputs" / "internal_wiper_moveit_inputs"
    for name in ("open_arch_tcp_poses_base_link.csv", "open_arch_seed_joints.csv"):
        copy_file(input_root / name, package_root / "evidence" / "authoritative_inputs" / name)

    d36 = ROOT / "outputs" / "stage3_h13_d36_system_evaluation" / "D36_SUMMARY.json"
    copy_file(d36, package_root / "evidence" / "replay" / "D36_SUMMARY.json")

    # Compact, directly interpretable D41 evidence. Do not copy raw multi-MB
    # clearance/jacobian/diagnostic tables or the full robustness trajectory set.
    compact_d41 = [
        "D41_AUTHORITATIVE_IDENTITY.json",
        "inputs/nominal_d40.csv",
        "native/robustness/cases.csv",
        "native/robustness/D41_native_case_summary.jsonl",
        "native/robustness/D41_native_provenance.json",
    ]
    for relative in compact_d41:
        copy_file(D41_RUN / relative, package_root / "evidence" / "replay" / "d41_native" / relative)

    logs = [
        "dls_batch.log",
        "native_build.log",
        "native_candidate.log",
        "native_nominal.log",
        "native_robustness.log",
        "strict_replay.log",
    ]
    for name in logs:
        copy_file(D41_RUN / "logs" / name, package_root / "evidence" / "replay" / "logs" / name)

    strict_files = [
        "audit_goal_requirements_strict.json",
        "final_acceptance_summary.json",
        "final_quality_report.csv",
        "ik_continuity_segments_summary.json",
        "moveit_collision_report.csv",
        "moveit_fk_tcp_trace.csv",
        "moveit_joint_dynamics_report.csv",
        "moveit_quality_report.csv",
        "moveit_runtime.log",
        "moveit_waypoint_joint_trajectory.csv",
        "production_readiness_check.json",
        "segmented_process_summary.json",
    ]
    for name in strict_files:
        copy_file(D41_RUN / "strict_replay" / name, package_root / "evidence" / "replay" / "d41_strict_replay" / name)

    write_text(
        package_root / "evidence" / "tests" / "D45_TEST_RESULTS.txt",
        """Command:
python -m pytest -q tests/test_stage3_d45_release.py tests/test_stage3_h13_d35_champion_manager.py tests/test_stage3_h13_d36_system_evaluation.py tests/test_stage3_h13_d37_robot_smoke.py tests/test_stage3_h13_d38.py tests/test_stage3_h13_d39.py tests/test_stage3_h13_d40.py tests/test_stage3_h13_d41.py tests/test_stage3_h13_d43_constrained_breakthrough_campaign.py tests/test_stage3_h13_d44_realized_response_campaign.py

Result: 41 passed in 14.53s

Command:
python -m pytest -q tests/test_stage3_d45_release.py tests/test_stage3_h13_d36_system_evaluation.py

Result: 6 passed in 9.65s

Additional checks:
- py_compile passed for the D45 wrapper, D45 gate, and D36 runner.
- git diff --check completed; only pre-existing line-ending warnings were emitted.
""",
    )
    write_text(
        package_root / "evidence" / "tests" / "D36_TEST_RESULTS.txt",
        """Command: python tools/stage3_h13_d36_system_evaluation.py

Two completed D36 runs returned PASS. The final persisted summary reports:
5 checkpoints; 230400 ledger episodes; 46080 scenario episodes; 225 summaries;
180 paired comparisons; validation/test/generalization splits; nominal and
sensor-noise conditions; horizons 1, 4, 8, 16, 32.
""",
    )

    write_text(
        package_root / "evidence" / "failures" / "D45_REPAIRED_RELEASE_GAPS.md",
        """# Repaired release gaps

The three accepted gaps are fully described in `D45_DEFECT_REPAIR_LEDGER.md`.
In brief: D36 did not persist its successful summary; D45 counted D41's named
nominal control in the robustness denominator; and no single executable closure
entrypoint existed. All were repaired without changing the canonical state.
No correctness blocker remains.
""",
    )
    write_text(
        package_root / "evidence" / "archive_validation_expectations.md",
        """# Archive validation expectations

The final archive must be a readable `tar.xz`, extract cleanly, contain exactly
one top-level handoff directory, and include `START_HERE_D45_HANDOFF.md`,
`USER_PROMPTS.md`, `final_release/release_manifest.json`, the canonical
checkpoint copy, the authoritative input pair, the focused source changes, and
the compact replay evidence. The completed validation result and archive
SHA-256 are reported by the creating Codex turn.
""",
    )

    write_json(
        package_root / "evidence" / "release_snapshot.json",
        {
            "release_manifest": manifest,
            "canonical_metrics": metrics,
            "robot_validation_status": robot,
            "adversarial_checks": adversarial,
            "archive_scope": "D45 final closure context; no unrelated worktree backup",
        },
    )


def scan_for_secrets(package_root: Path) -> list[str]:
    patterns = [
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret|cookie)\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-/.+=]{8,}"),
        re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+\S+"),
    ]
    findings: list[str] = []
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                findings.append(f"{path.relative_to(package_root)}:{line_number}")
    return findings


def write_hash_manifest(package_root: Path) -> None:
    rows: list[str] = []
    for path in sorted(p for p in package_root.rglob("*") if p.is_file()):
        if path.name == "file_hashes.sha256":
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(package_root).as_posix()}")
    write_text(package_root / "evidence" / "file_hashes.sha256", "\n".join(rows))


def build_archive(package_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    # Python's tarfile uses liblzma preset 9, the portable equivalent of the
    # requested tar + xz -9e pipeline on this Windows host.
    with tarfile.open(archive_path, mode="w:xz", preset=9) as archive:
        archive.add(package_root, arcname=package_root.name, recursive=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desktop", type=Path, required=True)
    args = parser.parse_args()

    if not RELEASE.is_dir():
        raise FileNotFoundError(RELEASE)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if any(not path.is_file() for path in ATTACHMENTS):
        missing = [str(path) for path in ATTACHMENTS if not path.is_file()]
        raise FileNotFoundError("Missing supplied attachment(s): " + ", ".join(missing))

    existing = sorted(args.desktop.glob("D45_STAGE3_FINAL_CLOSURE_CONTEXT_HANDOFF_*.tar.xz"))
    if existing:
        raise RuntimeError(
            "Desktop already contains a D45 handoff archive; refusing to create a competing archive: "
            + ", ".join(str(path) for path in existing)
        )

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    package_name = f"D45_STAGE3_FINAL_CLOSURE_CONTEXT_HANDOFF_{stamp}"
    staging = ROOT / "tmp" / package_name
    if staging.exists():
        raise RuntimeError(f"Refusing to overwrite existing staging directory: {staging}")
    package_root = staging / package_name
    package_root.mkdir(parents=True)

    prompt_archive(package_root, text(ROOT / "AGENTS.md"))
    copy_release(package_root)
    write_documents(package_root)

    findings = scan_for_secrets(package_root)
    if findings:
        raise RuntimeError("Potential secret-like content found: " + ", ".join(findings))
    write_hash_manifest(package_root)

    archive_path = args.desktop / f"{package_name}.tar.xz"
    build_archive(package_root, archive_path)
    print(json.dumps({
        "archive": str(archive_path),
        "staging": str(staging),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "top_level": package_name,
        "checkpoint_sha256": sha256(CHECKPOINT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
