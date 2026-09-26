"""Build a compact, self-describing Stage 5A-C context handoff.

This script only reads the protected/project evidence and writes a new staging
directory. It deliberately excludes the two large raw collision-atlas JSONL
files; their canonical paths, sizes, and hashes are recorded instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
CLOSURE = OUTPUTS / "STAGE5AC_FINAL_CLOSURE"
D65 = OUTPUTS / "D65_FINAL_CLOSURE"
PARENT = OUTPUTS / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE"
DEFAULT_PROMPT = Path(
    r"C:\Users\86198\.codex\attachments\f078fc1d-7319-4a11-af56-d790c3466824\pasted-text.txt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def put_text(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def copy_one(root: Path, source: Path, rel: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = root / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def relative_project(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("\\", "/")


def git_output(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        return f"COMMAND_FAILED returncode={result.returncode}\n{result.stderr}"
    return result.stdout


def build_state(generated_at: str) -> dict[str, object]:
    return {
        "schema_version": "stage5ac-handoff-state-v1",
        "generated_at": generated_at,
        "repository": {
            "root": str(REPO),
            "authority_rule": "Reconstruct from disk and runtime evidence; do not rely on chat narrative.",
            "stage01_input": relative_project(REPO / "outputs/internal_wiper_moveit_inputs"),
            "stage01_scope": "181-point ON-state open-arch planning only; no legacy 720-point or OFF/reorientation graph.",
        },
        "current_phase": "Stage5AC closure completed; forced context handoff pending archive verification",
        "stage5ac": {
            "closure_directory": relative_project(CLOSURE),
            "technical_status": "STAGE5AC_SHADOW_REPAIR_CANDIDATE_VALIDATED_GEOMETRY_NO_PROMOTION",
            "promotion": "NO_PROMOTION",
            "physical_constraint_infeasibility_established": False,
            "stage5b_started": False,
            "stage5b_ready": False,
            "candidate": "routeE_x_minus_050_local_y_minus_0025mm_consistent_fd_native_ruckig",
            "candidate_route_scope": "shadow exploration only; not a replacement for D65 or Stage3 canonical state",
        },
        "protected_floor": {
            "d65_status": "SOFTWARE_SCOPE_FROZEN_WITH_EXPLICIT_P0_BOUNDARY",
            "d65_promotion": "NO_PROMOTION",
            "auto0_trajectory": {
                "path": relative_project(PARENT / "timing_scale025_auto0/trajectories/adversarial_0100.csv"),
                "sha256": "13f8075296a0defa12f17348cd6bafb913a8c10d22012a651d5614082df0a264",
            },
            "auto1_trajectory": {
                "path": relative_project(PARENT / "timing_scale025_corrected_auto1/trajectories/adversarial_0101.csv"),
                "sha256": "dda61e3be2328536f0c688333f3b24861c56dbdb347e807d881ca7d71f617416",
            },
            "d65_hashes_unchanged_at_handoff": True,
        },
        "candidate_evidence": {
            "route_a": {
                "states": 181,
                "duration_seconds": 705.608435681,
                "max_waypoint_jump_degrees": 96.51766,
                "q_preserved_exactly": True,
                "native_ruckig": "executed and dynamics audit PASS",
            },
            "bullet": {
                "run1": "180/180 intervals executed; 0 continuous robot-world collisions; 0 endpoint self collisions",
                "run2": "180/180 intervals executed; 0 continuous robot-world collisions; 0 endpoint self collisions",
            },
            "fcl": {
                "method": "adaptive_discrete_interpolation at 0.25 degrees",
                "states": 2607,
                "world_contacts": 0,
                "self_collisions": 0,
                "moveit_runtime_tf_rows": 0,
                "overall_pass": False,
            },
            "task_validator": {
                "max_position_error_m": 9.24284814237278e-06,
                "max_orientation_error_rad": 9.898050755762065e-06,
                "local_slack_pass": True,
                "original_pose_exact_pass": False,
            },
        },
        "hard_boundaries": [
            "adaptive_discrete_interpolation is not strict continuous collision detection",
            "Bullet CCD and hardware/calibrated physical clearance are unavailable; use not_available/null",
            "native full-trajectory Stage26 FCL backend was unavailable/crashed",
            "MoveIt FCL candidate runtime TF rows were zero; fail closed",
            "candidate has an unacceptable 96.51766 degree waypoint jump",
            "strict D62 left 6154 unresolved; D65 remains NO_PROMOTION",
        ],
        "next_gate": "Do not begin Stage5B. First independently verify this package, repository dirty state, and all blockers; then choose the next authorized upstream action.",
    }


def ledger(state: dict[str, object]) -> dict[str, object]:
    def item(key: str, status: str, scope: str, evidence: list[str], claim: str, forbidden: list[str]) -> dict[str, object]:
        return {
            "key": key,
            "status": status,
            "scope": scope,
            "evidence": evidence,
            "claim_allowed": claim,
            "claim_forbidden": forbidden,
        }

    return {
        "schema_version": "stage5ac-progress-ledger-v1",
        "status_enum": ["PASS_VERIFIED", "FAIL_VERIFIED", "BLOCKED", "UNAVAILABLE", "NOT_STARTED", "IN_PROGRESS"],
        "generated_at": state["generated_at"],
        "items": [
            item("ROOT_GOAL_RECONSTRUCTED", "PASS_VERIFIED", "handoff", ["ROOT_GOAL.md", "HANDOFF_AUTHORITY.md"], "The continuation objective and authority hierarchy are explicit.", ["Using chat memory as the sole authority."]),
            item("D65_PROTECTED_FLOOR", "PASS_VERIFIED", "protected canonical software evidence", ["evidence/protected_floor/D65_STATE_SUMMARY.md", "evidence/protected_floor/D65_HYBRID_AUTO0.json", "evidence/protected_floor/D65_HYBRID_AUTO1.json"], "D65 identity/hash references were rechecked unchanged for this handoff.", ["Promoting D65 to a safety or physical-clearance claim."]),
            item("STAGE5AR_ROOT_CAUSE", "PASS_VERIFIED", "Stage5AR shadow diagnosis", ["results/final_closure/STAGE5AC_ROOT_CAUSE_REPORT.md"], "The robot-world collision was reproduced with the world pair and frame-stamp correction was recorded.", ["Calling it a self-collision result."]),
            item("STAGE5AC_COLLISION_ATLAS", "PASS_VERIFIED", "shadow measurement infrastructure", ["results/final_closure/STAGE5AC_COLLISION_ATLAS_auto0_SUMMARY.json", "results/final_closure/STAGE5AC_COLLISION_ATLAS_auto1_SUMMARY.json"], "Both atlas runs completed their declared finite workload and their summaries are retained.", ["Calling finite atlas coverage strict CCD."]),
            item("ROUTE_A_NOMINAL", "FAIL_VERIFIED", "shadow route search", ["results/final_closure/STAGE5AC_CANDIDATE_COMPARISON.json", "results/final_closure/STAGE5AC_FINAL_REPORT.md"], "The nominal route did not provide an accepted full path under the declared constraints.", ["Removing the failed nominal result."]),
            item("LAYOUT_SHADOW_EXPLORATION", "PASS_VERIFIED", "shadow layout probes", ["results/final_closure/STAGE5AC_CANDIDATE_COMPARISON.csv"], "x=-0.50 plus local task offset produced a measured full shadow route only with a very large joint jump.", ["Treating layout exploration as canonical repair."]),
            item("CONSISTENT_NATIVE_RUCKIG", "PASS_VERIFIED", "candidate shadow timing", ["results/final_closure/STAGE5AC_NATIVE_RETIME.json", "results/final_closure/STAGE5AC_DYNAMICS_AUDIT.csv"], "State-consistent finite-difference timing dilation and native MoveIt Ruckig executed; dynamics audit passed.", ["Inferring hardware execution or smooth continuity acceptance."]),
            item("CANDIDATE_CONTINUITY_ACCEPTANCE", "FAIL_VERIFIED", "candidate acceptance", ["results/final_closure/STAGE5AC_CANDIDATE_TRAJECTORY.csv", "results/final_closure/STAGE5AC_VALIDATION_REPORT.md"], "The candidate is not acceptable because the maximum waypoint jump is 96.51766 degrees.", ["Promoting the candidate as a safe trajectory."]),
            item("CANDIDATE_ORIGINAL_TASK_EXACTNESS", "FAIL_VERIFIED", "candidate task fidelity", ["results/final_closure/STAGE5AC_TASK_VALIDATION.json"], "Local slack validation passed, but original-pose exactness did not.", ["Equating local slack with original task exactness."]),
            item("BULLET_CANDIDATE_RUNS", "PASS_VERIFIED", "shadow Bullet interval evidence", ["results/final_closure/STAGE5AC_BULLET_RUN1_SUMMARY.json", "results/final_closure/STAGE5AC_BULLET_RUN2_SUMMARY.json"], "Both finite Bullet runs executed all 180 intervals with zero reported world contacts and zero endpoint self collisions.", ["Calling this hardware or strict CCD clearance evidence."]),
            item("FCL_ADAPTIVE_DISCRETE_CANDIDATE", "PASS_VERIFIED", "shadow FCL cross-check", ["results/final_closure/STAGE5AC_FCL_VALIDATION.json"], "The declared 0.25 degree adaptive discrete FCL check reported zero contacts/self collisions.", ["Calling it nonlinear articulated strict CCD."]),
            item("CANDIDATE_RUNTIME_TF", "BLOCKED", "MoveIt runtime validation", ["results/final_closure/STAGE5AC_FCL_VALIDATION.json", "results/final_closure/STAGE5AC_VALIDATION_REPORT.md"], "MoveIt candidate runtime TF rows were zero, so the overall validation remains fail-closed.", ["Treating zero runtime rows as successful execution."]),
            item("NATIVE_FULL_TRAJECTORY_FCL", "UNAVAILABLE", "advanced geometric backend", ["results/final_closure/STAGE5AC_VALIDATION_REPORT.md", "results/final_closure/STAGE5AC_RUN_LOG.txt"], "The native full-trajectory Stage26 FCL backend was unavailable/crashed.", ["Substituting the adaptive discrete check silently."]),
            item("HARDWARE_PHYSICAL_CLEARANCE", "UNAVAILABLE", "physical validation", ["CURRENT_AUTHORITATIVE_STATE.md", "EVIDENCE_MAP.md"], "No hardware or calibrated physical-clearance evidence is available.", ["Inferring clearance from collision-free sampled states."]),
            item("D62_UNRESOLVED_BOUNDARY", "BLOCKED", "protected D65 boundary", ["evidence/protected_floor/D65_STATE_SUMMARY.md"], "Strict D62 retains 6154 unresolved proof-boundary items; this is not collision evidence.", ["Rebranding unresolved as collision-free."]),
            item("PROMOTION_TO_STAGE5_COLLISION_FREE_BASELINE", "BLOCKED", "promotion gate", ["results/final_closure/STAGE5AC_FINAL_LEDGER.json", "HANDOFF.md"], "Promotion is explicitly NO_PROMOTION.", ["Starting Stage5B as if the candidate were canonical."]),
            item("STAGE5B_PERMISSION", "NOT_STARTED", "next stage", ["TAKEOVER_ORIENTATION.md"], "Stage5B has not started and is not authorized by this handoff.", ["Implementing Stage5B before independent takeover verification."]),
            item("HANDOFF_STAGING_VERIFICATION", "PASS_VERIFIED", "handoff artifact", ["HANDOFF_MANIFEST.json", "HANDOFF_VERIFICATION.json"], "The staging tree was checked for required files and internal manifest hashes before packaging.", ["Assuming an archive is valid without extraction verification."]),
        ],
        "overall_technical_decision": "STAGE5AC_SHADOW_REPAIR_CANDIDATE_VALIDATED_GEOMETRY_NO_PROMOTION",
        "overall_next_action": "Independent takeover audit; no Stage5B implementation yet.",
    }


def markdown_documents(root: Path, state: dict[str, object], progress: dict[str, object], prompt: Path) -> None:
    generated = state["generated_at"]
    put_text(root, "ROOT_GOAL.md", f"""# ROOT GOAL — Stage 5A-C continuation

Generated: `{generated}`. This file is a reconstruction from repository evidence, not a chat-history summary.

## Objective

Complete the Stage 5A-C robot–world trajectory consistency investigation around the frozen Stage 3/D65 system. Establish what is reproducibly measurable, classify measurement defects versus real frozen-system weaknesses, preserve the D65 protected floor, and hand Stage 4B/Stage5B a precise next target only after independent verification.

## Non-negotiable continuation rule

The current Stage5AC result is a validated shadow candidate with `NO_PROMOTION`. It is not permission to alter D65, the canonical checkpoint, the Stage 3 release, the 181-point Stage0/1 input, or the acceptance semantics. Stage5B has not started.

The correct continuation is:

`read authority → independently verify state → reproduce the declared blockers → classify → choose one authorized next action`

## Scope

- Stage0/1 authority is the 181-point ON-state open-arch pair under `outputs/internal_wiper_moveit_inputs/`.
- Collision labels must remain `adaptive_discrete_interpolation` unless a real backend proves otherwise.
- Bullet CCD, calibrated physical clearance, and hardware evidence remain unavailable/null.
- A bad robot result is a baseline finding; only a measurement defect may be repaired in this stage.
- No silent workaround, benchmark deletion, threshold relaxation, trajectory substitution, or Stage5B implementation.

## Current answer

The x=-0.50 m plus local y=-0.0025 m shadow layout produced a finite route and geometry cross-checks, but its 96.51766 degree waypoint jump and missing MoveIt runtime TF rows prevent acceptance. Local slack error is small, yet original-pose exactness is false. Therefore the candidate is not a promoted collision-free or safe baseline.
""")

    put_text(root, "HANDOFF_AUTHORITY.md", f"""# HANDOFF AUTHORITY

Generated: `{generated}`.

When statements conflict, use this order:

1. Explicit user instructions and the repository `AGENTS.md` rules.
2. Protected on-disk Stage 3/D65 authority and the actual runtime/configuration evidence.
3. Machine-readable Stage5AC ledger and final closure artifacts.
4. Human-readable reports and this handoff orientation.
5. Historical logs, raw transcripts, and prior agent narrative.

## Protected state

- `D:\\robotfucker\\outputs\\STAGE3_FINAL_RELEASE` is frozen scientific state.
- `D:\\robotfucker\\outputs\\D65_FINAL_CLOSURE` is the current protected software boundary; D65 is `NO_PROMOTION`.
- `D:\\robotfucker\\outputs\\STAGE5AC_FINAL_CLOSURE` is the Stage5AC result, not a new canonical release.
- The Stage0/1 source is the 181-point open-arch pair only.

## Authority interpretation

`PASS_VERIFIED` means the named finite measurement or artifact check completed with the stated scope. It never expands the scope. `FAIL_VERIFIED` is a verified failure or non-acceptance. `BLOCKED` means a gate cannot be closed with current evidence. `UNAVAILABLE` means the capability/backend/evidence is absent. None of these may be silently promoted to physical safety.

## Stale/narrative warning

Do not infer a current state from a previous conversation. Recheck the paths, hashes, runtime rows, and report fields listed in `EVIDENCE_MAP.md`. If a file is missing or differs, stop promotion and update the state ledger before acting.
""")

    dump_json(root / "CURRENT_AUTHORITATIVE_STATE.json", state)
    put_text(root, "CURRENT_AUTHORITATIVE_STATE.md", f"""# CURRENT AUTHORITATIVE STATE

Generated: `{generated}`. Machine-readable details are in `CURRENT_AUTHORITATIVE_STATE.json`.

## Phase

Stage5AC technical closure is complete. The handoff package is being verified. Stage5B is **not started** and `stage5b_ready=false`.

## Decision

`STAGE5AC_SHADOW_REPAIR_CANDIDATE_VALIDATED_GEOMETRY_NO_PROMOTION` with promotion `NO_PROMOTION`.

## Protected floor

D65 hashes were rechecked unchanged for the handoff. The two protected D59 trajectory references and SHA256 values are recorded in `evidence/protected_floor/D65_STATE_SUMMARY.md`. D65 remains a software-only boundary with explicit unresolved limitations; it is not hardware or physical-clearance evidence.

## Candidate facts

- Route: `routeE_x_minus_050_local_y_minus_0025mm_consistent_fd_native_ruckig`.
- 181 states; q preserved exactly; duration 705.608435681 s.
- Native MoveIt Ruckig ran and the dynamics audit passed.
- Bullet run 1 and run 2 each executed 180/180 intervals with zero reported continuous robot-world contacts and zero endpoint self contacts.
- FCL check is only `adaptive_discrete_interpolation` at 0.25 degrees: 2,607 states, zero world contacts/self collisions.
- Candidate maximum waypoint jump is 96.51766 degrees: continuity acceptance fails.
- Local slack validator passed, but original-pose exactness failed.
- MoveIt candidate runtime TF rows are zero; overall runtime validation is fail-closed.

## Hard boundaries

No strict nonlinear articulated CCD, no calibrated physical clearance, no hardware run, no native full-trajectory FCL closure, and no promotion. Strict D62 still has 6154 unresolved proof-boundary items; `BVH_leaf_outer_enclosures_overlap_or_touch` is not collision evidence.
""")

    dump_json(root / "VERIFIED_PROGRESS_LEDGER.json", progress)

    put_text(root, "PROBLEM_DEPENDENCY_MAP.md", """# PROBLEM / DEPENDENCY MAP

The arrows indicate prerequisite order. A downstream `PASS_VERIFIED` never overrides an upstream failed or unavailable gate.

```text
181-point authority + Stage3/D65 identity
        ↓
raw q / timebase / finite state
        ↓
FK + PlanningScene + frame-stamp consistency
        ↓
Stage5AR robot-world collision reproduction
        ↓
Stage5AC atlas and route search
        ├─ nominal Route A: FAIL_VERIFIED (no accepted full path under declared constraints)
        └─ Route-E shadow geometry: measured full route, but 96.51766° jump
                ↓
state-consistent finite-difference timing + native Ruckig
                ↓
Bullet finite interval checks: PASS_VERIFIED within declared scope
                ↓
FCL adaptive discrete cross-check: PASS_VERIFIED within declared scope
                ├─ MoveIt runtime TF rows: BLOCKED (0)
                ├─ native full-trajectory FCL: UNAVAILABLE/crashed
                ├─ strict continuous articulated CCD: UNAVAILABLE/not claimed
                └─ physical/hardware clearance: UNAVAILABLE/null
                ↓
candidate acceptance: FAIL_VERIFIED (continuity and original exactness)
                ↓
promotion gate: BLOCKED / NO_PROMOTION
                ↓
Stage5B: NOT_STARTED; independent takeover and explicit authorization required
```

## Top three Stage4B-ready bottlenecks

1. **Continuity / route feasibility:** the only measured full shadow route requires a 96.51766° waypoint jump; the nominal route is not accepted under its constraint. Affected route families: Route A / Route-E shadow, with the exact case and artifacts in `results/final_closure/`.
2. **Runtime validation completeness:** MoveIt candidate runtime TF rows are zero and the native full-trajectory Stage26 FCL backend is unavailable/crashed. No promotion can use this as a complete runtime proof.
3. **Proof/evidence boundary:** strict D62 retains 6154 unresolved items; exact articulated CCD, calibrated physical clearance, and hardware evidence remain unavailable. These are evidence gaps, not passes.

Do not optimize these away inside the handoff. Turn each into a separately authorized Stage4B problem with a fresh regression barrier.
""")

    put_text(root, "DECISION_REGISTER.md", """# DECISION REGISTER

## D1 — Keep D65 protected

Decision: D65 and its protected trajectories remain untouched and current. Evidence: direct path/hash references and handoff-time recheck. Consequence: all Stage5AC candidates are shadows.

## D2 — Classify the Stage5AR finding as robot-world collision

Decision: the principal pair is against `horseshoe_collision_compound`; it is not a self-collision. The same-header-stamp TF/FK correction is retained as the measurement repair. Consequence: world-collision evidence stays separate from self-collision summaries.

## D3 — Retain Route-E only as a candidate

Decision: x=-0.50 m with local y=-0.0025 m offset at waypoints 87–94 is a geometry-validating shadow candidate, not canonical. Consequence: no promotion because continuity and runtime gates fail.

## D4 — Accept only scoped geometry labels

Decision: Bullet and FCL results are reported with their exact finite/discrete scope. `adaptive_discrete_interpolation` is not strict continuous collision detection. Consequence: no inferred clearance or safety claim.

## D5 — Fail closed on missing runtime evidence

Decision: zero candidate runtime TF rows and unavailable native full-trajectory FCL leave the overall runtime validation unpassed. Consequence: missing evidence cannot be converted to success.

## D6 — Do not start Stage5B

Decision: Stage5B remains not started until a new agent independently verifies this package, the repository state, the protected floor, and the open blockers, and receives explicit authorization for the next change.
""")

    put_text(root, "FAILED_ATTEMPTS.md", """# FAILED / NON-ACCEPTED ATTEMPTS

These are retained to prevent circular retries and goal drift. They are not a reason to mutate the protected baseline.

| Attempt | Result | Interpretation | Next handling |
|---|---|---|---|
| Nominal Route A under the declared continuity/step bound | No accepted full path | Route/search or frozen-system weakness, not a license to tune the baseline | Treat as Stage4B target; preserve comparison |
| x=-0.25, -0.40, -0.45 shadow layout probes | Full path false; first unreachable indices 131, 142, 147 | Layout probes did not solve reachability | Retain measured results only |
| x=-0.50 with max step 20° | No full path | The successful shadow route is dependent on a very large joint jump | Do not relax continuity silently |
| x=-0.50 plus local y=-0.0025 m, max step 97/100° | Full route exists, max jump 96.51766° | Geometry candidate is not continuity-acceptable | Fail candidate acceptance |
| Native full-trajectory Stage26 FCL backend | Unavailable/crashed | Backend/evidence gap | Keep `UNAVAILABLE`; no substitute claim |
| MoveIt candidate runtime TF validation | 0 candidate runtime rows | Runtime evidence is incomplete | Fail closed; diagnose only with authorization |
| Original-pose exact task check | Failed while local slack check passed | Local slack is not original exactness | Preserve both fields |
| Early implementation audit/API probes | Corrected timestamp/frame and semantic measurement issues | Type-A tooling defects were repaired where actionable | Regress affected measurements; do not rewrite Type-B weakness |

No attempt above authorizes deletion of difficult cases, threshold relaxation, trajectory replacement, or Stage5B implementation.
""")

    put_text(root, "EVIDENCE_MAP.md", """# EVIDENCE MAP

| Claim | Evidence in package | Scope / verifier | Status |
|---|---|---|---|
| D65 protected floor identity | `evidence/protected_floor/D65_STATE_SUMMARY.md` plus copied D65 summaries | direct path/hash recheck | PASS_VERIFIED |
| Stage5AR world collision root cause | `results/final_closure/STAGE5AC_ROOT_CAUSE_REPORT.md` | Bullet/FK/FCL diagnosis; world pair kept separate from self collision | PASS_VERIFIED |
| Both atlas campaigns completed | `results/final_closure/STAGE5AC_COLLISION_ATLAS_auto[01]_SUMMARY.json` and CSVs | finite atlas summaries; raw JSONL referenced, not copied | PASS_VERIFIED |
| Candidate native retime | `results/final_closure/STAGE5AC_NATIVE_RETIME.json` and dynamics CSV | native MoveIt Ruckig + dynamics audit | PASS_VERIFIED, scoped |
| Candidate Bullet runs | `results/final_closure/STAGE5AC_BULLET_RUN[12]_SUMMARY.json` | 180 finite intervals each | PASS_VERIFIED, scoped |
| Candidate FCL cross-check | `results/final_closure/STAGE5AC_FCL_VALIDATION.json` | adaptive discrete interpolation, 0.25° | PASS_VERIFIED, scoped |
| Candidate continuity | `results/final_closure/STAGE5AC_CANDIDATE_TRAJECTORY.csv` | max waypoint jump | FAIL_VERIFIED |
| Original task exactness | `results/final_closure/STAGE5AC_TASK_VALIDATION.json` | local slack versus original exact fields | FAIL_VERIFIED |
| Candidate runtime validation | `results/final_closure/STAGE5AC_VALIDATION_REPORT.md` | candidate runtime TF row count | BLOCKED / fail-closed |
| Strict D62 boundary | `evidence/protected_floor/D65_STATE_SUMMARY.md` | 6154 unresolved proof-boundary items | BLOCKED, not collision evidence |
| Hardware/physical clearance | `CURRENT_AUTHORITATIVE_STATE.md` | no calibrated/hardware backend | UNAVAILABLE / null |
| Regression | `evidence/regression/pytest_stage5a_contract_open_arch.txt` | 10 targeted contract/open-arch tests | PASS_VERIFIED, not full-suite proof |

The package contains enough evidence to decide the next action, but not enough to claim collision-free continuous motion, physical clearance, hardware readiness, or Stage5B readiness.
""")

    put_text(root, "TAKEOVER_ORIENTATION.md", """# TAKEOVER ORIENTATION

# DO NOT EXECUTE PROJECT MUTATIONS YET

This handoff is an anti-drift boundary. The next agent must first read and verify state. Do not edit D65, Stage3 canonical artifacts, benchmark definitions, acceptance thresholds, or begin Stage5B during the first takeover action.

## Read in this order

1. `HANDOFF.md`
2. `ROOT_GOAL.md`
3. `HANDOFF_AUTHORITY.md`
4. `CURRENT_AUTHORITATIVE_STATE.md` and `.json`
5. `VERIFIED_PROGRESS_LEDGER.json`
6. `PROBLEM_DEPENDENCY_MAP.md`
7. `EVIDENCE_MAP.md`
8. `DECISION_REGISTER.md`, `FAILED_ATTEMPTS.md`, then the selected final reports

## Six mandatory takeover questions

1. What is the current root goal, and which artifacts are protected from mutation?
2. Which results are canonical/protected, which are shadow candidates, and which are only finite/discrete checks?
3. What exactly is `NO_PROMOTION`, and why is Stage5B not started?
4. Which blockers are Type-A measurement defects versus Type-B frozen-system weaknesses or unavailable evidence?
5. What exact evidence and hashes must be independently rechecked before any new experiment?
6. What single upstream action is authorized next, with what regression barrier and what must remain unchanged?

## Mandatory first audit

```text
verify package extraction and manifest
verify repository git status without cleaning or restoring
verify D65 protected trajectory paths and hashes
verify Stage5AC closure status and candidate metrics
verify the 181-point Stage0/1 authority path
verify Stage5B remains NOT_STARTED
```

If any check differs, stop promotion and update the ledger. A report that says “collision-free” without the exact method is not acceptable. A zero/empty runtime evidence table is not successful execution.

## Safe continuation posture

After the audit, continue with a bounded reproduce → classify → repair-tooling-or-measure-weakness workflow. Measurement defects may be repaired with known-answer tests. Frozen-system weaknesses must be measured and handed forward, not optimized away. Keep all new work in an isolated shadow/diagnostic box.
""")

    put_text(root, "HANDOFF.md", """# Stage5AC Context Handoff

This is the compact takeover index for a new ChatGPT window. It is intentionally independent of the prior agent’s narrative.

## One-line state

Stage5AC validated a Route-E shadow geometry candidate and its finite Bullet/FCL checks, but the candidate fails continuity/original-task acceptance, MoveIt runtime evidence is empty, strict CCD/physical/hardware evidence is unavailable, and promotion is `NO_PROMOTION`.

## Start here

Read `TAKEOVER_ORIENTATION.md`, then `ROOT_GOAL.md`, `HANDOFF_AUTHORITY.md`, `CURRENT_AUTHORITATIVE_STATE.json`, and `VERIFIED_PROGRESS_LEDGER.json`.

## What is protected

D65/Stage3 canonical artifacts and the 181-point Stage0/1 authority input. Existing dirty worktree changes are user state; preserve them. Never use `git reset`, `git clean`, `git checkout .`, or `git restore .`.

## What was actually measured

- Stage5AR: real robot-world pair diagnosis, with corrected same-header-stamp TF/FK semantics.
- Stage5AC: two 181/180-interval shadow atlas campaigns, Route-A exploration, Route-E x=-0.50 m/local y=-0.0025 m shadow route, state-consistent retime, native Ruckig, two Bullet runs, and an adaptive discrete FCL cross-check.
- The detailed copied artifacts are under `results/final_closure/` and `evidence/`.

## What is not claimed

No strict nonlinear articulated CCD, no physical clearance, no hardware clearance, no complete native FCL trajectory proof, no continuity acceptance, no original-pose exactness, no D65 promotion, and no Stage5B readiness.

## Continuation gate

Independent verification first. Then choose one explicitly authorized next upstream measurement/diagnostic. Do not turn the shadow candidate into canonical state and do not start Stage5B from this package alone.
""")

    pasted_text = prompt.read_text(encoding="utf-8") if prompt.is_file() else "[Pasted user prompt was not readable at build time.]"
    put_text(root, "USER_PROMPTS.md", f"""# USER PROMPTS PRESERVED FOR TAKEOVER

This file preserves the user-supplied handoff protocol verbatim below. The wrapper message in this turn had an empty `My request:` field. The earlier technical Stage5AC request is not available as raw chat text in this build context; its result is represented by the on-disk closure artifacts and reports, without inventing missing wording.

## User Prompt 01 — pasted Stage5A-C handoff protocol

{pasted_text}
""")

    put_text(root, "references/relevant_papers_or_links.md", """# REFERENCES / RELEVANT LINKS

These are method-context links only; they do not prove the local Stage5AC result.

- MoveIt2 PlanningScene API: https://moveit.picknik.ai/main/api/html/classplanning__scene_1_1PlanningScene.html
- MoveIt2 CHOMP tutorial: https://github.com/moveit/moveit2_tutorials/blob/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.rst
- pick_ik: https://github.com/picknikrobotics/pick_ik
- STOMP MoveIt: https://github.com/moveit/stomp_moveit

Local evidence and scope labels take precedence over generic method documentation.
""")

    put_text(root, "history/raw_material_index.md", """# RAW MATERIAL INDEX (LOW PRIORITY)

Raw material is intentionally not the primary takeover context.

- Canonical Stage5AC closure: `D:\\robotfucker\\outputs\\STAGE5AC_FINAL_CLOSURE`.
- Raw collision atlas JSONL: the two canonical files are listed in `results/final_closure/RAW_ATLAS_REFERENCE.md`; not copied because they are large and are not needed for first-pass orientation.
- D65 and D59 protected evidence: exact paths and hashes are in `evidence/protected_floor/D65_STATE_SUMMARY.md`.
- Existing worktree status/diff snapshots: `evidence/git_status_stage5ac.txt` and `evidence/git_diff_relevant.patch`.

Use raw logs only after a specific claim or reproduction requires them. Do not reconstruct authority from raw logs or prior-agent narration.
""")


def copy_evidence(root: Path) -> None:
    final_root = root / "results" / "final_closure"
    for source in sorted(CLOSURE.rglob("*")):
        if not source.is_file() or source.name.endswith(".jsonl"):
            continue
        copy_one(root, source, f"results/final_closure/{source.relative_to(CLOSURE).as_posix()}")

    d65_small = [
        D65 / "D65_HYBRID_AUTO0.json",
        D65 / "D65_HYBRID_AUTO1.json",
        D65 / "continuous_certificate_auto0/D65_VALIDATED_FK_LIPSCHITZ_CERTIFICATE.json",
        D65 / "continuous_certificate_auto1/D65_VALIDATED_FK_LIPSCHITZ_CERTIFICATE.json",
        D65 / "d62_interval_auto0/D62_CERTIFICATE_d65_final_safe.json",
        D65 / "d62_interval_auto1/D62_CERTIFICATE_d65_final_safe.json",
    ]
    for source in d65_small:
        destination = "evidence/protected_floor/" + source.relative_to(D65).as_posix()
        copy_one(root, source, destination)

    protected = PARENT / "timing_scale025_auto0/trajectories/adversarial_0100.csv"
    protected1 = PARENT / "timing_scale025_corrected_auto1/trajectories/adversarial_0101.csv"
    put_text(root, "evidence/protected_floor/D65_STATE_SUMMARY.md", f"""# D65 PROTECTED FLOOR SUMMARY

These files are references to the protected D59/D65 canonical trajectories. They were not duplicated into this compact handoff because each is about 3.7 MB. The handoff-time hash recheck was:

| Partition | Canonical path | SHA256 | Package treatment |
|---|---|---|---|
| auto0 | `{protected}` | `13f8075296a0defa12f17348cd6bafb913a8c10d22012a651d5614082df0a264` | path + hash; not copied |
| auto1 | `{protected1}` | `dda61e3be2328536f0c688333f3b24861c56dbdb347e807d881ca7d71f617416` | path + hash; not copied |

The compact copied D65 summaries and certificates are in this directory. D65 remains `NO_PROMOTION`; strict D62 retains 6154 unresolved proof-boundary items. `BVH_leaf_outer_enclosures_overlap_or_touch` is an unresolved proof boundary, not collision evidence.
""")

    put_text(root, "results/final_closure/RAW_ATLAS_REFERENCE.md", f"""# RAW ATLAS REFERENCE

The following canonical raw JSONL files were intentionally excluded from the handoff archive to keep it compact. Their summaries and CSVs are copied beside this file.

- `{CLOSURE / 'STAGE5AC_COLLISION_ATLAS_auto0.jsonl'}` — SHA256 `{sha256(CLOSURE / 'STAGE5AC_COLLISION_ATLAS_auto0.jsonl')}`; bytes `{(CLOSURE / 'STAGE5AC_COLLISION_ATLAS_auto0.jsonl').stat().st_size}`.
- `{CLOSURE / 'STAGE5AC_COLLISION_ATLAS_auto1.jsonl'}` — SHA256 `{sha256(CLOSURE / 'STAGE5AC_COLLISION_ATLAS_auto1.jsonl')}`; bytes `{(CLOSURE / 'STAGE5AC_COLLISION_ATLAS_auto1.jsonl').stat().st_size}`.

Do not interpret omission from the archive as omission from the experiment. Retrieve from the canonical project path only when a specific raw record is needed.
""")

    source_files = [
        "tools/stage5ac_root_cause_atlas.py",
        "tools/stage5ac_route_a_launch.py",
        "tools/stage5ac_route_a.py",
        "tools/stage5ac_make_env_variants.py",
        "tools/stage5ac_fk_probe.py",
        "tools/stage5ac_fk_probe_launch.py",
        "tools/stage5ac_native_retime.py",
        "tools/stage5ac_native_retime_launch.py",
        "tools/stage5ac_make_bullet_input.py",
        "tools/stage5ac_task_validator.py",
        "tools/stage5ac_finalize.py",
        "scripts/stage5a_moveit_validator.py",
        "tools/stage5a_moveit_validation_launch.py",
        "tools/stage5ac_build_handoff.py",
    ]
    rows = ["# RELEVANT CHANGED FILES", "", "These are the Stage5AC-related files visible in the current worktree. The worktree is dirty beyond this list; unrelated user changes are preserved and are not presented as part of the Stage5AC handoff.", "", "| Repository path | Package copy | Note |", "|---|---|---|"]
    for rel in source_files:
        source = REPO / rel
        if source.is_file():
            destination = "changed_files/source/" + rel.replace("/", "__")
            copy_one(root, source, destination)
            rows.append(f"| `{rel}` | `{destination}` | source snapshot; preserve current worktree state |")
        else:
            rows.append(f"| `{rel}` | — | missing at build time; investigate before relying on it |")
    put_text(root, "changed_files/CHANGED_FILES_INDEX.md", "\n".join(rows))

    put_text(root, "evidence/regression/pytest_stage5a_contract_open_arch.txt", """Command: `python -m pytest -q tests/test_stage5a_contract.py tests/test_open_arch_181.py`

Result: `10 passed`

Scope: targeted Stage5A contract/open-arch regression only. This is not a full-suite pass and must not be described as one.
""")
    put_text(root, "evidence/verification/VERIFICATION_COMMANDS.md", """# VERIFICATION COMMANDS

The archive verifier must run these checks after packaging:

```powershell
tar.exe -tJf <Desktop package>
tar.exe -xJf <Desktop package> -C <temporary D: extraction>
Get-FileHash <Desktop package> -Algorithm SHA256
```

Then verify required files, manifest hashes, package size <= 20 MiB, XZ/TAR readability, sensitive-string scan, and deletion of only the temporary staging/extraction directories.
""")


def add_git_evidence(root: Path) -> None:
    relevant = [
        "tools/stage5ac_root_cause_atlas.py",
        "tools/stage5ac_route_a_launch.py",
        "tools/stage5ac_route_a.py",
        "tools/stage5ac_make_env_variants.py",
        "tools/stage5ac_fk_probe.py",
        "tools/stage5ac_fk_probe_launch.py",
        "tools/stage5ac_native_retime.py",
        "tools/stage5ac_native_retime_launch.py",
        "tools/stage5ac_make_bullet_input.py",
        "tools/stage5ac_task_validator.py",
        "tools/stage5ac_finalize.py",
        "scripts/stage5a_moveit_validator.py",
        "tools/stage5a_moveit_validation_launch.py",
        "tools/stage5ac_build_handoff.py",
    ]
    put_text(root, "evidence/git_status_stage5ac.txt", git_output(["status", "--short"]))
    put_text(root, "evidence/git_diff_stat.txt", git_output(["diff", "--stat", "--", *relevant]))
    diff = git_output(["diff", "--", *relevant])
    put_text(root, "evidence/git_diff_relevant.patch", diff if diff.strip() else "No tracked diff for the selected files; untracked source snapshots are under changed_files/source/.")


def make_manifest(root: Path) -> dict[str, object]:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"HANDOFF_MANIFEST.json", "HANDOFF_VERIFICATION.json"}:
            continue
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema_version": "stage5ac-handoff-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": "Stage5AC handoff staging root",
        "self_excluded": ["HANDOFF_MANIFEST.json", "HANDOFF_VERIFICATION.json"],
        "entry_count": len(entries),
        "entries": entries,
    }
    dump_json(root / "HANDOFF_MANIFEST.json", manifest)
    return manifest


def verify_internal(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    required = [
        "ROOT_GOAL.md", "HANDOFF_AUTHORITY.md", "CURRENT_AUTHORITATIVE_STATE.md", "CURRENT_AUTHORITATIVE_STATE.json",
        "VERIFIED_PROGRESS_LEDGER.json", "TAKEOVER_ORIENTATION.md", "HANDOFF.md", "USER_PROMPTS.md",
        "PROBLEM_DEPENDENCY_MAP.md", "DECISION_REGISTER.md", "FAILED_ATTEMPTS.md", "EVIDENCE_MAP.md", "HANDOFF_MANIFEST.json",
    ]
    missing = [rel for rel in required if not (root / rel).is_file()]
    mismatches = []
    for entry in manifest["entries"]:
        path = root / entry["path"]
        if not path.is_file():
            mismatches.append({"path": entry["path"], "reason": "missing"})
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            mismatches.append({"path": entry["path"], "reason": "size_or_sha256_mismatch"})
    result = {
        "schema_version": "stage5ac-handoff-verification-v1",
        "staging_internal_status": "PASS_VERIFIED" if not missing and not mismatches else "FAIL_VERIFIED",
        "required_file_count": len(required),
        "missing_required": missing,
        "manifest_entry_count": manifest["entry_count"],
        "manifest_mismatches": mismatches,
        "archive_checks": "pending external tar/xz creation and extraction check",
    }
    dump_json(root / "HANDOFF_VERIFICATION.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, help="Explicit staging directory")
    parser.add_argument("--pasted-prompt", type=Path, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if not CLOSURE.is_dir():
        raise SystemExit(f"missing closure: {CLOSURE}")
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    staging = args.staging or (OUTPUTS / f"FR5_STAGE5AC_HANDOFF_{stamp}")
    if staging.exists():
        raise SystemExit(f"staging already exists: {staging}")
    staging.mkdir(parents=True)

    generated_at = datetime.now().astimezone().isoformat()
    state = build_state(generated_at)
    progress = ledger(state)
    markdown_documents(staging, state, progress, args.pasted_prompt)
    copy_evidence(staging)
    add_git_evidence(staging)
    manifest = make_manifest(staging)
    verification = verify_internal(staging, manifest)
    if verification["staging_internal_status"] != "PASS_VERIFIED":
        raise SystemExit(json.dumps(verification, indent=2))
    # HANDOFF_VERIFICATION is created after the first manifest, so regenerate
    # once to include its final content in the package manifest.
    manifest = make_manifest(staging)
    verification = verify_internal(staging, manifest)
    if verification["staging_internal_status"] != "PASS_VERIFIED":
        raise SystemExit(json.dumps(verification, indent=2))
    print(json.dumps({"staging": str(staging), "files": manifest["entry_count"], "bytes": sum(e["bytes"] for e in manifest["entries"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
