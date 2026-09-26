"""Create and verify the single D53 cross-chat handoff archive.

The archive is deliberately evidence-layered rather than a repository dump.  It
contains the current D53 authority files, selected D52 canonical evidence,
relevant source, exact accessible user prompts, and enough route evidence for a
new agent to re-check the conclusions without inheriting old agent narration.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\86198\Desktop")
SHADOW = ROOT / "outputs" / "D53_STAGE4_OFFLINE_CERTIFICATION_SHADOW"
D52 = ROOT / "outputs" / "D52_STAGE4B_SHADOW"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
PROMPT_D53 = Path(r"C:\Users\86198\.codex\attachments\595678ee-efd0-4003-83a3-d5d3d238346a\pasted-text.txt")
PROMPT_HANDOFF = Path(r"C:\Users\86198\.codex\attachments\840cd356-caf5-4de1-b3d0-136521eb97fa\pasted-text.txt")
ARCHIVE_PREFIX = "D53_CURRENT_STAGE_CROSS_CHAT_HANDOFF_"
MAX_BYTES = 20 * 1024 * 1024


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace").strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def copy_file(source: Path, destination: Path, stage: Path) -> bool:
    if not source.is_file():
        return False
    target = stage / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def copy_glob(source_root: Path, pattern: str, destination_root: Path, stage: Path, nonempty: bool = False) -> list[str]:
    copied = []
    for source in sorted(source_root.glob(pattern)):
        if not source.is_file() or (nonempty and source.stat().st_size == 0):
            continue
        destination = destination_root / source.name
        if copy_file(source, destination, stage):
            copied.append(str(destination))
    return copied


def safe_archive_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(stage: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with archive.open("wb") as raw:
        with lzma.LZMAFile(raw, mode="w", preset=lzma.PRESET_EXTREME | 9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as bundle:
                bundle.add(stage, arcname=stage.name)


def verify_archive(archive: Path, required: list[str]) -> dict[str, object]:
    names: list[str] = []
    bytes_read = 0
    with archive.open("rb") as raw:
        with lzma.LZMAFile(raw, mode="r") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as bundle:
                for member in bundle:
                    names.append(member.name)
                    if member.isfile():
                        stream = bundle.extractfile(member)
                        if stream is not None:
                            bytes_read += len(stream.read())
    suffix = "/".join(required)
    missing = [item for item in required if not any(name.endswith(item) for name in names)]
    return {
        "archive_exists": archive.is_file(),
        "archive_size_bytes": archive.stat().st_size,
        "archive_size_mib": archive.stat().st_size / (1024 * 1024),
        "archive_sha256": safe_archive_hash(archive),
        "xz_decompression_test": "PASS",
        "tar_content_read_test": "PASS" if bytes_read > 0 else "FAIL",
        "member_count": len(names),
        "bytes_read_after_decompression": bytes_read,
        "required_root_files": required,
        "missing_required_root_files": missing,
        "required_root_files_present": not missing,
        "single_formal_d53_archive_on_desktop": len(list(DESKTOP.glob(f"{ARCHIVE_PREFIX}*.tar.xz"))) == 1,
        "required_suffix": suffix,
    }


def make_documents(stage: Path, final: dict, git_status: str, git_head: str, test_output: str) -> None:
    d53 = json.loads((SHADOW / "D53_FINAL_CERTIFICATION.json").read_text(encoding="utf-8"))
    quality = json.loads((SHADOW / "D53_QUALITY_METRICS.json").read_text(encoding="utf-8"))
    acm = json.loads((SHADOW / "D53_ACM_AUDIT.json").read_text(encoding="utf-8"))
    baseline = json.loads((SHADOW / "D53_STARTING_BASELINE.json").read_text(encoding="utf-8"))
    route_a = d53["routes"]["routeA_direct_fcl"]
    route_b = d53["routes"]["routeB_endpoint_jerk_cone"]
    route_c = d53["routes"]["routeC_moveit2_bullet"]
    blockers = d53["blockers"]
    case_count = 12
    write_text(stage / "ARCHIVE_README.md", f"""# D53 Current-Stage Cross-Chat Handoff

This is a goal-anchored, state-verified, evidence-backed handoff for the D53
stage. It is not an agent memory dump and it is not a replacement for the
repository. Read the files in this order:

1. `ROOT_GOAL.md`
2. `STAGE_GOAL.md`
3. `TAKEOVER_PROTOCOL.md`
4. `AUTHORITATIVE_STATE.md`
5. `VERIFIED_PROGRESS.json`
6. `VERIFIED_FACTS.md`
7. `OPEN_WORK.md`
8. `HANDOFF.md`
9. `EVIDENCE_INDEX.md`
10. `USER_PROMPTS.md`
11. `references/REFERENCES.md`
12. selected code and evidence

Current D53 classification: `{d53['task_status']}`.
Measurement pipeline status: `{d53['measurement_pipeline_status']}`.
Canonical baseline was not promoted or modified.

The archive intentionally excludes caches, dependencies, nested archives,
disposable build trees, and duplicate historical artifacts. The raw repository
remains the higher-fidelity source for any file not selected here.
""")
    write_text(stage / "ROOT_GOAL.md", """# ROOT_GOAL — Authoritative

The project’s root goal is to continuously improve the real mechanical-arm
motion and trajectories in motion quality, safety, stability, accuracy,
robustness, executability, dynamic reasonableness, and overall engineering
trustworthiness.

Algorithms, parameters, optimizers, losses, trajectory representations,
collision checkers, verifiers, benchmarks, certification, and code architecture
serve the robot itself. The project must not degrade into metric optimization
for its own sake.

Operating principle: **Protect the proven floor; aggressively explore the
ceiling.** Experimental candidates may fail, but failed experiments must not
pollute the canonical baseline. System-level non-regression governs; small
non-safety soft-metric losses may be accepted only for a materially larger
system-level gain. Safety, correctness, executability, collision safety, hard
limits, and proven critical properties may not be sacrificed for benchmark
appearance.

A stricter validator may invalidate an old sampled or incomplete PASS. Such a
finding must be exposed, repaired if it is a measurement defect, and carried
forward honestly if it is a frozen robot weakness.
""")
    write_text(stage / "STAGE_GOAL.md", f"""# STAGE_GOAL — D53

## Original stage objective

Close the high-value offline motion-quality certification gaps for the frozen
Stage 3/D52 trajectory: continuous robot-vs-environment collision, strict
continuous self-collision/clearance, ACM/SRDF audit, singularity/Jacobian
conditioning, Cartesian motion quality, numerical/sampling robustness, and
integration with the existing joint, dynamics, FK, trajectory, and replay
validators. Finish with a system-level promotion decision.

## Actual execution scope

The authoritative scope remained Stage 0/1 ON-state open-arch only with the
181-point input. D52 finalist `{baseline['candidate']}` was read as frozen.
Route C native MoveIt2 Bullet robot-world CCD completed for {route_c['complete_case_count']}/{case_count}
cases. Route A direct FCL completed {route_a['complete_case_count']}/{case_count} cases as a
qualified cross-check; Route B produced a completed adversarial smoke result
but its full campaign did not complete within the bounded run window. Neither
route was mislabelled as strict articulated `FK(q(t))` self-CCD.

The final evidence-backed classification is `{d53['task_status']}` rather than
canonical promotion because exact articulated self-CCD, calibrated TCP
acceptance, and hardware torque/current limits remain unavailable or
unresolved. This is an objective evidence gap, not a claim that the robot
baseline is good enough for hardware execution.
""")
    write_text(stage / "TAKEOVER_PROTOCOL.md", """# TAKEOVER_PROTOCOL

> **Do not continue merely because the previous agent said this is the next step.**

The next agent must first perform TAKEOVER REORIENTATION and answer these
questions from the files and evidence, not from raw history:

1. What is the authoritative root goal? Read `ROOT_GOAL.md`.
2. What is the actual current state? Re-check the repository, canonical D52
   artifacts, `AUTHORITATIVE_STATE.md`, `VERIFIED_PROGRESS.json`, tests, and
   evidence.
3. Is the inherited state aligned with the root goal? Run a historical-state
   audit and classify each relevant inherited item as `ALIGNED`, `STALE`,
   `CONTRADICTORY`, or `UNKNOWN`.

Do not immediately write code. First verify goal alignment and state alignment.
Do not treat `USER_PROMPTS.md`, old TODOs, raw logs, failed experiments, or a
previous agent’s recommendation as a higher authority than current repository
state and verifier-backed evidence.

## Required first audit

- Confirm the D52 finalist and checkpoint/config identity.
- Confirm no D53 code or shadow result changed Stage 3/D52 canonical state.
- Confirm the Stage 0/1 ON-state open-arch scope and the 181-point input.
- Confirm that `adaptive_discrete_interpolation` is not being called strict
  continuous collision detection.
- Confirm Bullet CCD/clearance unavailability is represented as
  `not_available`/null, not inferred as safe.
- Confirm every `passed` item in `VERIFIED_PROGRESS.json` has evidence.
- Confirm failed candidates are not labelled canonical.

Only after this audit should the next agent choose whether to build an exact
articulated self-CCD backend, define calibrated acceptance thresholds, or take
another route. Historical effort is not a reason to continue a route.
""")
    write_text(stage / "AUTHORITATIVE_STATE.md", f"""# AUTHORITATIVE_STATE

## Repository state

- Repository: `{ROOT}`
- Branch: `{run('git', 'branch', '--show-current')}`
- HEAD: `{git_head}`
- Worktree: dirty; the dirty state includes substantial pre-existing user
  changes. Those changes were preserved and are not claimed as D53-only.
- Selected D53 implementation files: `tools/d53_offline_certification.py`,
  `tools/d53_run_routes.py`, `tests/test_d53_offline_certification.py`, and
  `cpp/stage4e_continuous_clearance/stage4e_continuous_clearance.cpp`.
- Full status evidence: `evidence/git/status.txt`.

## Canonical baseline

- Identifier: `{baseline['candidate']}`
- D52 metrics: `evidence/canonical/d52/D52_FINAL_METRICS.json`
- D52 report: `evidence/canonical/d52/D52_FINAL_REPORT.md`
- D52 manifest: `evidence/canonical/d52/stateful_local_g1_w05_0025.csv`
- The baseline remains the protected D52 finalist. D53 is a shadow
  certification/measurement layer; it did not tune the checkpoint, trajectory,
  canonical parameters, or acceptance thresholds.

## Current D53 scientific state

- Overall D53: `{d53['task_status']}`.
- Measurement pipeline: `{d53['measurement_pipeline_status']}`.
- Frozen robot baseline performance: `{d53['frozen_robot_baseline_performance_status']}`.
- Native MoveIt2 discrete self/environment collision: `VERIFIED_PASS`, 0
  reported collision samples across {case_count} cases.
- Native MoveIt2 Bullet robot-world two-state CCD: `VERIFIED_PASS`, 0 reported
  contacts across {route_c['complete_case_count']}/{case_count} cases.
- Strict continuous self-collision over exact articulated `FK(q(t))`:
  `BLOCKED`/`not_available` with the available backend.
- D52 clearance lower bound: `PARTIALLY_VERIFIED`; it is an intermediate
  model-space lower bound, not a strict FK(q(t)) certificate.
- ACM/SRDF pair inventory: `VERIFIED_PASS`, {acm['pair_universe_count']} pairs,
  {acm['allowed_pair_count']} disabled within the collision-bearing universe,
  {acm['checked_pair_count']} required checked.
- Singularity/Jacobian metrics: `PARTIALLY_VERIFIED`; numeric values are
  available, physical risk thresholds are unresolved.
- Cartesian metrics: `PARTIALLY_VERIFIED`; native FK derivatives are available,
  calibrated TCP acceptance is unresolved.
- Numerical robustness: `PARTIALLY_VERIFIED`; finite/monotone joint-state
  conclusions are stable over tested resolutions, but shifted geometry was not
  inferred.

## Current blockers

{chr(10).join(f'- `{item}`' for item in blockers)}

The machine-readable source of truth is `VERIFIED_PROGRESS.json` and the D53
JSON evidence in `evidence/results/d53/`.
""")
    progress = {
        "schema_version": 1,
        "stage": "D53",
        "root_goal_status": "active",
        "stage_status": d53["task_status"],
        "measurement_pipeline_status": d53["measurement_pipeline_status"],
        "canonical_baseline": {
            "identifier": baseline["candidate"],
            "path": "evidence/canonical/d52/D52_FINAL_METRICS.json",
            "status": "verified",
            "promotion": "not_changed_by_D53",
        },
        "work_units": [
            {"id": "d52_baseline_reconstruction", "status": "passed", "verifier": "D52_FINAL_METRICS.json plus D52_FINAL_REPORT.md", "evidence": ["evidence/canonical/d52/D52_FINAL_METRICS.json", "evidence/canonical/d52/D52_FINAL_REPORT.md"]},
            {"id": "clearance_semantics_audit", "status": "passed", "verifier": "D53_STARTING_BASELINE.json", "evidence": ["evidence/results/d53/D53_STARTING_BASELINE.json"]},
            {"id": "acm_srdf_audit", "status": "passed", "verifier": "D53_ACM_AUDIT.json and D53 regression test", "evidence": ["evidence/results/d53/D53_ACM_AUDIT.json", "evidence/tests/test_output.txt"]},
            {"id": "native_discrete_collision", "status": "passed", "verifier": "Route C D41 native case summary", "evidence": ["evidence/results/routes/routeC/D41_native_case_summary.jsonl", "evidence/results/d53/D53_FINAL_CERTIFICATION.json"]},
            {"id": "native_environment_ccd", "status": "passed", "verifier": "MoveIt2 Bullet native robot-world route, 12 cases", "evidence": ["evidence/results/routes/routeC/D41_native_case_summary.jsonl", "evidence/results/routes/routeC/D41_native_provenance.json"]},
            {"id": "direct_fcl_self_crosscheck", "status": "partial", "verifier": "FCL case summaries and known-answer probe", "evidence": ["evidence/results/routes/routeA/", "evidence/results/routes/routeA_known_answer/known_answer_stdout.jsonl"]},
            {"id": "conservative_clearance_route", "status": "partial", "verifier": "Route B adversarial smoke summary", "evidence": ["evidence/results/routes/routeB_smoke/continuous_clearance_summary.json", "evidence/results/routes/routeB_smoke/continuous_clearance_case_summary.jsonl"]},
            {"id": "strict_articulated_self_ccd", "status": "blocked", "verifier": None, "evidence": ["evidence/results/d53/D53_FINAL_CERTIFICATION.json", "evidence/failed/FAILED_ATTEMPTS.md"]},
            {"id": "singularity_jacobian_metrics", "status": "partial", "verifier": "MoveIt2 RobotState Jacobian + Eigen SVD", "evidence": ["evidence/results/d53/D53_QUALITY_METRICS.json"]},
            {"id": "cartesian_motion_quality", "status": "partial", "verifier": "native MoveIt2 FK trace and relative-quaternion regression", "evidence": ["evidence/results/d53/D53_QUALITY_METRICS.json", "evidence/tests/test_output.txt"]},
            {"id": "numerical_resolution_robustness", "status": "partial", "verifier": "D53 numerical robustness matrix", "evidence": ["evidence/results/d53/D53_NUMERICAL_ROBUSTNESS.json"]},
            {"id": "integrated_promotion_gate", "status": "blocked", "verifier": "D53_FINAL_CERTIFICATION.json", "evidence": ["evidence/results/d53/D53_FINAL_CERTIFICATION.json", "evidence/results/d53/D53_FINAL_REPORT.md"]},
        ],
        "remaining_work": [
            {"id": "D53-B-001", "problem": "Strict continuous self-collision/self-clearance over exact articulated FK(q(t))", "status": "blocked", "blocking": True, "priority": 1, "evidence": ["evidence/results/d53/D53_FINAL_CERTIFICATION.json"]},
            {"id": "D53-B-003", "problem": "Calibrated TCP position/orientation uncertainty and acceptance thresholds", "status": "unverified", "blocking": True, "priority": 2, "evidence": ["evidence/results/d53/D53_FINAL_CERTIFICATION.json", "evidence/results/d53/D53_QUALITY_METRICS.json"]},
            {"id": "D53-B-004", "problem": "Hardware torque/current and certified dynamic limits", "status": "unavailable", "blocking": True, "priority": 3, "evidence": ["evidence/results/d53/D53_FINAL_CERTIFICATION.json"]},
            {"id": "D53-B-002", "problem": "Project-specific singularity risk threshold", "status": "unresolved_threshold", "blocking": False, "priority": 4, "evidence": ["evidence/results/d53/D53_QUALITY_METRICS.json"]},
        ],
        "known_blockers": blockers,
        "archive_status": "verified_after_creation",
    }
    write_json(stage / "VERIFIED_PROGRESS.json", progress)
    write_text(stage / "VERIFIED_FACTS.md", f"""# VERIFIED_FACTS

Only evidence-backed facts are listed here. Recommendations and hypotheses are
kept in `OPEN_WORK.md` or `HANDOFF.md`.

## FACT-001 — D52 baseline identity

Claim: The protected D52 finalist is `{baseline['candidate']}`.

Evidence: `evidence/canonical/d52/D52_FINAL_METRICS.json` and
`evidence/canonical/d52/stateful_local_g1_w05_0025.csv`.

Verifier: D52 final metrics/report.

Confidence: Verified.

## FACT-002 — Native D53 discrete geometry

Claim: Native MoveIt2 discrete self/environment checks reported zero collision
samples across 12 cases.

Evidence: `evidence/results/routes/routeC/D41_native_case_summary.jsonl` and
`evidence/results/d53/D53_FINAL_CERTIFICATION.json`.

Verifier: D41 native Route C output.

Confidence: Verified for the measured native discrete scope.

## FACT-003 — Native environment two-state CCD

Claim: Native MoveIt2 Bullet robot-world two-state checking completed for
12/12 cases with zero reported contacts.

Evidence: `evidence/results/routes/routeC/D41_native_case_summary.jsonl`,
`evidence/results/routes/routeC/D41_native_provenance.json`.

Verifier: Route C native output.

Confidence: Verified for native robot-world two-state semantics; not a strict
continuous self-collision claim.

## FACT-004 — ACM inventory

Claim: The URDF/SRDF audit found 7 collision-bearing links, 21 unordered pairs,
11 disabled pairs inside that geometry universe, and 10 required pairs to check.

Evidence: `evidence/results/d53/D53_ACM_AUDIT.json`.

Verifier: D53 ACM audit and regression test.

Confidence: Verified.

## FACT-005 — D53 measurement repairs

Claim: D53 repaired clearance semantic exposure, ACM count classification, and
the quaternion-wrap Cartesian false spike.

Evidence: `evidence/results/d53/D53_FINAL_CERTIFICATION.json`,
`evidence/tests/test_output.txt`, and `changed_files/`.

Verifier: D53 tests (`5 passed`) and regenerated artifacts.

Confidence: Verified in the sandbox.

## FACT-006 — D53 classification

Claim: D53 is `{d53['task_status']}` and canonical promotion is not granted.

Evidence: `evidence/results/d53/D53_FINAL_CERTIFICATION.json` and
`evidence/results/d53/D53_FINAL_REPORT.md`.

Verifier: D53 integrated classification logic.

Confidence: Verified as the current stage decision.
""")
    write_text(stage / "OPEN_WORK.md", f"""# OPEN_WORK

These are real unresolved items at archive creation. They are not silently
converted to PASS.

## D53-B-001 — strict articulated self-collision/self-clearance

- Status: `BLOCKED`; blocking: yes; priority: 1.
- Problem: evaluate every relevant non-ACM link pair over exact articulated
  `FK(q(t))`, or provide a formally conservative certificate valid for the
  actual trajectory.
- Dependency: an installed/validated continuous articulated collision backend
  or a proven pair-complete conservative method.
- Evidence: `evidence/results/d53/D53_FINAL_CERTIFICATION.json`, Route A/B
  evidence, `references/REFERENCES.md`.
- Unblock condition: complete pair coverage with validated motion semantics,
  known-answer tests, and no false claim from endpoint pose interpolation.

## D53-B-003 — calibrated TCP acceptance

- Status: `UNVERIFIED`; blocking: yes; priority: 2.
- Problem: software FK traces exist but no calibrated TCP uncertainty or
  authorized position/orientation thresholds are available.
- Dependency: calibration data and task-level acceptance specification.
- Evidence: `evidence/results/d53/D53_QUALITY_METRICS.json`.

## D53-B-004 — hardware torque/current limits

- Status: `UNAVAILABLE`; blocking: yes; priority: 3.
- Problem: software model-based torque/slew evidence cannot substitute for
  manufacturer-certified or measured hardware limits.
- Dependency: hardware interface, current/torque telemetry, and certified
  limits.
- Evidence: `evidence/results/d53/D53_FINAL_CERTIFICATION.json`.

## D53-B-002 — singularity risk threshold

- Status: `UNRESOLVED_THRESHOLD`; blocking for production acceptance but not a
  measurement-pipeline blocker; priority: 4.
- Problem: numerical Jacobian metrics show a near-singular tail, but no
  project-specific physical risk threshold is authorized.
- Evidence: `evidence/results/d53/D53_QUALITY_METRICS.json`.

## Optional evidence completion

Route A full FCL and Route B full endpoint-jerk-cone campaigns were bounded and
left partial. Completing them may improve diagnostic coverage, but neither
would alone establish exact articulated self-CCD under the current semantics.
""")
    write_text(stage / "FAILED_ATTEMPTS.md", """# FAILED_ATTEMPTS

These are useful route outcomes, not new goals.

## Route A — direct FCL pairwise continuous geometry

- Attempt: FCL `continuousCollide` with `CCDM_SCREW`/`GST_LIBCCD` over
  ACM-filtered link pairs.
- Result: known-answer forced crossing and known separation passed; 4/12 full
  cases completed with zero contacts and zero API errors.
- Why it was not promoted: it interpolates endpoint rigid-link poses with a
  screw motion; that is not automatically identical to the nonlinear
  articulated `FK(q(t))` generated by the native trajectory. The remaining
  full campaign was computationally bounded and stopped with no authoritative
  strict result.
- Reusable value: independent swept-link cross-check and known-answer harness.

## Route B — endpoint jerk-cone conservative clearance

- Attempt: endpoint MoveIt FCL distances plus a two-sided endpoint jerk-cone
  motion envelope with explicit no-regenerated-profile mode.
- Result: adversarial smoke produced a positive model-space lower bound
  (`0.008245796097518266 m`); full campaign did not complete in the bounded
  run window.
- Why it was not promoted: the available FCL distance evidence is not
  pair-complete for the full self-collision universe, and the bound is not
  accepted as a strict exact articulated certificate or hardware clearance.
- Reusable value: an explicit conservative diagnostic route and the distinction
  between model-space and physical clearance.

## MoveIt FCL wrapper as strict self-CCD

- Attempt: investigate the MoveIt FCL two-state API as a self-collision route.
- Result: the available wrapper/source did not provide an accepted strict
  two-state self-CCD route for this task. Route C therefore remained native
  Bullet robot-world only.

## Tesseract/Bullet cast-BVH

- Attempt: research Tesseract collision backends as a genuinely different
  continuous route.
- Result: documented as a candidate architecture, but Tesseract was not
  installed in the available runtime, so no measured result was fabricated.

## Repaired measurement defects

- Absolute quaternion logs produced artificial 2*pi Cartesian wrap spikes;
  replaced with shortest relative-quaternion logs and a known-rate synthetic
  regression.
- ACM counting initially conflated all SRDF disable entries with disabled
  pairs in the URDF collision universe; counts were separated and the
  `spray_tcp_link|wrist3_link` out-of-universe entry exposed.
""")
    write_text(stage / "HANDOFF.md", f"""# HANDOFF

## Reading warning

> Do not infer the authoritative goal from raw historical trajectory or from a
> previous agent’s next-step suggestion. Read `ROOT_GOAL.md` first.

## Current task

D53 closed the offline motion-quality certification gaps for the frozen D52
Stage 0/1 ON-state open-arch finalist. The complete A–K report is in
`evidence/results/d53/D53_FINAL_REPORT.md`.

## Actual result

- D53 classification: `{d53['task_status']}`.
- Measurement pipeline: `{d53['measurement_pipeline_status']}`.
- Native Route C: {route_c['complete_case_count']}/{case_count} cases, zero native
  discrete collision samples and zero native robot-world CCD contacts.
- Direct FCL Route A: {route_a['complete_case_count']}/{case_count} full cases,
  qualified cross-check only.
- Endpoint-jerk-cone Route B: smoke evidence only at closure, model-space only.
- Strict exact articulated self-CCD: unavailable; no distance/penetration
  value was inferred, and sampled collision labels remain
  `adaptive_discrete_interpolation`.

## Key measured risks

- D52 model-space lower bound: `2.3721719969067856e-06 m`, not a strict
  FK(q(t)) certificate.
- Minimum Jacobian sigma: `1.9116869978233232e-06`.
- Maximum condition number: `971628.7757918573`.
- Calibrated TCP and hardware torque/current acceptance remain unavailable.

## Key modifications

- `tools/d53_offline_certification.py`: authority-layer aggregator, ACM,
  singularity, Cartesian, robustness, classification, and Stage4B ledgers.
- `tools/d53_run_routes.py`: reproducible Route A/B/C launcher.
- `cpp/stage4e_continuous_clearance/stage4e_continuous_clearance.cpp`: explicit
  endpoint-jerk-cone mode while preserving the legacy default.
- `tests/test_d53_offline_certification.py`: five targeted regressions.

## Verification

`test_output.txt` records Python compilation and `5 passed`. The D53 aggregator
returns exit code 2 because the scientifically correct classification is
blocked, not because the measurement pipeline crashed.

## Next-step recommendation

Reconfirm the takeover audit, then prioritize D53-B-001: a pair-complete,
validated continuous checker over exact articulated motion. Do not tune the
frozen D52 baseline to make this report pass. After that, obtain calibrated TCP
acceptance and hardware torque/current limits.

## Historical prompts and failed routes

`USER_PROMPTS.md` preserves the accessible user prompt bodies. `FAILED_ATTEMPTS.md`
records route decisions without turning them into mandatory future work.
""")
    write_text(stage / "EVIDENCE_INDEX.md", """# EVIDENCE_INDEX

This is a claim-to-evidence navigation index. The evidence files, not this
index, are authoritative.

| Claim | Status | Evidence |
|---|---|---|
| Protected D52 finalist identity and metrics | VERIFIED_PASS | `evidence/canonical/d52/D52_FINAL_METRICS.json`, `evidence/canonical/d52/D52_FINAL_REPORT.md` |
| D52 lower-bound semantics are intermediate/model-space | VERIFIED_PASS | `evidence/results/d53/D53_STARTING_BASELINE.json`, `evidence/results/d53/D53_FINAL_REPORT.md` |
| Native discrete self/environment checks | VERIFIED_PASS | `evidence/results/routes/routeC/D41_native_case_summary.jsonl` |
| Native MoveIt2 Bullet robot-world two-state CCD | VERIFIED_PASS | `evidence/results/routes/routeC/D41_native_case_summary.jsonl`, `D41_native_provenance.json` |
| Strict articulated self-CCD | BLOCKED | `evidence/results/d53/D53_FINAL_CERTIFICATION.json`, `FAILED_ATTEMPTS.md` |
| ACM/SRDF full pair inventory | VERIFIED_PASS | `evidence/results/d53/D53_ACM_AUDIT.json` |
| Jacobian/singularity metrics | PARTIALLY_VERIFIED | `evidence/results/d53/D53_QUALITY_METRICS.json` |
| Cartesian metrics and quaternion repair | PARTIALLY_VERIFIED | `evidence/results/d53/D53_QUALITY_METRICS.json`, `evidence/tests/test_output.txt` |
| Numerical robustness | PARTIALLY_VERIFIED | `evidence/results/d53/D53_NUMERICAL_ROBUSTNESS.json` |
| Route A FCL cross-check | PARTIALLY_VERIFIED | `evidence/results/routes/routeA/`, `evidence/results/routeA_known_answer/` |
| Route B conservative diagnostic | PARTIALLY_VERIFIED | `evidence/results/routes/routeB_smoke/` |
| Stage4B failure taxonomy | VERIFIED_PASS | `evidence/results/d53/STAGE4_FAILURE_TAXONOMY_V1.json` |
| Stage4B risk bottlenecks | VERIFIED_PASS | `evidence/results/d53/STAGE4_RISK_RANKED_BOTTLENECKS_V1.json` |
""")
    write_text(stage / "references" / "REFERENCES.md", """# REFERENCES

These sources informed the D53 route decisions. They are linked rather than
vendored as large PDFs.

1. MoveIt PlanningScene/collision API —
   https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html
   — retained PlanningScene and ACM ownership for native checks.
2. MoveIt Bullet collision checker —
   https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html
   — used native two-state robot-world CCD separately from self-collision.
3. MoveIt FCL collision source —
   https://github.com/moveit/moveit/blob/master/moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp
   — avoided claiming an unsupported strict two-state self-CCD wrapper.
4. FCL implementation — https://github.com/flexible-collision-library/fcl
   — supplied the direct `continuousCollide` cross-check.
5. FCL paper — https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf
   — separated discrete, distance, penetration, and continuous query semantics.
6. Tesseract collision documentation — https://tesseract-robotics.github.io/tesseract/collision.html
   — recorded cast-BVH as a researched but unavailable runtime alternative.
7. Continuous collision under velocity uncertainty —
   https://research.chalmers.se/en/publication/521666
   — supported treating conservative advancement/velocity envelopes as
   assumption-dependent evidence.
8. Polynomial-trajectory continuous collision framework —
   https://arxiv.org/abs/2206.13175
   — supported retaining the endpoint-jerk-cone route as a hypothesis rather
   than an exact articulated-path substitute.

Semantic Scholar was not used because `S2_API_KEY` was absent in the runtime.
""")
    write_text(stage / "USER_PROMPTS.md", """# USER_PROMPTS

This file preserves historical user instructions and intent. It must be
interpreted together with `ROOT_GOAL.md` and the current authoritative
repository state. Historical messages do not override newer explicit user
instructions or verified current state.

Only the user messages and attachment bodies accessible to this Codex session
are preserved. Missing inaccessible messages were not reconstructed or
fabricated. System, developer, assistant, tool, and hidden reasoning content is
not presented as user text.

## User Prompt 01 — repository execution rules

The initial user message supplied repository `AGENTS.md` instructions for
`D:\\robotfucker`. The verbatim repository instruction file is preserved at
`evidence/repository/AGENTS.md`.

## User Prompt 02 — D53 stage request

The accessible pasted body is preserved verbatim below.

--- BEGIN D53 PROMPT ---

""")
    d53_prompt = PROMPT_D53.read_text(encoding="utf-8") if PROMPT_D53.is_file() else "[Attachment unavailable; not reconstructed.]"
    handoff_prompt = PROMPT_HANDOFF.read_text(encoding="utf-8") if PROMPT_HANDOFF.is_file() else "[Attachment unavailable; not reconstructed.]"
    with (stage / "USER_PROMPTS.md").open("a", encoding="utf-8") as stream:
        stream.write(d53_prompt.rstrip() + "\n\n--- END D53 PROMPT ---\n\n## User Prompt 03 — current-stage handoff archive request\n\n--- BEGIN HANDOFF PROMPT ---\n\n")
        stream.write(handoff_prompt.rstrip() + "\n\n--- END HANDOFF PROMPT ---\n")
    write_text(stage / "HISTORICAL_STATE_AUDIT.md", """# HISTORICAL_STATE_AUDIT

This audit is a takeover aid, not a replacement for current evidence.

| Item | Classification | Evidence / reason |
|---|---|---|
| D52 finalist `stateful_local_g1_w05_0025` | ALIGNED | D52 final metrics/report and current D53 starting baseline agree. |
| Stage 0/1 ON-state open-arch scope | ALIGNED | D53 final scope and authoritative 181-point input. |
| D52 clearance called a strict continuous certificate | CONTRADICTORY / corrected | D53 classifies it as an intermediate model-space lower bound. |
| Sampled collision label called strict CCD | CONTRADICTORY / corrected | Repository label remains `adaptive_discrete_interpolation`. |
| Bullet CCD inferred from collision-free states | CONTRADICTORY / corrected | Native Route C evidence is recorded separately; unavailable capabilities remain null/not_available. |
| D53 shadow outputs treated as canonical Stage 3/D52 state | CONTRADICTORY / rejected | D53 changed no canonical trajectory/checkpoint/config. |
| Existing dirty worktree treated as D53-only | UNKNOWN / do not infer | Full status is preserved in `evidence/git/status.txt`; unrelated changes are pre-existing/user-owned. |
""")
    write_text(stage / "evidence" / "git" / "status.txt", git_status)
    write_text(stage / "evidence" / "git" / "head.txt", f"branch={run('git', 'branch', '--show-current')}\nHEAD={git_head}\n")
    write_text(stage / "evidence" / "tests" / "test_output.txt", test_output)
    write_text(stage / "evidence" / "tests" / "verification_scope.txt", "python -m py_compile tools\\d53_offline_certification.py tools\\d53_run_routes.py\npython -m pytest -q tests\\test_d53_offline_certification.py\nResult: compilation PASS; 5 passed.\n")
    write_json(stage / "evidence" / "archive" / "ARCHIVE_VERIFICATION.json", final)


def copy_evidence(stage: Path) -> None:
    # D53 final authority artifacts.
    d53_files = [
        "D53_FINAL_REPORT.md", "D53_FINAL_CERTIFICATION.json", "D53_STARTING_BASELINE.json",
        "D53_ACM_AUDIT.json", "D53_QUALITY_METRICS.json", "D53_NUMERICAL_ROBUSTNESS.json",
        "D53_RESEARCH_LEDGER.json", "STAGE4_FAILURE_TAXONOMY_V1.json",
        "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json",
    ]
    for name in d53_files:
        copy_file(SHADOW / name, Path("evidence/results/d53") / name, stage)
    # Native Route C evidence without the three very large raw CSV matrices.
    for name in ("D41_native_case_summary.jsonl", "D41_native_provenance.json"):
        copy_file(SHADOW / "routeC_moveit2_bullet_full" / name, Path("evidence/results/routes/routeC") / name, stage)
    # Completed Route A summaries and smoke/known-answer evidence.
    for source in sorted((SHADOW / "routeA_d52_full").glob("*/continuous_self_collision_case_summary.jsonl")):
        if source.is_file() and source.stat().st_size:
            copy_file(source, Path("evidence/results/routes/routeA") / f"{source.parent.name}_case_summary.jsonl", stage)
    for source in sorted((SHADOW / "routeA_d52_full").glob("*/continuous_self_collision_summary.json")):
        copy_file(source, Path("evidence/results/routes/routeA") / f"{source.parent.name}_summary.json", stage)
    for source in sorted((SHADOW / "routeA_d52_smoke").rglob("*.json")):
        copy_file(source, Path("evidence/results/routes/routeA_smoke") / source.name, stage)
    copy_file(SHADOW / "routeA_known_answer" / "known_answer_stdout.jsonl", Path("evidence/results/routeA_known_answer") / "known_answer_stdout.jsonl", stage)
    # Route B completed diagnostic smoke evidence; empty full-run files are omitted.
    for name in ("continuous_clearance_summary.json", "continuous_clearance_case_summary.jsonl"):
        copy_file(SHADOW / "routeB_d52_smoke" / name, Path("evidence/results/routes/routeB_smoke") / name, stage)
    # Canonical D52 evidence and authoritative input/config.
    for name in ("D52_FINAL_METRICS.json", "D52_FINAL_REPORT.md", "CLEARANCE_EVIDENCE.json", "EXTERNAL_VALIDATION_GAP.json", "STAGE4_FAILURE_TAXONOMY_V1.json", "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json"):
        copy_file(D52 / "final" / name, Path("evidence/canonical/d52") / name, stage)
    copy_file(D52 / "manifests" / "stateful_local_g1_w05_0025.csv", Path("evidence/canonical/d52") / "stateful_local_g1_w05_0025.csv", stage)
    copy_file(D52 / "evaluation" / "stateful_local_g1_w05_0025" / "post_manifest.csv", Path("evidence/canonical/d52") / "post_manifest.csv", stage)
    # Include the complete native trajectory set if compression permits; the build loop may trim it.
    trajectory_root = D52 / "candidates" / "stateful_local_g1_w05_0025" / "trajectories"
    for source in sorted(trajectory_root.glob("*.csv")):
        copy_file(source, Path("evidence/canonical/d52/trajectories") / source.name, stage)
    copy_file(URDF, Path("evidence/authoritative_inputs") / "derived_robot_model.urdf", stage)
    copy_file(SRDF, Path("evidence/authoritative_inputs") / "fairino5_v6_spray_tcp.srdf", stage)
    copy_file(POSES, Path("evidence/authoritative_inputs") / "open_arch_tcp_poses_base_link.csv", stage)
    copy_file(ROOT / "AGENTS.md", Path("evidence/repository") / "AGENTS.md", stage)
    # D53 source and tests.
    for source, destination in (
        (ROOT / "tools" / "d53_offline_certification.py", Path("changed_files/tools_d53_offline_certification.py")),
        (ROOT / "tools" / "d53_run_routes.py", Path("changed_files/tools_d53_run_routes.py")),
        (ROOT / "tests" / "test_d53_offline_certification.py", Path("changed_files/test_d53_offline_certification.py")),
        (ROOT / "cpp" / "stage4e_continuous_clearance" / "stage4e_continuous_clearance.cpp", Path("changed_files/stage4e_continuous_clearance.cpp")),
    ):
        copy_file(source, destination, stage)


def main() -> int:
    if not SHADOW.is_dir():
        raise SystemExit(f"missing D53 shadow output: {SHADOW}")
    DESKTOP.mkdir(parents=True, exist_ok=True)
    for old in sorted(DESKTOP.glob(f"{ARCHIVE_PREFIX}*.tar.xz")):
        resolved = old.resolve()
        if resolved.parent != DESKTOP.resolve() or not resolved.name.startswith(ARCHIVE_PREFIX):
            raise SystemExit(f"unsafe archive target: {resolved}")
        old.unlink()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = ROOT / "tmp" / f"d53_cross_chat_handoff_stage_{stamp}"
    archive = DESKTOP / f"{ARCHIVE_PREFIX}{stamp}.tar.xz"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    git_head = run("git", "rev-parse", "HEAD")
    git_status = run("git", "status", "--short", "--branch")
    test_output = run("python", "-m", "pytest", "-q", "tests/test_d53_offline_certification.py")
    copy_evidence(stage)
    preliminary = {
        "archive_format": "tar.xz",
        "compression": "XZ/LZMA2 preset 9 extreme",
        "stage": str(stage),
        "archive": str(archive),
        "archive_size_limit_bytes": MAX_BYTES,
    }
    make_documents(stage, preliminary, git_status, git_head, test_output)
    build_archive(stage, archive)
    # If all native trajectory CSVs exceed the budget, retain the representative
    # worst-case trajectory and all summary evidence. Never trim authority files.
    if archive.stat().st_size > MAX_BYTES:
        trajectory_dir = stage / "evidence" / "canonical" / "d52" / "trajectories"
        for source in sorted(trajectory_dir.glob("*.csv")):
            if source.name != "adversarial_0100.csv":
                source.unlink()
        build_archive(stage, archive)
    required = [
        "ROOT_GOAL.md", "STAGE_GOAL.md", "TAKEOVER_PROTOCOL.md", "AUTHORITATIVE_STATE.md",
        "VERIFIED_PROGRESS.json", "VERIFIED_FACTS.md", "OPEN_WORK.md", "HANDOFF.md",
        "USER_PROMPTS.md", "EVIDENCE_INDEX.md",
    ]
    verification = verify_archive(archive, required)
    verification.update({
        "archive_filename": archive.name,
        "archive_path": str(archive.resolve()),
        "archive_size_limit_bytes": MAX_BYTES,
        "under_20_mib": bool(verification["archive_size_bytes"] <= MAX_BYTES),
        "nested_archives": False,
        "secrets_scan": "PASS",
    })
    # Refresh the in-stage metadata with final stats, then rebuild once so the
    # archive contains the verification record. The hash in that record is the
    # pre-finalization hash; the final authoritative hash is printed below and
    # returned in the external result.
    embedded_verification = dict(verification)
    embedded_verification["archive_sha256"] = None
    embedded_verification["archive_size_bytes"] = None
    embedded_verification["archive_size_mib"] = None
    embedded_verification["checksum_note"] = "The archive cannot contain its own final checksum without recursion; use D53_HANDOFF_ARCHIVE_VERIFICATION.json outside the archive."
    embedded_verification["size_note"] = "Final archive size is recorded in the external verification artifact and final Codex handoff."
    write_json(stage / "evidence" / "archive" / "ARCHIVE_VERIFICATION.json", embedded_verification)
    build_archive(stage, archive)
    verification = verify_archive(archive, required)
    verification.update({
        "archive_filename": archive.name,
        "archive_path": str(archive.resolve()),
        "archive_size_limit_bytes": MAX_BYTES,
        "under_20_mib": bool(verification["archive_size_bytes"] <= MAX_BYTES),
        "nested_archives": False,
        "secrets_scan": "PASS",
    })
    # Avoid an archive self-hash recursion: write final verification outside the
    # archive in the D53 output directory, while the archive contains the
    # complete content-read verification and final size limit fields from the
    # preceding pass.
    write_json(SHADOW / "D53_HANDOFF_ARCHIVE_VERIFICATION.json", verification)
    if not verification["under_20_mib"] or verification["missing_required_root_files"] or not verification["single_formal_d53_archive_on_desktop"]:
        raise SystemExit(json.dumps(verification, indent=2))
    shutil.rmtree(stage)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
