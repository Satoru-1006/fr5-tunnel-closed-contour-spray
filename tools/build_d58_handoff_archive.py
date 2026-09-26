"""Build the goal-anchored D58 cross-chat handoff staging tree.

The final tar.xz is created and validated by the calling PowerShell/WSL
commands.  This script only assembles a compact, evidence-linked staging tree.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D58_STAGE4_FULL_SYSTEM_CLOSURE"
TMP = ROOT / "tmp"
ATTACH = Path(r"C:\Users\86198\.codex\attachments")
PREVIOUS_D58 = ATTACH / "a0897de3-f7d7-48e8-993e-ee4dba33ddf4" / "pasted-text.txt"
CURRENT_HANDOFF = ATTACH / "1d4ecf11-6952-4f19-b8fb-410f62476258" / "pasted-text.txt"
DESKTOP = Path(r"C:\Users\86198\Desktop")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_file(source: Path, destination: Path, copied: list[str]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(destination.as_posix())


def extract_root_goal(prompt: str) -> str:
    marker = "## ROOT GOAL\n"
    start = prompt.index(marker) + len(marker)
    end_marker = "\n---\n\nThis file must be treated as independent of the historical trajectory."
    end = prompt.index(end_marker, start)
    return prompt[start:end].strip()


def git(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return result.stdout + result.stderr


def main() -> None:
    if not PREVIOUS_D58.is_file() or not CURRENT_HANDOFF.is_file():
        raise FileNotFoundError("required user-provided D58 attachments are missing")
    tz = timezone(timedelta(hours=8))
    stamp = datetime.now(tz).strftime("%Y%m%dT%H%M%S+0800")
    stage = TMP / f"D58_CROSS_CHAT_HANDOFF_STAGING_{stamp}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    copied: list[str] = []

    previous_d58_text = PREVIOUS_D58.read_text(encoding="utf-8")
    current_handoff_text = CURRENT_HANDOFF.read_text(encoding="utf-8")
    root_goal = extract_root_goal(current_handoff_text)
    d58_status = load(OUT / "D58_FINAL_STATUS.json")
    d58_comparison = load(OUT / "D58_SYSTEM_COMPARISON.json")
    d56_status = load(ROOT / "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/D56_FINAL_STATUS.json")
    d57_status = load(ROOT / "outputs/D57_STAGE4_ALGORITHMIC_CLOSURE/D57_FINAL_STATUS.json")
    phase1 = load(OUT / "D58_PHASE1_STATEFUL_CLOSURE_REPORT.json")
    robust = load(OUT / "D58_ROBUST_CLEARANCE.json")

    write(stage / "ROOT_GOAL.md", "# ROOT GOAL\n\n" + root_goal + "\n")
    write(
        stage / "AUTHORITY_ORDER.md",
        """# Authority Order

The receiving Agent must resolve conflicts in this order:

0. **New explicit user instruction** — controls if the user changes the goal.
1. **ROOT GOAL** — `ROOT_GOAL.md` defines the invariant project objective.
2. **Current machine-readable authoritative state** — `AUTHORITATIVE_STATE.json`, `STATE_LEDGER.json`, promotion/canonical records, and verified identity.
3. **Verified facts and evidence** — `VERIFIED_FACTS.md`, machine-readable results, tests, and exact repository artifacts.
4. **Goal/state analysis** — `GOAL_STATE_GAP.md` and `TAKEOVER_REORIENTATION.md`.
5. **Human-readable summaries** — `HANDOFF.md` and D58 reports.
6. **Historical user prompts** — `USER_PROMPTS.md`, with later explicit instructions superseding earlier contradictory ones.
7. **Failed experiments, historical trajectory, and raw logs** — reference only; label as `NON_AUTHORITATIVE_HISTORY`.

Packaged prose never overrides the current machine-readable state or the root goal. Canonical, protected floor, verified champion, best shadow, and failed candidate are separate concepts.
""",
    )

    state_ledger = {
        "schema_version": "d58-cross-chat-state-ledger-v1",
        "stage": "D58",
        "status_vocabulary": ["VERIFIED_DONE", "PARTIAL", "NOT_STARTED", "FAILED_REVERTED", "BLOCKED_EXTERNAL", "INVALIDATED"],
        "work_units": [
            {"work_unit_id": "D58_INHERITED_D57_AUTO_0", "requirement": "Preserve and recertify D57 C4 auto_0 shadow", "depends_on": [], "status": "VERIFIED_DONE", "verification_type": "native summary and trajectory artifact", "evidence_paths": ["results/D58/stateful_consistent_scale001_auto0/execution_form_summary.json", "results/D58/D58_CANDIDATE_LEDGER.json"], "result_summary": "Native stateful post-Ruckig execution completed.", "baseline_impact": "shadow only; protected floor unchanged", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_INHERITED_D57_AUTO_1", "requirement": "Preserve and recertify D57 C4 auto_1 shadow", "depends_on": [], "status": "VERIFIED_DONE", "verification_type": "native summary and trajectory artifact", "evidence_paths": ["results/D58/stateful_consistent_scale001_auto1/execution_form_summary.json", "results/D58/D58_CANDIDATE_LEDGER.json"], "result_summary": "Native stateful post-Ruckig execution completed.", "baseline_impact": "shadow only; protected floor unchanged", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_NATIVE_RUCKIG", "requirement": "Run both shadows through native MoveIt2 Ruckig", "depends_on": ["D58_INHERITED_D57_AUTO_0", "D58_INHERITED_D57_AUTO_1"], "status": "VERIFIED_DONE", "verification_type": "native_post_ruckig=true and clean process exit", "evidence_paths": ["results/D58/stateful_consistent_scale001_auto0/execution_form_summary.json", "results/D58/stateful_consistent_scale001_auto1/execution_form_summary.json"], "result_summary": "2/2 PASS; q path preserved through stateful consistent-time repair.", "baseline_impact": "no canonical mutation", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_PROFILE_J", "requirement": "Use native Ruckig Profile.j as jerk truth", "depends_on": ["D58_NATIVE_RUCKIG"], "status": "VERIFIED_DONE", "verification_type": "7004/7004 successful Profile.j replays per candidate; D50 known answer 8808/8808", "evidence_paths": ["results/D58/profile_j_stateful_auto0/summary.json", "results/D58/profile_j_stateful_auto1/summary.json", "results/D58/oracle_replay_d50_baseline_profile_only/summary.json"], "result_summary": "Maximum native analytic jerk 8 rad/s3 and ratio 1.0 for both candidates.", "baseline_impact": "measurement contract clarified; no baseline mutation", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_JOINT_LIMITS_FK_GEOMETRY", "requirement": "Validate joint limits, finite state, FK, world/self collision and task observables", "depends_on": ["D58_NATIVE_RUCKIG"], "status": "VERIFIED_DONE", "verification_type": "native MoveIt2/FK metrics", "evidence_paths": ["results/D58/full_stateful_metrics_auto0/metrics.json", "results/D58/full_stateful_metrics_auto1/metrics.json"], "result_summary": "Finite and zero adaptive-discrete collision cases for both candidates.", "baseline_impact": "shadow only", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_CONTINUOUS_SELF_CERT", "requirement": "Certify exact articulated FK(q(t)) self collision", "depends_on": ["D58_NATIVE_RUCKIG"], "status": "PARTIAL", "verification_type": "FK-aware adaptive certificate plus dense stride-1 rerun", "evidence_paths": ["results/D58/articulated_cert_stateful_auto0_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json", "results/D58/articulated_cert_stateful_auto1_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"], "result_summary": "Both remain UNRESOLVED with 152 unresolved regions; zero collision regions.", "baseline_impact": "no promotion", "next_action_if_incomplete": "Independent agent must revalidate exact articulated CCD architecture before new optimization.", "authoritative": True},
            {"work_unit_id": "D58_FCL_ROUTE_A", "requirement": "Run an independent continuous collision cross-check", "depends_on": ["D58_NATIVE_RUCKIG"], "status": "VERIFIED_DONE", "verification_type": "FCL 0.7 continuousCollide full interval sweep", "evidence_paths": ["results/D58/fcl_routeA_stateful_auto0/continuous_self_collision_summary.json", "results/D58/fcl_routeA_stateful_auto1/continuous_self_collision_summary.json"], "result_summary": "7004 intervals and 10995 swept pair calls per candidate; zero contacts/API errors.", "baseline_impact": "qualified cross-check only; no exact articulated claim", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_ROBUST_CLEARANCE", "requirement": "Replace positive clearance with uncertainty-aware margin", "depends_on": ["D58_JOINT_LIMITS_FK_GEOMETRY"], "status": "PARTIAL", "verification_type": "model clearance plus specified-repeatability sensitivity", "evidence_paths": ["results/D58/D58_ROBUST_CLEARANCE.json", "results/D58/D58_ROBUST_CLEARANCE.md"], "result_summary": "Framework created; total uncertainty and physical threshold unresolved.", "baseline_impact": "no physical safety claim", "next_action_if_incomplete": "Obtain calibration/encoder/geometry/compliance evidence or retain blocked status.", "authoritative": True},
            {"work_unit_id": "D58_SINGULARITY", "requirement": "Measure or formalize task-relevant singularity robustness", "depends_on": ["D58_JOINT_LIMITS_FK_GEOMETRY"], "status": "PARTIAL", "verification_type": "native Jacobian SVD observables", "evidence_paths": ["results/D58/full_stateful_metrics_auto0/metrics.json", "results/D58/full_stateful_metrics_auto1/metrics.json"], "result_summary": "Finite sigma/condition metrics; no universal acceptance threshold.", "baseline_impact": "shadow only", "next_action_if_incomplete": "Independently validate a task-relevant threshold.", "authoritative": True},
            {"work_unit_id": "D58_TASK_ACCURACY", "requirement": "Formalize task accuracy acceptance", "depends_on": ["D58_JOINT_LIMITS_FK_GEOMETRY"], "status": "PARTIAL", "verification_type": "exact source-path Cartesian metrics", "evidence_paths": ["results/D58/full_stateful_metrics_auto0/metrics.json", "results/D58/full_stateful_metrics_auto1/metrics.json"], "result_summary": "Finite path/terminal metrics; calibrated threshold unresolved.", "baseline_impact": "no threshold relaxation", "next_action_if_incomplete": "Use calibrated TCP/task acceptance data.", "authoritative": True},
            {"work_unit_id": "D58_DYNAMICS_TORQUE", "requirement": "Validate inverse dynamics, torque and torque slew", "depends_on": ["D58_NATIVE_RUCKIG"], "status": "VERIFIED_DONE", "verification_type": "Pinocchio RNEA+ABA model audit", "evidence_paths": ["results/D58/dynamics_stateful_auto_pair/native_dynamics_report.json"], "result_summary": "PASS for model consistency; hardware torque unavailable.", "baseline_impact": "post-hoc audit only", "next_action_if_incomplete": "Hardware/current data or validated dynamics-constrained generator.", "authoritative": True},
            {"work_unit_id": "D58_TORQUE_AWARE_GENERATION", "requirement": "Make torque influence trajectory generation", "depends_on": ["D58_DYNAMICS_TORQUE"], "status": "NOT_STARTED", "verification_type": "no accepted upstream torque-constrained generator", "evidence_paths": ["results/D57/D57_FINAL_STATUS.json", "results/D58/D58_UNRESOLVED_ITEMS.json"], "result_summary": "Torque remains a post-hoc model audit.", "baseline_impact": "no unsupported generator promoted", "next_action_if_incomplete": "Design and validate an upstream torque-aware shadow.", "authoritative": True},
            {"work_unit_id": "D58_ADVERSARIAL_FULL_CHAIN", "requirement": "Upgrade adversarial probes to executable full-chain robustness", "depends_on": ["D58_NATIVE_RUCKIG"], "status": "PARTIAL", "verification_type": "2 selected realized shadows plus inherited 64/64 q-only probes", "evidence_paths": ["results/D58/D58_ADVERSARIAL_ROBUSTNESS.json", "results/D57/D57_FINAL_REPORT.md"], "result_summary": "Selected full-chain coverage completed; broader perturbation campaign remains.", "baseline_impact": "no promotion", "next_action_if_incomplete": "Run broader perturbation families through the same native chain.", "authoritative": True},
            {"work_unit_id": "D58_DETERMINISM", "requirement": "Prove deterministic native replay", "depends_on": ["D58_NATIVE_RUCKIG"], "status": "VERIFIED_DONE", "verification_type": "independent rerun byte-identical trajectory hashes", "evidence_paths": ["results/D58/D58_DETERMINISTIC_REPLAY.json"], "result_summary": "Both candidates replayed byte-identically.", "baseline_impact": "no baseline mutation", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_REGRESSION", "requirement": "Protect previously correct measurement capabilities", "depends_on": ["D58_PROFILE_J"], "status": "VERIFIED_DONE", "verification_type": "12 focused pytest tests and D50 known-answer replay", "evidence_paths": ["results/D58/D58_REGRESSION.json", "evidence/test_results.txt"], "result_summary": "12 passed; no failure; protected floor not modified.", "baseline_impact": "non-regressive tooling evidence", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_FLOOR_COMPARISON", "requirement": "Compare against protected floor", "depends_on": ["D58_NATIVE_RUCKIG", "D58_DYNAMICS_TORQUE"], "status": "VERIFIED_DONE", "verification_type": "D58 system comparison and D56 status", "evidence_paths": ["results/D58/D58_SYSTEM_COMPARISON.json", "results/D56/D56_FINAL_STATUS.json"], "result_summary": "No verified candidate net gain cleared all gates.", "baseline_impact": "floor retained", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_PROTECTED_FLOOR_PROMOTION", "requirement": "Promote only a verified net improvement", "depends_on": ["D58_FLOOR_COMPARISON"], "status": "VERIFIED_DONE", "verification_type": "promotion decision in final status", "evidence_paths": ["results/D58/D58_FINAL_STATUS.json"], "result_summary": "NO_PROMOTION.", "baseline_impact": "protected champion retained", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_CANONICAL_PROMOTION", "requirement": "Advance canonical only with all requirements", "depends_on": ["D58_PROTECTED_FLOOR_PROMOTION"], "status": "VERIFIED_DONE", "verification_type": "canonical/promotion records", "evidence_paths": ["results/D58/D58_FINAL_STATUS.json", "results/D47/PROMOTION_RECORD.json"], "result_summary": "Canonical B3 unchanged.", "baseline_impact": "canonical protected", "next_action_if_incomplete": None, "authoritative": True},
            {"work_unit_id": "D58_HARDWARE_BOUNDARIES", "requirement": "Identify requirements needing physical measurements", "depends_on": [], "status": "BLOCKED_EXTERNAL", "verification_type": "explicit unresolved ledger", "evidence_paths": ["results/D58/D58_UNRESOLVED_ITEMS.json", "results/D58/D58_ROBUST_CLEARANCE.json"], "result_summary": "Total uncertainty, hardware TCP/torque/current and physical clearance remain unavailable.", "baseline_impact": "no physical claim", "next_action_if_incomplete": "Collect hardware/calibration evidence.", "authoritative": True},
        ],
    }
    write(stage / "STATE_LEDGER.json", json.dumps(state_ledger, indent=2, ensure_ascii=False) + "\n")

    authoritative_state = {
        "schema_version": "d58-authoritative-state-v1",
        "stage": "D58",
        "task_status": d58_status["task_status"],
        "measurement_pipeline_status": d58_status["measurement_pipeline_status"],
        "canonical": {"identity": d58_comparison["protected_canonical"], "status": "CANONICAL_UNCHANGED"},
        "protected_floor": {"identity": d56_status["final_champion"], "evidence": "results/D56/D56_FINAL_STATUS.json", "status": "PROTECTED_FLOOR_UNCHANGED"},
        "verified_champion": {"identity": d56_status["final_champion"], "stage": "D56", "status": "VERIFIED_CHAMPION"},
        "best_shadow": {"identity": ["D58 stateful_consistent_scale001_auto0", "D58 stateful_consistent_scale001_auto1"], "status": "BEST_UNPROMOTED_SHADOW", "promotion": "NO_PROMOTION"},
        "failed_candidate": {"identity": "D58 generic q-only TOTG scale001/scale0025 route", "status": "NON_AUTHORITATIVE_SHADOW"},
        "inherited_baseline": {"d56_status": d56_status["task_status"], "d57_status": d57_status["status"], "protected_canonical": d57_status["protected_canonical"]},
        "final_accepted_state": "D56 protected champion retained; no D58 candidate accepted",
        "principal_verified_metrics": {
            "d58_profile_j_auto0": phase1["authoritative_profile_j"]["adversarial_0100"],
            "d58_profile_j_auto1": phase1["authoritative_profile_j"]["adversarial_0101"],
            "d58_fcl_auto0": phase1["route_a_fcl_cross_check"]["auto0"],
            "d58_fcl_auto1": phase1["route_a_fcl_cross_check"]["auto1"],
            "d58_model_clearance_status": robust["status"],
            "d56_certified_self_clearance_m": d56_status["clearance"]["minimum_after_m"],
        },
        "principal_hard_gate_status": {"d56_protected_floor": "PASS", "d58_native_execution": "PASS_2_OF_2", "d58_articulated_certificate": "UNRESOLVED_2_OF_2", "d58_promotion": "NO_PROMOTION"},
        "known_unresolved_items": load(OUT / "D58_UNRESOLVED_ITEMS.json"),
        "hardware_only_unresolved_items": ["total geometric uncertainty", "calibrated TCP accuracy", "encoder/backlash/compliance/tracking", "physical clearance threshold", "actuator torque/current certification"],
        "authoritative_evidence": ["results/D58/D58_FINAL_STATUS.json", "results/D58/D58_SYSTEM_COMPARISON.json", "results/D58/D58_CANDIDATE_LEDGER.json", "results/D58/D58_CERTIFICATION_MATRIX.json"],
    }
    write(stage / "AUTHORITATIVE_STATE.json", json.dumps(authoritative_state, indent=2, ensure_ascii=False) + "\n")

    user_prompts = (
        "# User Prompt 01\n\n"
        "刚刚任务中断了，继续完成这个任务，检查1现在。的进度如何？然后在原有的基础上继续完成  这是上一版本的相关记录：\n\n"
        "The earlier user message also supplied the D58 task text below as a user-provided attachment. It is preserved verbatim.\n\n"
        + previous_d58_text
        + "\n\n# User Prompt 02\n\n"
        + current_handoff_text
        + "\n\n# Accessibility note\n\n"
        "Only the user messages and user-provided D58 instruction attachments accessible in this continuation are preserved here. Inaccessible earlier conversation messages are not fabricated. The separate previous-continuation attachment contained assistant history and is not treated as a user prompt.\n"
    )
    write(stage / "USER_PROMPTS.md", user_prompts)

    write(
        stage / "USER_REQUIREMENT_LEDGER.md",
        """# User Requirement Ledger

| ID | Requirement | Source | Current status | Evidence/handling |
|---|---|---|---|---|
| UR-01 | Keep the invariant FR5 Stage 0/1 ON-state open-arch robot-system goal authoritative. | Prompt 02, frozen ROOT GOAL | ACTIVE | `ROOT_GOAL.md`; no benchmark-only promotion. |
| UR-02 | Complete D58 closure with honest native Ruckig, FK, geometry, certificate, clearance, singularity, task, dynamics, robustness, replay and regression evidence. | Prompt 01 D58 text | CLARIFIED | `results/D58/D58_FINAL_REPORT.md` and `STATE_LEDGER.json`. |
| UR-03 | Preserve hard safety/executability gates and do not optimize away Type-B weaknesses. | Prompt 01 D58 text | COMPLETED | `results/D58/D58_FINAL_STATUS.json`; `promotion=NO_PROMOTION`. |
| UR-04 | Distinguish canonical, protected floor, verified champion and shadow candidates. | Prompt 02 sections 3, 7, 9 | COMPLETED | `AUTHORITATIVE_STATE.json`; `AUTHORITY_ORDER.md`. |
| UR-05 | Preserve all accessible user prompts verbatim and do not fabricate inaccessible history. | Prompt 02 section 5 | COMPLETED | `USER_PROMPTS.md`. |
| UR-06 | Create one final <=20 MB `.tar.xz` on Desktop and independently validate it. | Prompt 02 sections 24–27 | ACTIVE | Archive validation is performed after packaging; final response reports exact result. |
| UR-07 | Receiving Agent must reorient from ROOT_GOAL and authoritative state before new science. | Prompt 02 sections 14–17, 28 | COMPLETED | `TAKEOVER_REORIENTATION.md`, `HANDOFF.md`. |
""",
    )

    write(
        stage / "VERIFIED_FACTS.md",
        """# Verified Facts

## Verified by repository execution

- D58 native stateful post-Ruckig execution passed for both selected shadows; both have `native_post_ruckig=true`, clean exits, and exact q-path preservation. Evidence: `results/D58/stateful_consistent_scale001_auto0/execution_form_summary.json`, `results/D58/stateful_consistent_scale001_auto1/execution_form_summary.json`.
- Native Ruckig `Profile.j` replay passed 7004/7004 segments per shadow with maximum analytic jerk `8 rad/s3` and no invalid-input errors. Evidence: `results/D58/profile_j_stateful_auto0/summary.json`, `results/D58/profile_j_stateful_auto1/summary.json`.
- D50 known-answer Profile.j replay passed 8808/8808 segments. Evidence: `results/D58/oracle_replay_d50_baseline_profile_only/summary.json`.
- Native MoveIt2/FK geometry metrics are finite with zero adaptive-discrete environment/self collision cases for both candidates. Evidence: `results/D58/full_stateful_metrics_auto0/metrics.json`, `results/D58/full_stateful_metrics_auto1/metrics.json`.
- FCL Route A completed 7004 intervals and 10995 swept pair calls per candidate, with zero contacts and zero API errors. Evidence: `results/D58/fcl_routeA_stateful_auto0/continuous_self_collision_summary.json`, `results/D58/fcl_routeA_stateful_auto1/continuous_self_collision_summary.json`.
- The FK-aware articulated certificate remains `UNRESOLVED` with 152 unresolved regions per candidate, including dense initial stride 1 reruns. Evidence: `results/D58/articulated_cert_stateful_auto0_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json`, `results/D58/articulated_cert_stateful_auto1_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json`.
- Pinocchio RNEA+ABA model dynamics checks passed. Evidence: `results/D58/dynamics_stateful_auto_pair/native_dynamics_report.json`.
- Both independent native replays are byte-identical. Evidence: `results/D58/D58_DETERMINISTIC_REPLAY.json`.
- Focused regression passed 12 tests. Evidence: `results/D58/D58_REGRESSION.json`, `evidence/test_results.txt`.

## Model-derived

- Clearance, singularity, task error and torque values are software/model observables, not hardware certification. See `results/D58/D58_ROBUST_CLEARANCE.json` and `results/D58/D58_SYSTEM_COMPARISON.json`.
- FCL Route A is a qualified rigid endpoint-sweep cross-check, not exact nonlinear articulated `FK(q(t))` self-CCD.

## External specification

- FAIRINO's FR5 specification lists repeatability of ±0.02 mm. It is used only as one contributor in a sensitivity analysis: `results/D58/D58_ROBUST_CLEARANCE.json`.

## Unknown / hardware required

- Total geometric uncertainty, calibration, encoder/state uncertainty, backlash, compliance, payload deflection, tracking error, physical clearance threshold, calibrated TCP acceptance, and hardware torque/current limits remain unresolved.
""",
    )

    write(
        stage / "EVIDENCE_INDEX.md",
        """# Evidence Index

Claim: both D57 shadows completed native post-Ruckig.
Verifier: stateful D58 ROS launch and native execution summary.
Result: PASS 2/2.
Evidence: `results/D58/stateful_consistent_scale001_auto0/execution_form_summary.json`; `results/D58/stateful_consistent_scale001_auto1/execution_form_summary.json`.

Claim: Profile.j is the jerk truth.
Verifier: native Ruckig boundary replay plus D50 known-answer.
Result: PASS 7004/7004 per D58 candidate; PASS 8808/8808 known answer.
Evidence: `results/D58/profile_j_stateful_auto0/summary.json`; `results/D58/profile_j_stateful_auto1/summary.json`; `results/D58/oracle_replay_d50_baseline_profile_only/summary.json`.

Claim: native FK/geometry is finite and collision-free under the repository label.
Verifier: native MoveIt2/FK metrics.
Result: PASS for the measured domains; collision label remains `adaptive_discrete_interpolation`.
Evidence: `results/D58/full_stateful_metrics_auto0/metrics.json`; `results/D58/full_stateful_metrics_auto1/metrics.json`.

Claim: exact articulated self-CCD is closed.
Verifier: FK-aware certificate and FCL Route A.
Result: NOT PASS. Conservative certificate is `UNRESOLVED/152` per case; FCL is a qualified endpoint-sweep cross-check only.
Evidence: `results/D58/articulated_cert_stateful_auto0_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json`; `results/D58/fcl_routeA_stateful_auto0/continuous_self_collision_summary.json`.

Claim: robust physical clearance is closed.
Verifier: uncertainty-aware sensitivity report.
Result: UNRESOLVED because total uncertainty and physical threshold are missing.
Evidence: `results/D58/D58_ROBUST_CLEARANCE.json`.

Claim: protected floor advanced.
Verifier: D58 final status and comparison.
Result: NO.
Evidence: `results/D58/D58_FINAL_STATUS.json`; `results/D58/D58_SYSTEM_COMPARISON.json`.

Claim: tool repairs were non-regressive.
Verifier: focused pytest command and D50 known answer.
Result: 12 passed; D50 8808/8808.
Evidence: `results/D58/D58_REGRESSION.json`; `evidence/test_results.txt`.
""",
    )

    write(
        stage / "FAILED_ATTEMPTS.md",
        """# Failed and Non-Authoritative Attempts

All entries below are `NON_AUTHORITATIVE_HISTORY` unless explicitly marked otherwise.

## D58 generic q-only TOTG route — FAILED_REVERTED_FROM_ACCEPTANCE

- Intended purpose: pass D57 q-only shadows through a generic native TOTG/Ruckig path.
- Failure: finite-difference jerk diagnostics flagged violations and certificate input/time continuity was not trustworthy.
- Lesson: q-only input is insufficient for articulated certification and FD jerk is not the authoritative Ruckig jerk truth.
- Handling: retained only as diagnostic history; stateful consistent-time repair was used for accepted D58 shadow evidence.

## D58 scalar velocity-limit contract — TYPE-A FIXED

- Intended purpose: replay Ruckig with runtime limits.
- Failure: a scalar `0.0315` was incorrectly applied to all six joints; native MoveIt limits are `[0.0315,0.0315,0.0315,0.032,0.032,0.032]` for the selected scaling.
- Repair: per-joint limit parsing and known-answer regression.
- Evidence: `changed_files/tools/d58_ruckig_profile_oracle.cpp`; `results/D58/oracle_replay_d50_baseline_profile_only/summary.json`.

## D58 initial replay environment — TYPE-A FIXED

- Failure: ROS launch could not find `fairino5_v6_moveit2_config` until `/mnt/d/robotfucker/install/setup.bash` was sourced.
- Repair: replay commands now load the installed repository ROS environment; both reruns completed cleanly.
- Evidence: `results/D58/D58_DETERMINISTIC_REPLAY.json`.

## FK-aware certificate dense rerun — PARTIAL, NOT A TOOL CRASH

- Intended purpose: determine whether 152 unresolved regions were caused by sparse initial sampling.
- Result: `initial_stride=1` and `max_depth=12` still returned `UNRESOLVED/152` for both candidates with zero collision regions.
- Handling: unresolved remains a hard promotion blocker; no false PASS was created.

## FCL Route A — QUALIFIED CROSS-CHECK ONLY

- Result: zero contacts and zero API errors for both candidates.
- Limitation: rigid endpoint sweeps do not prove exact nonlinear articulated `FK(q(t))` equivalence.
- Handling: preserved as an independent cross-check, not labeled exact articulated self-CCD.
""",
    )

    write(
        stage / "GOAL_STATE_GAP.md",
        """# Goal–State Gap

| Root-goal dimension | Status | Evidence / gap |
|---|---|---|
| Motion quality | PARTIAL | Native shadow is executable, but its repaired duration is long; no D58 promotion. |
| Trajectory quality | STRONGLY_VERIFIED for selected shadows | Native Ruckig and Profile.j pass; broader robustness remains. |
| Discrete collision safety | STRONGLY_VERIFIED for measured cases | Native metrics show zero cases under `adaptive_discrete_interpolation`. |
| Continuous self-collision | OPEN | FK-aware certificate is unresolved; FCL is only a qualified cross-check. |
| Clearance robustness | OPEN | Positive model clearance exists, but total uncertainty and threshold are unresolved. |
| Singularity robustness | PARTIAL | Native sigma/condition metrics are finite and improved on selected shadows; no universal threshold. |
| Task accuracy | PARTIAL | Exact source-path metrics are finite; calibrated acceptance threshold is unresolved. |
| Dynamics | STRONGLY_VERIFIED as model audit | Pinocchio RNEA/ABA consistency passes. |
| Torque / torque slew | PARTIAL | Model-based post-hoc audit passes; hardware and upstream torque-aware generation remain open. |
| Jerk | STRONGLY_VERIFIED for native Profile.j semantics | Profile.j replay passes; FD values remain diagnostic. |
| Adversarial robustness | PARTIAL | Two realized full-chain shadows plus inherited q-only probe coverage. |
| Determinism | STRONGLY_VERIFIED for two reruns | Byte-identical native replay. |
| Regression stability | STRONGLY_VERIFIED for focused tests | 12 tests passed. |
| Hardware validation | BLOCKED_BY_HARDWARE | Calibration, total uncertainty, physical TCP/clearance and actuator limits absent. |

The first genuinely unfinished scientific dependency is a backend-independent, conservative exact-articulated self-collision certificate for the realized `FK(q(t))` trajectory, or an independently verified alternative with equivalent scope. The next Agent must revalidate this from the authority files before acting.
""",
    )

    write(
        stage / "TAKEOVER_REORIENTATION.md",
        """# Takeover Reorientation Protocol

**DO NOT begin new scientific execution immediately after opening the archive.**

First read, in order:

1. `ROOT_GOAL.md`
2. `AUTHORITY_ORDER.md`
3. `AUTHORITATIVE_STATE.json`
4. `STATE_LEDGER.json`
5. `VERIFIED_FACTS.md`
6. `GOAL_STATE_GAP.md`
7. `TAKEOVER_REORIENTATION.md`
8. `HANDOFF.md`

Then independently inspect the referenced repository evidence. Do not infer current state from the previous Agent's prose.

Before any new scientific work, produce this exact structure in the new conversation:

```text
TAKEOVER_REORIENTATION_RESULT

ROOT_GOAL:
...

AUTHORITATIVE_CURRENT_STATE:
...

HISTORICAL_MISALIGNMENTS:
...

STATE_CORRECTIONS_REQUIRED:
...

FIRST_GENUINELY_UNFINISHED_ACTION:
...
```

The receiving Agent must audit whether any degraded canonical, stale trajectory, invalid checkpoint, historical workaround, wrong assumption, or stale completion status remains active. Cognitive recognition of the goal is not state correction. Correct or isolate conflicting state before optimizing.
""",
    )

    write(
        stage / "HANDOFF.md",
        """# D58 Cross-Chat Handoff

**Do not immediately continue scientific work from this summary. First read ROOT_GOAL.md, AUTHORITY_ORDER.md, AUTHORITATIVE_STATE.json, STATE_LEDGER.json, VERIFIED_FACTS.md, GOAL_STATE_GAP.md, and TAKEOVER_REORIENTATION.md.**

## 1. Stage task

D58 was intended to close the D57 post-Ruckig/full-system gap, strengthen collision/clearance/singularity/task/dynamics/robustness evidence, and advance the protected floor only on verified system-level net gain.

## 2. Starting state

D58 inherited the D57 automatic C4 shadows, the D56 verified champion `c4_clearance_adversarial0101_amp005_consistent_0025`, and canonical `outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3`. The D57 shadows lacked realized native post-Ruckig and complete downstream authority.

## 3. User requirements

The frozen root goal and full D58 requirements are preserved in `ROOT_GOAL.md`, `USER_PROMPTS.md`, and `USER_REQUIREMENT_LEDGER.md`. The final cross-chat requirement is one compact validated `.tar.xz` archive, with state and evidence separated from history.

## 4. Work actually executed

Stateful consistent-time shadow repair; native MoveIt2 Ruckig execution; Profile.j replay; native FK/geometry/task metrics; FK-aware certificate at default and dense stride 1; FCL Route A; Pinocchio dynamics; robust-clearance sensitivity; deterministic replay; focused regression; final evidence generation.

## 5. Implementation changes

Relevant shadow-only files are under `changed_files/`: D58 stateful wrapper/launch, Profile.j oracle, phase1 aggregate, robust-clearance report, and final evidence generator. Protected Stage 3/D56/D57 state was not modified.

## 6. Research

FCL, MoveIt, Tesseract, articulated CCD literature and FAIRINO specification references are recorded in `results/D58/D58_RESEARCH_LEDGER.md`.

## 7. Verification

Native Ruckig and Profile.j passed for both shadows; FCL endpoint sweeps found zero contacts; articulated certificate remained `UNRESOLVED/152` per candidate; model dynamics and deterministic replay passed; focused regression was 12/12.

## 8. Promotions

- Canonical: `outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3`.
- Protected/verified D56 champion: `c4_clearance_adversarial0101_amp005_consistent_0025`.
- D58 shadows: `stateful_consistent_scale001_auto0` and `stateful_consistent_scale001_auto1`.
- Promotion: `NO_PROMOTION`.

## 9. Current result

D58 is `PARTIALLY_EXECUTED` with a materially stronger measurement/certification evidence chain, but no new protected floor. This is not a physical robot certification.

## 10. Unresolved issues

Exact articulated self-CCD, total uncertainty/physical clearance, calibrated TCP thresholds, torque-aware generation, hardware torque/current and broader perturbation robustness remain open or external.

## 11. Provisional next action

Revalidate the first unfinished dependency: exact articulated self-collision certification scope and available backend, then run only a bounded experiment that can distinguish a true certificate from another qualified cross-check. This is a recommendation, not authority; reorientation must precede execution.

## 12. Recommended reading order

`ROOT_GOAL.md` → `AUTHORITY_ORDER.md` → `AUTHORITATIVE_STATE.json` → `STATE_LEDGER.json` → `VERIFIED_FACTS.md` → `GOAL_STATE_GAP.md` → `TAKEOVER_REORIENTATION.md` → `HANDOFF.md` → `USER_REQUIREMENT_LEDGER.md` → `USER_PROMPTS.md` → `EVIDENCE_INDEX.md` → D58 final report → relevant source/test files → `FAILED_ATTEMPTS.md` → selected raw/reference logs last.
""",
    )

    # Compact evidence collection. Large hotspot JSONL and unrelated raw trees are intentionally excluded.
    evidence_files = [
        (OUT / "D58_FINAL_REPORT.md", "results/D58/D58_FINAL_REPORT.md"),
        (OUT / "D58_FINAL_STATUS.json", "results/D58/D58_FINAL_STATUS.json"),
        (OUT / "D58_SYSTEM_COMPARISON.json", "results/D58/D58_SYSTEM_COMPARISON.json"),
        (OUT / "D58_CANDIDATE_LEDGER.json", "results/D58/D58_CANDIDATE_LEDGER.json"),
        (OUT / "D58_CERTIFICATION_MATRIX.json", "results/D58/D58_CERTIFICATION_MATRIX.json"),
        (OUT / "D58_DETERMINISTIC_REPLAY.json", "results/D58/D58_DETERMINISTIC_REPLAY.json"),
        (OUT / "D58_REGRESSION.json", "results/D58/D58_REGRESSION.json"),
        (OUT / "D58_ADVERSARIAL_ROBUSTNESS.json", "results/D58/D58_ADVERSARIAL_ROBUSTNESS.json"),
        (OUT / "D58_ROBUST_CLEARANCE.json", "results/D58/D58_ROBUST_CLEARANCE.json"),
        (OUT / "D58_ROBUST_CLEARANCE.md", "results/D58/D58_ROBUST_CLEARANCE.md"),
        (OUT / "D58_UNRESOLVED_ITEMS.json", "results/D58/D58_UNRESOLVED_ITEMS.json"),
        (OUT / "D58_PHASE1_STATEFUL_CLOSURE_REPORT.json", "results/D58/D58_PHASE1_STATEFUL_CLOSURE_REPORT.json"),
        (OUT / "D58_RESEARCH_LEDGER.md", "results/D58/D58_RESEARCH_LEDGER.md"),
        (ROOT / "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/D56_FINAL_STATUS.json", "results/D56/D56_FINAL_STATUS.json"),
        (ROOT / "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/D56_FINAL_REPORT.md", "results/D56/D56_FINAL_REPORT.md"),
        (ROOT / "outputs/D57_STAGE4_ALGORITHMIC_CLOSURE/D57_FINAL_STATUS.json", "results/D57/D57_FINAL_STATUS.json"),
        (ROOT / "outputs/D57_STAGE4_ALGORITHMIC_CLOSURE/D57_FINAL_REPORT.md", "results/D57/D57_FINAL_REPORT.md"),
        (ROOT / "outputs/D57_STAGE4_ALGORITHMIC_CLOSURE/D57_SYSTEM_COMPARISON.json", "results/D57/D57_SYSTEM_COMPARISON.json"),
        (ROOT / "outputs/D57_STAGE4_ALGORITHMIC_CLOSURE/D57_RESEARCH_LEDGER.md", "results/D57/D57_RESEARCH_LEDGER.md"),
        (ROOT / "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3/PROMOTION_RECORD.json", "results/D47/PROMOTION_RECORD.json"),
        (ROOT / "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3/metrics.json", "results/D47/metrics.json"),
        (OUT / "dynamics_stateful_auto_pair/native_dynamics_report.json", "results/D58/dynamics_stateful_auto_pair/native_dynamics_report.json"),
        (OUT / "oracle_replay_d50_baseline_profile_only/summary.json", "results/D58/oracle_replay_d50_baseline_profile_only/summary.json"),
        (OUT / "profile_j_stateful_auto0/summary.json", "results/D58/profile_j_stateful_auto0/summary.json"),
        (OUT / "profile_j_stateful_auto1/summary.json", "results/D58/profile_j_stateful_auto1/summary.json"),
        (OUT / "full_stateful_metrics_auto0/metrics.json", "results/D58/full_stateful_metrics_auto0/metrics.json"),
        (OUT / "full_stateful_metrics_auto1/metrics.json", "results/D58/full_stateful_metrics_auto1/metrics.json"),
        (OUT / "stateful_consistent_scale001_auto0/execution_form_summary.json", "results/D58/stateful_consistent_scale001_auto0/execution_form_summary.json"),
        (OUT / "stateful_consistent_scale001_auto1/execution_form_summary.json", "results/D58/stateful_consistent_scale001_auto1/execution_form_summary.json"),
        (OUT / "stateful_consistent_scale001_auto0/trajectories/adversarial_0100.csv", "results/D58/stateful_consistent_scale001_auto0/trajectories/adversarial_0100.csv"),
        (OUT / "stateful_consistent_scale001_auto1/trajectories/adversarial_0101.csv", "results/D58/stateful_consistent_scale001_auto1/trajectories/adversarial_0101.csv"),
        (OUT / "articulated_cert_stateful_auto0/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json", "results/D58/articulated_cert_stateful_auto0/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"),
        (OUT / "articulated_cert_stateful_auto1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json", "results/D58/articulated_cert_stateful_auto1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"),
        (OUT / "articulated_cert_stateful_auto0_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json", "results/D58/articulated_cert_stateful_auto0_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"),
        (OUT / "articulated_cert_stateful_auto1_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json", "results/D58/articulated_cert_stateful_auto1_stride1/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"),
        (OUT / "articulated_cert_stateful_auto0_stride1/D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl", "results/D58/articulated_cert_stateful_auto0_stride1/D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl"),
        (OUT / "articulated_cert_stateful_auto1_stride1/D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl", "results/D58/articulated_cert_stateful_auto1_stride1/D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl"),
        (OUT / "fcl_routeA_stateful_auto0/continuous_self_collision_summary.json", "results/D58/fcl_routeA_stateful_auto0/continuous_self_collision_summary.json"),
        (OUT / "fcl_routeA_stateful_auto1/continuous_self_collision_summary.json", "results/D58/fcl_routeA_stateful_auto1/continuous_self_collision_summary.json"),
        (OUT / "fcl_routeA_stateful_auto0/continuous_self_collision_backend.json", "results/D58/fcl_routeA_stateful_auto0/continuous_self_collision_backend.json"),
        (OUT / "fcl_routeA_stateful_auto1/continuous_self_collision_backend.json", "results/D58/fcl_routeA_stateful_auto1/continuous_self_collision_backend.json"),
    ]
    for source, destination in evidence_files:
        copy_file(source, stage / destination, copied)

    changed_files = [
        ROOT / "tools/d58_stateful_shadow_native.py",
        ROOT / "tools/d58_ruckig_profile_oracle.cpp",
        ROOT / "tools/d58_candidate_articulated_certificate.py",
        ROOT / "tools/d58_candidate_certification.py",
        ROOT / "tools/d58_phase1_stateful_aggregate.py",
        ROOT / "tools/d58_robust_clearance_report.py",
        ROOT / "tools/d58_finalize_evidence.py",
        ROOT / "ros2_moveit_bridge/launch/d58_stateful_shadow_native.launch.py",
        ROOT / "tests/test_d50_measurement_repairs.py",
        ROOT / "tests/test_d54_articulated_certificate.py",
        ROOT / "tests/test_d57_system_closure.py",
        ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml",
        ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
        ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf",
    ]
    for source in changed_files:
        copy_file(source, stage / "changed_files" / source.relative_to(ROOT), copied)

    write(stage / "evidence/test_results.txt", "Command: python -m pytest -q tests/test_d50_measurement_repairs.py tests/test_d54_articulated_certificate.py tests/test_d57_system_closure.py\nResult: 12 passed in 0.45s\nKnown-answer Profile.j: 8808/8808 PASS.\n")
    write(stage / "evidence/replay_hashes.txt", "auto0 original/replay SHA256: 7BE4AB2F8EF2840F7DAFAF66F72827297779352620D928E38D5CCB1380C4F287\nauto1 original/replay SHA256: 081E703F64F0730F489C212FA316B9393B47E2589CFA152B29CBEBD76DBD2A6A\nBoth pairs byte-identical: PASS\n")
    write(stage / "evidence/archive_scope.txt", "Compact D58 handoff scope. Excluded: profile hotspot JSONL, duplicate replays, legacy raw trees, caches, build outputs, unrelated dirty-worktree artifacts, and hidden reasoning.\n")
    write(stage / "evidence/git_status.txt", git(["git", "status", "--short"]))
    write(stage / "evidence/git_diff_stat.txt", git(["git", "diff", "--stat"]))
    write(stage / "evidence/native_run_contract.txt", "D58 scope: Stage 0/1 ON-state open-arch only. Collision label: adaptive_discrete_interpolation. Continuous self-collision exact articulated status: unresolved/not available. Hardware torque and physical uncertainty: not available.\n")

    write(
        stage / "README.txt",
        "D58 goal-anchored cross-chat handoff archive. Read ROOT_GOAL.md first, then AUTHORITY_ORDER.md, AUTHORITATIVE_STATE.json, STATE_LEDGER.json, VERIFIED_FACTS.md, GOAL_STATE_GAP.md, TAKEOVER_REORIENTATION.md, and HANDOFF.md.\n",
    )
    # Count files, not directory entries.  The field is consumed by the
    # independent archive validator and must have an unambiguous meaning.
    file_count = sum(1 for path in stage.rglob("*") if path.is_file())
    manifest = {"schema_version": "d58-handoff-staging-manifest-v1", "stage": "D58", "created_at": stamp, "file_count": file_count, "copied_evidence_files": copied, "excluded_large_intermediates": ["profile_j_stateful_auto0/hotspots.jsonl", "profile_j_stateful_auto1/hotspots.jsonl", "duplicate replay trajectories", "legacy raw output trees"], "archive_target": "Desktop/D58_CROSS_CHAT_HANDOFF_<timestamp>.tar.xz"}
    write(stage / "STAGING_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": str(stage), "stamp": stamp, "file_count": file_count, "copied_evidence_files": len(copied)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
