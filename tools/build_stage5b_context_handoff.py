"""Build, compress, and verify the single FR5 Stage 5B takeover package."""

from __future__ import annotations

import hashlib
import json
import lzma
import re
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "outputs" / "STAGE5B_HANDOFF_STAGING"
DESKTOP = Path(r"C:\Users\86198\Desktop")
PACKAGE_NAME = "FR5_STAGE5B_HANDOFF"
ATTACHMENTS = [
    Path(r"C:\Users\86198\.codex\attachments\b5c7fdfe-cba2-466f-98ed-b19c952f5ef7\pasted-text.txt"),
    Path(r"C:\Users\86198\.codex\attachments\76e51f62-911c-48fd-b3a7-4d65b6f5ed79\pasted-text.txt"),
    Path(r"C:\Users\86198\.codex\attachments\d35d523e-b99a-4f15-9e1c-738057d9479d\pasted-text.txt"),
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(relative: str, text: str) -> None:
    path = STAGING / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def copy_file(relative: str, source: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = STAGING / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def current_user_prompts() -> str:
    chunks = ["# USER_PROMPTS\n\nONLY CURRENTLY ACCESSIBLE USER MESSAGES ARE INCLUDED.\n"]
    labels = [
        "User Prompt 01 — FR5 Stage 5B-0 + Stage 5B-1 execution task",
        "User Prompt 02 — Stage 5B supplementary execution instruction",
        "User Prompt 03 — direct completion instruction",
        "User Prompt 04 — forced Context Handoff task",
    ]
    for index, label in enumerate(labels):
        chunks.append(f"## {label}\n")
        if index < 2 or index == 3:
            path = ATTACHMENTS[index if index < 2 else 2]
            if path.is_file():
                chunks.append(path.read_text(encoding="utf-8"))
            else:
                chunks.append(f"ATTACHMENT_UNAVAILABLE: {path}")
        else:
            chunks.append("彻底完成我的目标，不完成不准停")
        chunks.append("\n")
    return "\n".join(chunks)


def build_documents(audit: dict, blocker: dict, authority: dict, dense: dict) -> None:
    continuity = audit["trajectory"]["wp66_to_wp67"]
    action = audit["execution"]["action_result"]
    max_error = max(audit["execution"]["controller_error_max_abs_rad_by_joint"])
    source_hash = audit["trajectory"]["source_sha256"]
    root_goal = """# ROOT_GOAL

## Long-term objective

For the real FR5 arm and horseshoe-tunnel complex-surface spraying task, build a complete, research-grade robot-system loop from surface-task definition, TCP pose and orientation planning, MoveIt2/IK, Ruckig time parameterization, FK, collision and continuity verification, dynamics, ROS 2/ros2_control/Gazebo execution, uncertainty experiments, Sim-to-Real, and final FR5 hardware validation.

The object being improved is the real robot motion and spraying system: task validity, safety, stability, robustness, executability, repeatability, and scientific credibility. It is not a benchmark number or a single offline metric.

## Current stage

Stage5AC is closed as historical context. The active stage is **Stage5B**:

1. Stage5B-0: close the wp66 -> wp67 IK branch discontinuity and establish the software nominal-world authority without inventing a physical transform.
2. Stage5B-1: bring up FR5 -> MoveIt2 -> ros2_control -> JointTrajectoryController -> Gazebo Sim -> simulator joint state/TF/TCP.
3. Stage5B-2: execute the complete complex-surface trajectory.
4. Stage5B-3: verify tracking, TF, TCP, collision/contact, dynamics, replay, and trajectory integrity.

Do not create Stage5A-D/E or drift back to endless optimization of 99.x offline metrics.

## Non-negotiable rules

- No physical TCP calibration, fixture calibration, real base alignment, physical clearance, hardware dynamics, or hardware safety may be written as PASS without measurements; use `UNVERIFIED`.
- The identity `base_link` scene is the nominal software simulation authority. `x=-0.50 m` is a shadow hypothesis only; collision-free shadow behavior cannot establish its physical correctness.
- Do not move the nominal world merely to make a test pass.
- Preserve D65, Stage5AR, Stage5AC, the protected D47 canonical, and the protected floor. Experiments belong in isolated shadow/candidate space.
- Do not hide or postpone an actionable Stage5B defect. Repair measurement infrastructure aggressively, but do not optimize away a real robot-system weakness.
- `adaptive_discrete_interpolation` is not strict continuous collision detection. Missing exact articulated CCD, clearance, and physical authority stay `not_available`/`UNVERIFIED`.
- A controller action result of 0 is not by itself a continuity, collision, tracking, or safety PASS.
"""
    write_text("ROOT_GOAL.md", root_goal)

    state = f"""# AUTHORITATIVE_STATE

## LAST VERIFIED STAGE

`D65` software/model closure remains verified historical authority with `NO_PROMOTION`.

## CURRENT STAGE

`STAGE5B` — this handoff records Stage5B-0/1 progress and the bounded Stage5B-2/3 execution evidence.

## CURRENT BASELINE

Protected canonical: `outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3`.

Protected floor: `c4_clearance_adversarial0101_amp005_consistent_0025`.

No canonical or protected baseline was modified. Promotion status is `NO_PROMOTION`.

## CURRENT SIMULATION INPUT

The candidate replay source is the 181-row `STAGE5AC_CANDIDATE_TRAJECTORY.csv`, SHA256 `{source_hash}`, joint order `j1..j6`, duration `{audit['trajectory']['duration_s']:.9f} s`, with q/dq/ddq and timestamps preserved. It is not an accepted Stage5B trajectory.

## CURRENT WORLD AUTHORITY

Nominal software authority is the identity `base_link` scene generated from the authoritative 181-point ON-state open-arch pose/seed pair, with 0.260 m stand-off, 0.025 m wall thickness, 1.10 m transverse extent, stride 1, open path, and no floor. Physical installation alignment is `UNVERIFIED`.

`x=-0.50 m` remains `SHADOW_ONLY_UNVERIFIED_NOT_PROMOTED`.

## VERIFIED STAGE5B RESULTS

- Gazebo Sim + `gz_ros2_control` + controller manager + JTC + joint-state broadcaster: operational.
- x=-0.50 shadow: one complete 181-point FJT goal accepted and completed with result code 0; 70,565 formal controller/joint-state samples captured.
- The same replay exposed a maximum reference-to-feedback error of `{max_error:.9f} rad`; the candidate is rejected.
- Nominal full physics replay was blocked by real wall-contact slowdown: pre-goal drift `{blocker['initial_state']['measured_max_pre_goal_drift_rad']:.9f} rad`; a 12 s probe advanced Gazebo clock only `{blocker['execution_observation']['clock_sample_s']:.2f} s`.

## CURRENT BLOCKERS

1. wp66 -> wp67 remains `{continuity['wrapped_max_abs_deg']:.9f} deg` wrap-aware versus the 20 deg gate; one interval fails.
2. 72 dense fixed-endpoint, 128-subdivision detour experiments remain `UNRESOLVED`; best bottleneck `{min(float(x['minimum_bottleneck_step_deg']) for x in dense['experiments'] if x.get('minimum_bottleneck_step_deg') is not None):.9f} deg`.
3. q/dq/ddq/timestamp input is incompatible with the observed native JTC spline reference in the branch-cut region.
4. Nominal contact physics is too slow for bounded full replay and cannot be bypassed by changing the nominal scene.

## PHYSICAL-DOMAIN UNVERIFIED ITEMS

Physical TCP calibration, fixture/base transform, physical clearance, actuator current/torque/thermal data, hardware execution, and exact external nonlinear articulated self-CCD.

## NEXT EXACT ACTION

Construct a fixed-endpoint task-space/IK route through wp66 -> wp67 with adjacent joint step <=20 deg, then jointly regenerate q/dq/ddq/timestamps and re-run the full software validation before any promotion.
"""
    write_text("AUTHORITATIVE_STATE.md", state)

    requirements = """# CURRENT_USER_REQUIREMENTS

## Active requirements

- The active phase is Stage5B. Do not invent Stage5A-D/E subphases or return to benchmark-only optimization.
- Finish Stage5B-0 first: close the known wp66 -> wp67 approximately 96.52 degree joint-space branch discontinuity and settle nominal-world versus x=-0.50 shadow authority.
- Then execute the real software chain through MoveIt2, ros2_control, JointTrajectoryController, Gazebo Sim, simulator state, TF, and TCP, followed by a full 181-point trajectory replay and Stage5B-3 verification.
- Progress is preferred, including combined IK/densification/planner experiments, but no fake PASS, silent workaround, arbitrary world movement, candidate promotion, or baseline regression is allowed.
- Hardware/vendor execution is not required to establish the software simulation loop, but all physical claims remain `UNVERIFIED` until measured.
- Preserve authoritative input identity, source mapping, q/dq/ddq/timestamps, and explicit evidence boundaries. A full action result is not sufficient for continuity or safety.
- Create one compact, self-contained, hash-verified Stage5B Context Handoff archive for a new ChatGPT window. The archive must contain independent root goal, authority hierarchy, current state, user requirements, state tracker, completion gate, evidence index, decisions, failed attempts, historical audit, takeover protocol, manifest, and sensitive-information scan.

## Precedence

The latest Context Handoff request adds the packaging and takeover requirements; it does not cancel the earlier Stage5B execution requirements. Earlier Stage5AC/D65 material is historical/protected context, not a reason to skip Stage5B verification. Current machine-readable state and verifier evidence outrank prose summaries and raw logs.
"""
    write_text("CURRENT_USER_REQUIREMENTS.md", requirements)
    write_text("USER_PROMPTS.md", current_user_prompts())

    context_authority = """# CONTEXT_AUTHORITY

## Level 0 — ROOT

`ROOT_GOAL.md` is the highest authority for what the project is trying to do. Historical Agent behavior cannot override it.

## Level 1 — CURRENT VERIFIED STATE

`AUTHORITATIVE_STATE.md`, `STATE_TRACKER.json`, `COMPLETION_GATE.json`, protected baseline identity, and current verifier-backed results determine where the project actually is.

## Level 2 — VERIFIED EVIDENCE

Evidence includes the Stage5B execution audit, FJT action result, input hashes, world authority record, dense-search JSON, controller/Gazebo configuration, build result, and raw-log metadata. Each claim keeps its scope.

## Level 3 — DECISIONS / FAILED ATTEMPTS

`DECISION_LOG.md` and `FAILED_ATTEMPTS.md` explain why current methods and boundaries were chosen. They cannot override Levels 0–2.

## Level 4 — RAW REFERENCE

Large raw JSONL logs, shell output, old reports, and historical transcripts are `REFERENCE_ONLY`. They are not allowed to redefine goal, authority, or PASS status.

## Interpretation rule

Do not inherit the previous Agent's direction. Reconstruct ROOT, state, and evidence independently, then plan the next action.
"""
    write_text("CONTEXT_AUTHORITY.md", context_authority)

    handoff = f"""# HANDOFF — FR5 Stage5B Context Takeover

## READ ORDER

1. `ROOT_GOAL.md`
2. `AUTHORITATIVE_STATE.md`
3. `CURRENT_USER_REQUIREMENTS.md`
4. `STATE_TRACKER.json`
5. `COMPLETION_GATE.json`
6. `HANDOFF.md`
7. `USER_PROMPTS.md`
8. `EVIDENCE_INDEX.md`
9. `DECISION_LOG.md`
10. `FAILED_ATTEMPTS.md`
11. `HISTORICAL_STATE_AUDIT.md`
12. `TAKEOVER_PROTOCOL.md`
13. selected results and changed files

## Actual result

`STAGE5B_STATUS = BLOCKED_EXACT_CONTINUITY_AND_NOMINAL_PHYSICS`

The software control loop was brought up and the x=-0.50 shadow completed a full FJT action. The trajectory itself is not accepted: wp66 -> wp67 is `{continuity['wrapped_max_abs_deg']:.6f} deg` wrap-aware against a 20 deg gate, and native JTC spline replay produced a maximum reference-to-feedback error of `{max_error:.6f} rad`. The nominal identity world remains software-only authority; its physics replay hit real contact slowdown and was safely stopped.

## What is authoritative

- Long-term direction: `ROOT_GOAL.md`.
- Current Stage5B state: `AUTHORITATIVE_STATE.md` and `STATE_TRACKER.json`.
- Gate decisions: `COMPLETION_GATE.json`.
- Nominal world identity: `results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json`.
- Current x=-0.50 replay evidence: `results/STAGE5B_EXECUTION_AUDIT.json` and `results/x_minus_050/`.

## What is not authoritative

The historical Stage5AC candidate, x=-0.50 layout, action result alone, endpoint FCL/Bullet screens, adaptive discrete collision result, and any physical/hardware interpretation do not promote a trajectory or establish safety.

## No regression

The D47 canonical, protected floor, D64/D65 lineage, and earlier Stage5AR/Stage5AC records were not overwritten. New work is shadow/diagnostic only.

## Exact continuation

Search for a fixed-endpoint <=20 degree IK/task-space path through wp66 -> wp67; after that regenerate derivatives/timestamps jointly and re-run the full FJT and Stage5B-3 audit. Do not close the gate based on this package.
"""
    write_text("HANDOFF.md", handoff)

    protocol = """# TAKEOVER_PROTOCOL

## Mandatory reorientation before code changes

### Q1 — What is the root goal?

Answer only from `ROOT_GOAL.md`, not from the last Agent message.

### Q2 — What is the real current state?

Answer from `AUTHORITATIVE_STATE.md`, `STATE_TRACKER.json`, `COMPLETION_GATE.json`, the repository, and the selected evidence files. Distinguish PASS scope from FAIL, UNVERIFIED, and BLOCKED.

### Q3 — What conflicts with the root goal?

Check stale candidate promotion, wrong world authority, branch discontinuity, q/dq/ddq/timestamp mismatch, nominal contact slowdown, temporary workarounds, and baseline contamination.

## Cheap read-only verification

Before editing, verify:

```powershell
Get-Content ROOT_GOAL.md
Get-Content AUTHORITATIVE_STATE.md
Get-Content STATE_TRACKER.json
Get-Content COMPLETION_GATE.json
Get-FileHash outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv
Get-FileHash outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv
git status --short
```

Confirm canonical/protected paths and inspect `STAGE5B_EXECUTION_AUDIT.json` before rerunning anything expensive. Never use `git reset`, `git clean`, `git checkout .`, or `git restore .`.

## First useful action after verification

Implement or test one fixed-endpoint <=20 degree IK/task-space route through wp66 -> wp67. Keep the candidate in shadow space, preserve the 181-point source identity, then validate q/dq/ddq/timestamps against native JTC semantics.
"""
    write_text("TAKEOVER_PROTOCOL.md", protocol)

    audit_doc = f"""# HISTORICAL_STATE_AUDIT

## Result

`REPOSITORY_STATE = DIRTY_BY_DESIGN; PROTECTED_STATE_PRESERVED; STAGE5B_SHADOW_ONLY`

The repository contains older D65 root/state documents and extensive historical outputs. Those remain valid for D65 history but do not state that Stage5B is complete. This package is the Stage5B-specific takeover authority.

## Checks performed

- Protected canonical and floor were not modified during Stage5B work.
- Authoritative 181-point pose/seed input hashes match `STAGE5B_NOMINAL_WORLD_AUTHORITY.json`.
- The x=-0.50 environment is labelled shadow-only and has no physical installation authority.
- The candidate trajectory remains under `outputs/STAGE5AC_FINAL_CLOSURE/`; no promotion alias was created.
- The non-fixed-endpoint graph-search PASS is explicitly a diagnostic false partial and is not admissible for promotion.
- The fixed-endpoint dense search is `UNRESOLVED`.
- Gazebo launch/resource-path/controller changes are isolated Stage5B tooling/configuration; they do not change the canonical planning baseline.
- Large raw runtime JSONL logs remain outside the package; their paths, sizes, and hashes are indexed rather than copied.

## Residual drift

The current candidate still has the known branch cut and derivative/controller mismatch. Nominal physics also remains contact-limited. These are visible failures, not stale aliases or silently promoted shadows.
"""
    write_text("HISTORICAL_STATE_AUDIT.md", audit_doc)

    tracker = {
        "schema_version": "stage5b-state-tracker-v1",
        "task_id": "FR5_STAGE5B_20260903",
        "task_name": "Stage5B-0/1 continuity repair, authority settlement, simulation bring-up and full replay",
        "overall_status": "IN_PROGRESS",
        "promotion": "NO_PROMOTION",
        "items": [
            {"task_id": "stage5b-0-input-authority", "task_name": "181-point ON-state input identity", "status": "PASS", "authority": "VERIFIED", "verifier": "input SHA256 comparison", "evidence": ["results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json"], "artifact_path": "results/internal_wiper_moveit_inputs/", "hash": authority["authoritative_inputs"]["pose_csv_sha256"], "blocking_reason": None, "next_action": "preserve identity"},
            {"task_id": "stage5b-0-continuity", "task_name": "wp66 -> wp67 branch repair", "status": "FAIL", "authority": "SHADOW_DIAGNOSTIC", "verifier": "stage5b_audit_execution.py and fixed-endpoint graph search", "evidence": ["results/STAGE5B_EXECUTION_AUDIT.json", "results/dense_branch_search.json"], "artifact_path": "results/STAGE5B_EXECUTION_AUDIT.json", "hash": source_hash, "blocking_reason": f"wrap-aware step {continuity['wrapped_max_abs_deg']:.9f} deg exceeds 20 deg; dense fixed-endpoint search unresolved", "next_action": "find fixed-endpoint <=20 degree IK/task-space path"},
            {"task_id": "stage5b-0-world-authority", "task_name": "nominal identity world and x=-0.50 classification", "status": "PASS", "authority": "SOFTWARE_SIMULATION_ONLY", "verifier": "nominal world authority record", "evidence": ["results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json"], "artifact_path": "results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json", "hash": authority["authoritative_inputs"]["ik_graph_config_sha256"], "blocking_reason": "physical transform remains unverified", "next_action": "retain identity nominal; do not promote x=-0.50"},
            {"task_id": "stage5b-1-bringup", "task_name": "MoveIt2/Gazebo/ros2_control/JTC chain", "status": "PASS", "authority": "VERIFIED_RUNTIME", "verifier": "colcon build plus Gazebo controller lifecycle and joint-state observation", "evidence": ["evidence/bringup_summary.json", "changed_files/ros2_moveit_bridge/launch/stage5b_gazebo.launch.py"], "artifact_path": "changed_files/ros2_moveit_bridge/", "hash": None, "blocking_reason": None, "next_action": "reuse in next candidate replay"},
            {"task_id": "stage5b-2-xminus-replay", "task_name": "full x=-0.50 shadow FJT replay", "status": "PASS", "authority": "EXECUTION_ONLY", "verifier": "FollowJointTrajectory result code 0 with 181 points", "evidence": ["results/x_minus_050/action_execution.json", "results/STAGE5B_EXECUTION_AUDIT.json"], "artifact_path": "results/x_minus_050/", "hash": source_hash, "blocking_reason": "candidate quality and continuity failed", "next_action": "rerun only after candidate repair"},
            {"task_id": "stage5b-2-nominal-replay", "task_name": "full nominal physics replay", "status": "FAIL", "authority": "RUNTIME_DIAGNOSTIC", "verifier": "nominal blocker record and clock probe", "evidence": ["results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json"], "artifact_path": "results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json", "hash": None, "blocking_reason": "wall contact reduced physics progress to 0.67 s per 12 s probe", "next_action": "resolve physical/software collision behavior without moving nominal world"},
            {"task_id": "stage5b-3-tracking", "task_name": "commanded versus simulated q/tracking", "status": "FAIL", "authority": "VERIFIED_RUNTIME_FAILURE", "verifier": "execution audit", "evidence": ["results/STAGE5B_EXECUTION_AUDIT.json"], "artifact_path": "results/STAGE5B_EXECUTION_AUDIT.json", "hash": None, "blocking_reason": f"maximum reference-feedback error {max_error:.9f} rad", "next_action": "jointly regenerate derivatives and timestamps"},
            {"task_id": "stage5b-3-physical", "task_name": "physical TCP/clearance/hardware safety", "status": "UNVERIFIED", "authority": "NONE", "verifier": "not available", "evidence": ["results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json"], "artifact_path": None, "hash": None, "blocking_reason": "no physical calibration or hardware measurement", "next_action": "separate authorized physical campaign"},
            {"task_id": "handoff-package", "task_name": "single context handoff archive", "status": "PASS", "authority": "PACKAGE_VERIFICATION", "verifier": "manifest, tar.xz extraction and sensitive scan", "evidence": ["MANIFEST.json", "SENSITIVE_SCAN.md"], "artifact_path": "MANIFEST.json", "hash": None, "blocking_reason": None, "next_action": "preserve the verified single Desktop archive"},
        ],
    }
    write_text("STATE_TRACKER.json", json.dumps(tracker, indent=2, ensure_ascii=False))

    gate = {
        "schema_version": "stage5b-completion-gate-v1",
        "overall_status": "IN_PROGRESS",
        "promotion": "NO_PROMOTION",
        "status_enum": ["NOT_STARTED", "IN_PROGRESS", "PASS", "FAIL", "BLOCKED_EXTERNAL", "UNVERIFIED", "SUPERSEDED"],
        "stage5b_0": {
            "ik_jump_root_cause": {"status": "FAIL", "verifier": "stage5b_audit_execution.py", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": "branch/limit transition at wp66 -> wp67; source wrap-aware step exceeds gate"},
            "96_52_degree_repair": {"status": "FAIL", "verifier": "dense fixed-endpoint graph search", "evidence": "results/dense_branch_search.json", "detail": "72 experiments, 128 subdivisions, no admissible path"},
            "max_adjacent_joint_delta": {"status": "FAIL", "verifier": "stage5b_audit_execution.py", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": f"{continuity['wrapped_max_abs_deg']:.9f} deg wrap-aware; gate 20 deg"},
            "other_abnormal_branch_jumps": {"status": "PASS", "verifier": "stage5b_audit_execution.py", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": "one source interval over the 20 degree gate"},
            "TCP_FK": {"status": "UNVERIFIED", "verifier": "not available in full replay", "evidence": "results/x_minus_050/tf_capture_manifest.json", "detail": "TF captured; direct measured TCP trajectory audit not completed"},
            "Ruckig": {"status": "UNVERIFIED", "verifier": "historical Stage5AC evidence only", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": "Stage5B q/dq/ddq/timestamp/JTC compatibility failed"},
            "dynamics": {"status": "UNVERIFIED", "verifier": "not available", "evidence": "results/STAGE5B_FINAL_HANDOFF.json", "detail": "no hardware torque/current claim"},
            "Bullet": {"status": "PASS", "verifier": "historical shadow screen", "evidence": "results/STAGE5B_FINAL_HANDOFF.json", "detail": "scoped endpoint/state screen only; not exact articulated CCD"},
            "FCL": {"status": "PASS", "verifier": "historical shadow screen", "evidence": "results/STAGE5B_FINAL_HANDOFF.json", "detail": "scoped discrete state contacts only; not clearance proof"},
            "deterministic_replay": {"status": "PASS", "verifier": "source hash and one-goal execution record", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": "execution reproducible at input identity level; quality gate still fails"},
            "nominal_world_provenance": {"status": "PASS", "verifier": "world authority record", "evidence": "results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json", "detail": "software identity authority only"},
            "physical_authority": {"status": "UNVERIFIED", "verifier": "not available", "evidence": "results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json", "detail": "x=-0.50 has no CAD/calibration/hardware authority"},
        },
        "stage5b_1": {key: {"status": status, "verifier": verifier, "evidence": evidence, "detail": detail} for key, status, verifier, evidence, detail in [
            ("ROS2_distro", "PASS", "runtime", "evidence/bringup_summary.json", "Ubuntu 24.04-D / ROS 2 Jazzy"),
            ("FR5_robot_description", "PASS", "xacro expansion and Gazebo spawn", "changed_files/ros2_moveit_bridge/config/stage5b_gazebo.urdf.xacro", "robot entity created"),
            ("joint_mapping", "PASS", "controller and joint-state logs", "evidence/bringup_summary.json", "j1..j6"),
            ("TF", "PASS", "runtime recorder", "results/x_minus_050/tf_capture_manifest.json", "dynamic TF rows and static TCP edge captured"),
            ("spray_TCP_static_edge", "PASS", "runtime recorder", "results/x_minus_050/tf_capture_manifest.json", "spray_tcp fixed edge present"),
            ("MoveIt2", "PASS", "MoveItPy initialization/search", "results/dense_branch_search.json", "MoveIt model and KDL search ran"),
            ("ros2_control", "PASS", "controller lifecycle", "evidence/bringup_summary.json", "hardware configured/activated"),
            ("controller_manager", "PASS", "lifecycle logs", "evidence/bringup_summary.json", "services responsive after spawner"),
            ("JointTrajectoryController", "PASS", "lifecycle and action result", "results/x_minus_050/action_execution.json", "active and accepted FJT"),
            ("Gazebo_Sim", "PASS", "launch and robot spawn", "evidence/bringup_summary.json", "server-only fixed-step world"),
            ("controller_active", "PASS", "list_controllers", "evidence/bringup_summary.json", "JTC and broadcaster active"),
            ("command_path", "PASS", "FJT action", "results/x_minus_050/action_execution.json", "one action goal accepted/completed"),
            ("simulator_state_feedback", "PASS", "joint-state recorder", "results/x_minus_050/action_execution.json", "70,887 joint-state messages"),
        ]},
        "stage5b_2": {
            "full_trajectory_loaded": {"status": "PASS", "verifier": "goal builder", "evidence": "results/x_minus_050/follow_joint_trajectory_goal.json", "detail": "181 points, q/dq/ddq/timestamps preserved"},
            "full_x_minus_050_trajectory_executed": {"status": "PASS", "verifier": "FJT result", "evidence": "results/x_minus_050/action_execution.json", "detail": "result code 0"},
            "full_nominal_trajectory_executed": {"status": "FAIL", "verifier": "nominal blocker", "evidence": "results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json", "detail": "goal accepted in log but physics did not progress within bounded window"},
            "no_unexpected_abort_x_minus_050": {"status": "PASS", "verifier": "FJT result", "evidence": "results/x_minus_050/action_execution.json", "detail": "no cancellation/preemption/abort"},
            "route_identity_verified": {"status": "PASS", "verifier": "SHA256", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": source_hash},
            "execution_evidence_captured": {"status": "PASS", "verifier": "runtime recorder", "evidence": "results/x_minus_050/", "detail": "controller, joint-state, action, TF files"},
        },
        "stage5b_3": {
            "commanded_vs_simulated_q": {"status": "FAIL", "verifier": "execution audit", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": f"max reference-feedback error {max_error:.9f} rad"},
            "TCP": {"status": "UNVERIFIED", "verifier": "not completed", "evidence": "results/x_minus_050/tf_capture_manifest.json", "detail": "TCP path composition/audit remains open"},
            "tracking_error": {"status": "FAIL", "verifier": "execution audit", "evidence": "results/STAGE5B_EXECUTION_AUDIT.json", "detail": "native spline overshoot"},
            "collision_contact": {"status": "FAIL", "verifier": "nominal runtime diagnostic", "evidence": "results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json", "detail": "real nominal wall contact; shadow contact not exact CCD"},
            "dynamics": {"status": "UNVERIFIED", "verifier": "not available", "evidence": "results/STAGE5B_FINAL_HANDOFF.json", "detail": "no torque/current/hardware domain"},
            "replay": {"status": "PASS", "verifier": "x=-0.50 one-goal replay", "evidence": "results/x_minus_050/action_execution.json", "detail": "software execution replay completed"},
            "known_limitations": {"status": "PASS", "verifier": "claim fence audit", "evidence": "results/STAGE5B_FINAL_HANDOFF.json", "detail": "CCD/clearance/physical limitations explicit"},
        },
    }
    write_text("COMPLETION_GATE.json", json.dumps(gate, indent=2, ensure_ascii=False))

    evidence = f"""# EVIDENCE_INDEX

| Claim | Status | Evidence | Authority / scope |
|---|---|---|---|
| 181-point source identity preserved | PASS | `results/STAGE5B_EXECUTION_AUDIT.json` | verified hash `{source_hash}` |
| wp66 -> wp67 repaired | FAIL | `results/STAGE5B_EXECUTION_AUDIT.json`; `results/dense_branch_search.json` | fixed endpoints; no admissible <=20 degree path |
| x=-0.50 full FJT action completed | PASS | `results/x_minus_050/action_execution.json` | execution only, shadow world |
| x=-0.50 trajectory quality | FAIL | `results/STAGE5B_EXECUTION_AUDIT.json` | max reference-feedback error `{max_error:.6f} rad` |
| nominal identity world defined | PASS | `results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json` | software simulation only |
| x=-0.50 physical authority | UNVERIFIED | `results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json` | no CAD/calibration/hardware evidence |
| nominal full physics replay | FAIL | `results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json` | actual wall contact slowed clock |
| Gazebo/JTC/controller loop | PASS | `evidence/bringup_summary.json` | ROS 2 Jazzy software runtime |
| TF capture | PASS | `results/x_minus_050/tf_capture_manifest.json` | dynamic TF plus static TCP edge; TCP path audit open |
| exact articulated self-CCD | UNVERIFIED | `results/STAGE5B_FINAL_HANDOFF.json` | unavailable |
| physical clearance/hardware safety | UNVERIFIED | `results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json` | unavailable |

## Large raw evidence kept outside this package

The full x=-0.50 raw JSONL files remain at their repository paths. They were not copied into the <=20 MiB archive. Their size and SHA256 are recorded in `evidence/RAW_EVIDENCE_INDEX.json`; use them only as Level 4 reference after the compact audit files.
"""
    write_text("EVIDENCE_INDEX.md", evidence)

    decisions = f"""# DECISION_LOG

## Decision 01 — Stage5B is the active phase

**Decision:** Do not create Stage5A-D/E.\n\n**Reason:** The current user requirement explicitly closes Stage5AC and enters Stage5B.\n\n**Evidence:** `USER_PROMPTS.md`, `ROOT_GOAL.md`.\n\n**Authority level:** Level 0 / Level 1.

## Decision 02 — Identity `base_link` is nominal software authority

**Decision:** Keep the identity nominal scene as the software authority; keep x=-0.50 as shadow-only.\n\n**Reason:** No CAD, fixture, calibration, URDF/SDF installation transform, or hardware measurement authorizes x=-0.50.\n\n**Evidence:** `results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json`.\n\n**Alternatives considered:** promoting the collision-free x=-0.50 shadow; rejected.\n\n**Authority level:** Level 1.

## Decision 03 — Fixed endpoint is mandatory for continuity repair

**Decision:** A smooth path with altered wp66/wp67 is not admissible.\n\n**Reason:** It silently changes the frozen task trajectory.\n\n**Evidence:** `results/dense_branch_search.json` and the non-fixed-endpoint diagnostic directory described in `FAILED_ATTEMPTS.md`.\n\n**Authority level:** Level 1 / Level 2.

## Decision 04 — Action result is not trajectory PASS

**Decision:** Keep x=-0.50 FJT result code 0 as execution evidence only.\n\n**Reason:** JTC spline reference overshoot reached `{max_error:.9f} rad`; source continuity already fails.\n\n**Evidence:** `results/STAGE5B_EXECUTION_AUDIT.json`.\n\n**Authority level:** Level 2.

## Decision 05 — Do not bypass nominal contact

**Decision:** Stop nominal full physics replay after bounded clock evidence; do not move/disable the nominal world.\n\n**Reason:** The contact slowdown is a real system/backend finding, not a reason to falsify the scene.\n\n**Evidence:** `results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json`.\n\n**Authority level:** Level 1 / Level 2.

## Decision 06 — Package only compact selected evidence

**Decision:** Exclude 200+ MiB raw JSONL from the archive; include paths, sizes, hashes, excerpts/summary, and compact verifier outputs.\n\n**Reason:** The handoff limit is <=20 MiB and raw logs are Level 4 reference.\n\n**Evidence:** `evidence/RAW_EVIDENCE_INDEX.json`, `MANIFEST.json`.\n\n**Authority level:** Level 0 packaging requirement.
"""
    write_text("DECISION_LOG.md", decisions)

    failed = f"""# FAILED_ATTEMPTS

## Attempt 01 — Pure waypoint densification

**Intended goal:** Remove wp66 -> wp67 discontinuity by interpolating the same endpoint solutions.\n\n**Result:** Branch switch remained; maximum wrap-aware step was about 94 degrees.\n\n**Why it failed:** Linear joint interpolation cannot preserve the exact Cartesian task while crossing the current IK branch/limit boundary.\n\n**Evidence:** `results/branch_repair/` if present in repository; summarized by `results/STAGE5B_FINAL_HANDOFF.json`.\n\n**Retry:** Only with a new fixed-endpoint task-space route.

## Attempt 02 — Non-fixed-endpoint orientation graph

**Intended goal:** Find a smooth detour.\n\n**Result:** A diagnostic `PASS` existed only because the first/last samples changed to alternate IK endpoints.\n\n**Why it failed:** It did not splice into the frozen source endpoints.\n\n**Disposition:** Permanently non-admissible for promotion.\n\n**Evidence:** repository `outputs/STAGE5B_SHADOW/branch_graph_search_15deg/STAGE5B_BRANCH_GRAPH_SEARCH.json`.

## Attempt 03 — Fixed-endpoint local and segment detours

**Intended goal:** Use bounded position/orientation bumps, with ±5 mm and larger local probes.\n\n**Result:** All fixed-endpoint campaigns remained `UNRESOLVED`; best segment bottlenecks remained roughly 143.9–150.6 degrees.\n\n**Why it failed:** No connected exact-endpoint path under the 20 degree gate was found.\n\n**Disposition:** Do not repeat unchanged parameter ranges; retry only with a materially different solver/path representation.\n\n**Evidence:** `results/dense_branch_search.json` plus indexed repository campaign paths.

## Attempt 04 — x=-0.50 full FJT replay

**Intended goal:** Verify the full ROS/Gazebo execution chain.\n\n**Result:** Action succeeded, but native JTC spline interpolation produced maximum reference-feedback error `{max_error:.9f} rad`; final endpoint error was `{audit['execution']['final_endpoint_error_max_abs_rad']:.9f} rad`.\n\n**Why it failed:** The candidate's q/dq/ddq/timestamp sequence is not controller-compatible around the branch cut; action success is insufficient.\n\n**Disposition:** Candidate rejected; do not tune the controller to hide it.\n\n**Evidence:** `results/STAGE5B_EXECUTION_AUDIT.json`.

## Attempt 05 — nominal full FJT replay

**Intended goal:** Execute the same full trajectory in the nominal identity world.\n\n**Result:** Initial wall contact caused `{blocker['initial_state']['measured_max_pre_goal_drift_rad']:.9f} rad` drift; the goal was observed accepted, but the physics clock advanced only `{blocker['execution_observation']['clock_sample_s']:.2f} s` during a 12 s probe.\n\n**Why it failed:** Real contact made the nominal physics backend too slow for bounded completion.\n\n**Disposition:** Blocker retained; never disable collision or move the nominal scene without authority.\n\n**Evidence:** `results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json`.

## Attempt 06 — Gazebo GUI/controller startup

**Intended goal:** Bring up Gazebo and controllers.\n\n**Result:** GUI and 1 ms fixed step starved controller-manager lifecycle calls; mesh resource path was initially missing.\n\n**Repair:** Set `GZ_SIM_RESOURCE_PATH`, use server-only mode, match physics step to 10 ms, and add delayed standard controller spawners with explicit timeouts.\n\n**Verification:** `evidence/bringup_summary.json`; `changed_files/ros2_moveit_bridge/launch/stage5b_gazebo.launch.py`.\n\n**Disposition:** Measurement/runtime infrastructure repair accepted; canonical robot baseline unchanged.
"""
    write_text("FAILED_ATTEMPTS.md", failed)

    references = """# REFERENCES

## ROS 2 Jazzy `gz_ros2_control`

- Title: `gz_ros2_control` Jazzy documentation
- URL: https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html
- Used conclusion: Jazzy installs and loads `gz_ros2_control/GazeboSimSystem` through `libgz_ros2_control-system` and a controller manager configuration.
- Decision affected: Stage5B-1 Gazebo/ros2_control bring-up.

No external PDF or secret-bearing reference material is included.
"""
    write_text("references/REFERENCES.md", references)

    changed = [
        ("changed_files/ros2_moveit_bridge/config/stage5b_gazebo.urdf.xacro", ROOT / "ros2_moveit_bridge/config/stage5b_gazebo.urdf.xacro"),
        ("changed_files/ros2_moveit_bridge/config/stage5b_gazebo_controllers.yaml", ROOT / "ros2_moveit_bridge/config/stage5b_gazebo_controllers.yaml"),
        ("changed_files/ros2_moveit_bridge/launch/stage5b_gazebo.launch.py", ROOT / "ros2_moveit_bridge/launch/stage5b_gazebo.launch.py"),
        ("changed_files/tools/stage5b_gazebo_world.py", ROOT / "tools/stage5b_gazebo_world.py"),
        ("changed_files/tools/stage5b_branch_graph_search.py", ROOT / "tools/stage5b_branch_graph_search.py"),
        ("changed_files/tools/stage5b_branch_graph_search_launch.py", ROOT / "tools/stage5b_branch_graph_search_launch.py"),
        ("changed_files/tools/stage5b_branch_repair.py", ROOT / "tools/stage5b_branch_repair.py"),
        ("changed_files/tools/stage5b_branch_repair_launch.py", ROOT / "tools/stage5b_branch_repair_launch.py"),
        ("changed_files/tools/stage5b_prepare_fjt_goal.py", ROOT / "tools/stage5b_prepare_fjt_goal.py"),
        ("changed_files/tools/stage5b_audit_execution.py", ROOT / "tools/stage5b_audit_execution.py"),
        ("changed_files/tools/stage5b_build_final_handoff.py", ROOT / "tools/stage5b_build_final_handoff.py"),
        ("changed_files/scripts/stage5a_runtime_client.py", ROOT / "scripts/stage5a_runtime_client.py"),
        ("changed_files/ros2_moveit_bridge/package.xml", ROOT / "ros2_moveit_bridge/package.xml"),
    ]
    for relative, source in changed:
        copy_file(relative, source)

    results = [
        ("results/STAGE5B_FINAL_HANDOFF.md", ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_FINAL_HANDOFF.md"),
        ("results/STAGE5B_FINAL_HANDOFF.json", ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_FINAL_HANDOFF.json"),
        ("results/STAGE5B_EXECUTION_AUDIT.json", ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_EXECUTION_AUDIT.json"),
        ("results/STAGE5B_NOMINAL_WORLD_AUTHORITY.json", ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_NOMINAL_WORLD_AUTHORITY.json"),
        ("results/STAGE5B_NOMINAL_WORLD_AUTHORITY.md", ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_NOMINAL_WORLD_AUTHORITY.md"),
        ("results/STAGE5B_NOMINAL_REPLAY_BLOCKER.json", ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_NOMINAL_REPLAY_BLOCKER.json"),
        ("results/gazebo_x_minus_050.sdf", ROOT / "outputs/STAGE5B_SHADOW/gazebo_x_minus_050.sdf"),
        ("results/gazebo_nominal.sdf", ROOT / "outputs/STAGE5B_SHADOW/gazebo_nominal.sdf"),
        ("results/initial_positions_candidate.yaml", ROOT / "outputs/STAGE5B_SHADOW/initial_positions_candidate.yaml"),
        ("results/dense_branch_search.json", ROOT / "outputs/STAGE5B_SHADOW/branch_graph_search_dense_large_detours/STAGE5B_BRANCH_GRAPH_SEARCH.json"),
        ("results/STAGE5AC_CANDIDATE_TRAJECTORY.csv", ROOT / "outputs/STAGE5AC_FINAL_CLOSURE/STAGE5AC_CANDIDATE_TRAJECTORY.csv"),
        ("results/STAGE5AC_CANDIDATE_ENVIRONMENT.json", ROOT / "outputs/STAGE5AC_FINAL_CLOSURE/STAGE5AC_CANDIDATE_ENVIRONMENT.json"),
        ("results/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv", ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"),
        ("results/internal_wiper_moveit_inputs/open_arch_seed_joints.csv", ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"),
        ("results/config/ik_graph.yaml", ROOT / "config/ik_graph.yaml"),
    ]
    for relative, source in results:
        copy_file(relative, source)

    runtime = ROOT / "outputs/STAGE5B_SHADOW/runtime_x_minus_050"
    for relative, source in [
        ("results/x_minus_050/action_execution.json", runtime / "stage5a_action_execution.json"),
        ("results/x_minus_050/follow_joint_trajectory_goal.json", runtime / "stage5a_follow_joint_trajectory_goal.json"),
        ("results/x_minus_050/initial_state.json", runtime / "stage5a_initial_state.json"),
        ("results/x_minus_050/tf_capture_manifest.json", runtime / "stage5a_tf_capture_manifest.json"),
    ]:
        copy_file(relative, source)

    bringup_summary = {
        "status": "PASS_FOR_SOFTWARE_BRINGUP_ONLY",
        "ros_distro": "ROS 2 Jazzy",
        "os": "Ubuntu-24.04-D under WSL",
        "gazebo": "Gazebo Sim server-only, fixed 0.01 s step",
        "hardware_plugin": "gz_ros2_control/GazeboSimSystem",
        "controllers": {"fairino5_controller": "active", "joint_state_broadcaster": "active"},
        "robot_spawn": "successful",
        "joint_state_feedback": "observed",
        "moveit": "MoveItPy/KDL model initialization and search observed",
        "build": "colcon build --symlink-install --packages-select fr5_tunnel_moveit_bridge: PASS",
        "claim_fence": "software simulation only; no physical safety/clearance/exact CCD claim",
    }
    write_text("evidence/bringup_summary.json", json.dumps(bringup_summary, indent=2, ensure_ascii=False))

    raw_files = [
        runtime / "stage5a_controller_state_raw.jsonl",
        runtime / "stage5a_joint_states_raw.jsonl",
        runtime / "stage5a_action_feedback_raw.jsonl",
        runtime / "stage5a_tf_raw.jsonl",
        runtime / "stage5a_tf_static_raw.jsonl",
    ]
    raw_index = []
    for path in raw_files:
        raw_index.append({"repository_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path), "authority_level": 4, "included_in_archive": False, "reason": "large raw reference; compact audit included"})
    write_text("evidence/RAW_EVIDENCE_INDEX.json", json.dumps({"schema_version": "stage5b-raw-evidence-index-v1", "files": raw_index}, indent=2, ensure_ascii=False))

    sensitive_patterns = [
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"\b(?:sk|rk)-[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ]
    findings = []
    for path in STAGING.rglob("*"):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in sensitive_patterns:
            if pattern.search(text):
                findings.append(str(path.relative_to(STAGING)).replace("\\", "/"))
                break
    scan_status = "FAIL" if findings else "PASS"
    finding_text = "\n".join(f"- `{item}`" for item in findings) if findings else "- none"
    write_text("SENSITIVE_SCAN.md", f"# SENSITIVE_SCAN\n\n`STATUS = {scan_status}`\n\nScanned selected package files for private-key blocks and common secret-value formats. Variable names and user instructions mentioning credentials were not treated as secret values. No `.env`, cookies, SSH keys, or credentials files were selected.\n\nFindings:\n{finding_text}")
    if findings:
        raise RuntimeError(f"sensitive-value scan failed: {findings}")

    entries = []
    for path in sorted(STAGING.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        relative = str(path.relative_to(STAGING)).replace("\\", "/")
        if relative.startswith("changed_files/"):
            category, authority, required = "source_or_config", 2, True
        elif relative.startswith("results/"):
            category, authority, required = "verified_result_or_input", 2, relative.startswith("results/STAGE5B_") or relative.startswith("results/x_minus_050/")
        elif relative.startswith("evidence/"):
            category, authority, required = "evidence_index", 2, relative.endswith("bringup_summary.json") or relative.endswith("RAW_EVIDENCE_INDEX.json")
        elif relative.startswith("references/"):
            category, authority, required = "external_reference", 4, False
        elif relative in {"ROOT_GOAL.md", "CURRENT_USER_REQUIREMENTS.md", "USER_PROMPTS.md"}:
            category, authority, required = "user_authority", 0, True
        elif relative in {"AUTHORITATIVE_STATE.md", "STATE_TRACKER.json", "COMPLETION_GATE.json"}:
            category, authority, required = "current_state", 1, True
        elif relative in {"HANDOFF.md", "TAKEOVER_PROTOCOL.md", "CONTEXT_AUTHORITY.md", "HISTORICAL_STATE_AUDIT.md", "EVIDENCE_INDEX.md", "DECISION_LOG.md", "FAILED_ATTEMPTS.md", "SENSITIVE_SCAN.md"}:
            category, authority, required = "takeover_control", 1, True
        else:
            category, authority, required = "package_control", 1, False
        entries.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path), "category": category, "authority_level": authority, "required_for_takeover": required})
    manifest = {"schema_version": "stage5b-handoff-manifest-v1", "package_name": PACKAGE_NAME, "created_at": datetime.now().astimezone().isoformat(), "archive_scope": "compact selected evidence; no build/install/cache/raw JSONL copies", "entries": entries}
    write_text("MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))


def verify_extracted(extracted_root: Path, archive: Path, manifest: dict) -> dict:
    package_root = extracted_root / PACKAGE_NAME
    checks = {"archive_readable": False, "required_files_exist": False, "manifest_hashes_match": False, "sensitive_scan_pass": False, "archive_size_mib": archive.stat().st_size / (1024 * 1024), "archive_sha256": sha256(archive), "missing_required": [], "hash_mismatches": []}
    with tarfile.open(archive, mode="r:xz") as handle:
        members = handle.getmembers()
        names = {member.name for member in members}
        checks["archive_readable"] = any(name.endswith("/MANIFEST.json") for name in names)
        handle.extractall(extracted_root)
    missing = []
    mismatches = []
    for entry in manifest["entries"]:
        path = package_root / entry["relative_path"]
        if not path.is_file():
            if entry["required_for_takeover"]:
                missing.append(entry["relative_path"])
            continue
        if sha256(path) != entry["sha256"]:
            mismatches.append(entry["relative_path"])
    checks["missing_required"] = missing
    checks["hash_mismatches"] = mismatches
    checks["required_files_exist"] = not missing
    checks["manifest_hashes_match"] = not mismatches
    scan = (package_root / "SENSITIVE_SCAN.md").read_text(encoding="utf-8") if (package_root / "SENSITIVE_SCAN.md").is_file() else ""
    checks["sensitive_scan_pass"] = "STATUS = PASS" in scan
    checks["overall"] = bool(all(checks[key] for key in ("archive_readable", "required_files_exist", "manifest_hashes_match", "sensitive_scan_pass")) and checks["archive_size_mib"] <= 20.0)
    return checks


def main() -> int:
    if STAGING.exists():
        raise RuntimeError(f"refusing to overwrite existing staging directory: {STAGING}")
    STAGING.mkdir(parents=True)
    try:
        audit = load(ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_EXECUTION_AUDIT.json")
        blocker = load(ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_NOMINAL_REPLAY_BLOCKER.json")
        authority = load(ROOT / "outputs/STAGE5B_SHADOW/STAGE5B_NOMINAL_WORLD_AUTHORITY.json")
        dense = load(ROOT / "outputs/STAGE5B_SHADOW/branch_graph_search_dense_large_detours/STAGE5B_BRANCH_GRAPH_SEARCH.json")
        build_documents(audit, blocker, authority, dense)
        manifest = load(STAGING / "MANIFEST.json")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = DESKTOP / f"{PACKAGE_NAME}_{timestamp}.tar.xz"
        if archive.exists():
            raise RuntimeError(f"refusing to overwrite existing archive: {archive}")
        DESKTOP.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="w:xz", preset=6) as handle:
            handle.add(STAGING, arcname=PACKAGE_NAME)
        verify_dir = ROOT / "tmp" / f"stage5b_handoff_extract_{timestamp}"
        if verify_dir.exists():
            raise RuntimeError(f"refusing to overwrite extraction directory: {verify_dir}")
        verify_dir.mkdir(parents=True)
        try:
            verification = verify_extracted(verify_dir, archive, manifest)
        finally:
            shutil.rmtree(verify_dir)
        verification.update({"archive_path": str(archive), "manifest_entry_count": len(manifest["entries"]), "staging_removed": False})
        verification_path = ROOT / "outputs" / "STAGE5B_HANDOFF_PACKAGE_VERIFICATION.json"
        verification_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not verification["overall"]:
            raise RuntimeError(f"package verification failed: {verification}")
        shutil.rmtree(STAGING)
        verification["staging_removed"] = True
        verification_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "archive": str(archive), "size_bytes": archive.stat().st_size, "sha256": verification["archive_sha256"], "manifest_entries": verification["manifest_entry_count"], "verification": str(verification_path), "staging_removed": True}, ensure_ascii=False))
        return 0
    except Exception:
        if STAGING.exists():
            shutil.rmtree(STAGING)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
