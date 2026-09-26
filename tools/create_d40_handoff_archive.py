"""Create and validate the formal D40 cross-chat handoff archive.

This is packaging-only tooling.  It copies the already-produced D39/D40 evidence,
records the accessible prompt text, writes the handoff ledgers, validates the
archive as tar.xz, and removes only its own temporary staging directory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(os.environ.get("USERPROFILE", "C:/Users/86198")) / "Desktop"
ORIGINAL_D40_PROMPT = Path(
    r"C:\Users\86198\.codex\attachments\5b6aa700-f08b-4a46-98f2-8c79469fd021\pasted-text.txt"
)
CURRENT_HANDOFF_PROMPT = Path(
    r"C:\Users\86198\.codex\attachments\9d00f97d-9697-4c54-a19c-a729ae01014e\pasted-text.txt"
)
AGENTS_FILE = ROOT / "AGENTS.md"
D40_OUTPUT = ROOT / "outputs/stage3_h13_d40_task_space_geometry_locked_causal_cartesian_recovery_execution"
D40_WINNER = D40_OUTPUT / "routeA_DLS_position_dominant_v4"
D39_OUTPUT = ROOT / "outputs/stage3_h13_d39_causal_task_space_recovery"
INPUTS = ROOT / "outputs/internal_wiper_moveit_inputs"


def run_capture(args: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout + f"\n[exit_code={result.returncode}]\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_file(src: Path, dst: Path, missing: list[str], *, required: bool = True) -> None:
    if not src.exists():
        if required:
            missing.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    DESKTOP.mkdir(parents=True, exist_ok=True)
    existing = sorted(DESKTOP.glob("D40_TASK_SPACE_GEOMETRY_LOCKED_CAUSAL_CARTESIAN_RECOVERY_HANDOFF_*.tar.xz"))
    if existing:
        raise RuntimeError(
            "Desktop already contains formal D40 handoff archive(s); refusing to create a second final archive: "
            + ", ".join(str(p) for p in existing)
        )

    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S+0800")
    archive = DESKTOP / f"D40_TASK_SPACE_GEOMETRY_LOCKED_CAUSAL_CARTESIAN_RECOVERY_HANDOFF_{timestamp}.tar.xz"
    staging = Path(tempfile.mkdtemp(prefix=".d40_handoff_staging_", dir=str(ROOT)))
    missing: list[str] = []

    try:
        # Root prompt record.  The two attachment files are the verbatim prompt
        # bodies supplied in this session; the wrapper text and accessibility
        # note make the provenance explicit without inventing hidden history.
        prompt_parts = [
            "# USER_PROMPTS.md\n",
            "This file preserves the user-provided prompt bodies accessible in the current Codex session.\n",
            "The outer chat wrappers contained an empty `My request:` field; the full request text was in the pasted-text attachments below.\n\n",
            "## Prompt 01 — D40 execution authorization (verbatim attachment body)\n",
        ]
        if ORIGINAL_D40_PROMPT.exists():
            prompt_parts.append(ORIGINAL_D40_PROMPT.read_text(encoding="utf-8"))
        else:
            missing.append(str(ORIGINAL_D40_PROMPT))
            prompt_parts.append("[INACCESSIBLE ATTACHMENT: original D40 authorization prompt]\n")
        prompt_parts.extend(
            [
                "\n\n## Prompt 02 — D40 cross-chat handoff archive request (verbatim attachment body)\n",
            ]
        )
        if CURRENT_HANDOFF_PROMPT.exists():
            prompt_parts.append(CURRENT_HANDOFF_PROMPT.read_text(encoding="utf-8"))
        else:
            missing.append(str(CURRENT_HANDOFF_PROMPT))
            prompt_parts.append("[INACCESSIBLE ATTACHMENT: current handoff request]\n")
        prompt_parts.extend(
            [
                "\n\n## Accessibility boundary\n",
                "No inaccessible prior-chat history, hidden messages, or unobserved prompts were reconstructed. "
                "Only the two prompt attachment bodies above and the user-provided repository instruction file were available to this packaging run.\n",
            ]
        )
        write_text(staging / "USER_PROMPTS.md", "".join(prompt_parts))

        write_text(
            staging / "HANDOFF.md",
            f"""# D40 Cross-Chat Handoff

## 1. Archive identity and reading order

- Formal archive: `{archive.name}`
- Archive root: this file (`HANDOFF.md`)
- Scope: D40 task-space geometry locked causal Cartesian recovery and native certification
- Created locally: {dt.datetime.now().isoformat(timespec='seconds')}
- Recommended reading order: `HANDOFF.md` → `USER_PROMPTS.md` → `results/D40_FINAL_REPORT.md` → `results/D40_METHOD_LEDGER.md` → `results/D39_VS_D40_COMPARISON.md` → `evidence/task_space/final_acceptance_summary.json` → `evidence/task_space/moveit_quality_report.csv` → `evidence/FAILURE_LEDGER.md` → `references/RESEARCH_LEDGER.md`.

## 2. Final D40 state

- `TASK_STATUS = PASS`
- `FIRST_UNRESOLVED_BLOCKER = NONE`
- `D39_RETENTION_BASELINE_1 = stable_velocity_residual_update` remains the retained B1 self-fed baseline.
- `D40_RETENTION_BASELINE_2 = routeA_DLS_position_dominant_v4` is locked.
- `CANONICAL_PROMOTION = YES`: the D40 retention state is promoted as `D40_RETENTION_BASELINE_2_LOCKED`; this did not overwrite the D39 result files.
- System champion: `stable_velocity_residual_update + normal_constrained_DLS`.
- H1 selected update: `update_480`.
- D41 was not started.

## 3. What changed and why

D39 retained the causal self-fed controller but its native task-space geometry gate failed. D40 added a native MoveIt2 FK/Jacobian task-space projection path, using position-dominant damped-least-squares trust-region updates with normal constraints. The winning implementation is `build_normal_constrained_dls_trajectory` in the bridge and the `normal_dls_waypoints` planning mode. It was promoted only after the complete native MoveIt2 validation path passed.

The existing `ros2_moveit_bridge/plan_closed_contour_moveit.py` was already dirty before D40. The archive includes its current full working copy and a mixed-worktree diff, explicitly labelled so unrelated pre-existing edits are not attributed to D40. The new D40 runner and D40 tests are included under `changed_files/`.

## 4. Before / after result

| Metric | D39 native result | D40 winner |
|---|---:|---:|
| Path deviation p95 | 52.557053961281554 mm | 0.028219554641211297 mm |
| Path deviation max | 52.696673331918014 mm | 0.14126918172561212 mm |
| Standoff max absolute error | 27.786585975634324 mm | 0.10550011330623388 mm |
| Normal max error | 6.1632178604247° | 1.0068129181254613° |
| TCP speed mean | not promoted | 0.003005357453672808 m/s |
| TCP speed p05–p95 | not promoted | 0.0029999999996098235–0.0030000000004324918 m/s |
| Max joint step | not promoted | 14.991790333724396° |
| Collision count | D39 geometry gate failed | 0 |

D39 B1 retention remains: 181/181 self-fed points, no future joint reads, no joint-limit violation, no p33 warning, no hard rollout violation, H32 ≈ 0.131635 rad, and p35 q4 ≈ 0.273695 rad. Those are not relabelled as new post-projection policy metrics.

## 5. Gate and certification answers

- B1 causal self-fed retention: `SOLVED / RETAINED` from D39; future joint reads are 0.
- B2 native task-space geometry: `SOLVED / PASS` in D40.
- B3 full-robot certification: `SOLVED / PASS` for this 181-point native ON-state simulation scope.
- B4 stability: `SOLVED / PASS`; post-Ruckig smoothed trajectory and dynamics checks passed.
- Native quality: `PASS`.
- Collision reporting: `adaptive_discrete_interpolation`; it is not strict continuous collision detection.
- Bullet CCD: `not_available`.
- Clearance: unavailable and represented as JSON nulls; no clearance was inferred from collision-free states.
- Native acceptance: PlanningScene, FK, collision, joint dynamics, and post-Ruckig checks all ran. The winning route is not a surrogate-only or placeholder pass.
- Final acceptance dynamics ratios: max velocity ratio `0.016421107188169345`, max acceleration ratio `0.009579396534353725`, max jerk ratio `0.00023042942821655058`.

The B3 PASS is a native 181-point ON-state certification result, not a physical-robot execution claim or a claim about unmeasured TCP calibration beyond the declared scope.

## 6. Tried methods and failures

- Full-pose seeded MoveIt2 IK from the D39 stable prior failed at waypoint 67 because all branches exceeded the 20° continuity gate.
- Widened beam search (`beam8`), initial-anchor seeding, authoritative seed control as an IK route, and initial-anchor roll9 search also failed at waypoint 67; the direct authoritative seed-joint control route itself passed native certification but was not selected as champion.
- Native normal-constrained DLS v1 stalled at waypoint 87 (0.9299 mm position, 0.0632° normal); v2 stalled at waypoint 88 (9.3528 mm, 1.5597°); v3 stalled at waypoint 87 (8.9594 mm, 1.4910°).
- Position-dominant DLS v4 completed all 181 points and passed the native gate. Full details are in `results/D40_METHOD_LEDGER.md` and `evidence/FAILURE_LEDGER.md`.

## 7. Scope, inputs, and reproducibility

- Authoritative input is the 181-point open-arch pair under `outputs/internal_wiper_moveit_inputs/`.
- This is ON-state open-arch planning only. No OFF states, reorientation, retreat, approach, closed-contour transitions, GNN, PPO, LSTM, or Transformer were added.
- The archive carries the selected inputs, D39/D40 summaries, winner traces, runtime log, route failure summaries, source files, git evidence, and test logs.
- Tests: `python -m pytest -q tests/test_stage3_h13_d39.py tests/test_stage3_h13_d40.py` passed `8 passed in 1.66s` in the source run; the packaged test log is authoritative for this handoff.
- Compilation: the D39/D40 scripts, bridge, and tests were compiled by `py_compile`; see `evidence/tests/py_compile.txt`.
- No scientific result placeholder was created. No D39 result file was overwritten.

## 8. Research / external-source disclosure

External internet research was **not performed** for D40. No external citation or online source is being claimed. The local MoveIt2 binding source inspected was `tmp/stage3_h7_8_moveit2_upstream/moveit_py/src/moveit/moveit_core/robot_state/robot_state.cpp`, specifically the `RobotState.get_jacobian` binding. The local implementation consequence is documented in `references/RESEARCH_LEDGER.md`.

## 9. Next-chat action

The next chat should treat D40 as complete and locked, read the files listed in the reading order, and not restart D40 or invent a new blocker. Any next-stage work requires explicit authorization. A sensible authorized follow-up is to validate the locked native result against measured TCP/calibration data or begin the next declared stage; neither was performed here.

## 10. Integrity boundary

`USER_PROMPTS.md` contains the two accessible verbatim prompt attachment bodies. Inaccessible prior-chat history was not reconstructed. The repository was already a dirty worktree; the archive preserves that fact and does not claim the mixed bridge diff is D40-only.\n""",
        )

        # The repository instruction file is included separately so roles and
        # provenance are not blurred with the chat prompt record.
        copy_file(AGENTS_FILE, staging / "evidence/repository/AGENTS.md", missing)

        # Primary D40 results.
        d40_result_files = [
            "D40_FINAL_REPORT.md",
            "D40_SUMMARY.json",
            "D40_RETENTION_BASELINES.json",
            "D40_BLOCKER_STATUS.json",
            "D40_TASK_SPACE_EVALUATION.csv",
            "D40_ROBOT_CERTIFICATION.csv",
            "D40_SEED_MANIFEST.json",
        ]
        for name in d40_result_files:
            copy_file(D40_OUTPUT / name, staging / f"results/{name}", missing)

        d39_result_files = [
            "D39_FINAL_REPORT.md",
            "D39_SUMMARY.json",
            "D39_RETENTION_BASELINES.json",
            "D39_CERTIFICATION_SUMMARY.json",
            "D39_TASK_SPACE_EVALUATION.csv",
            "D39_ROBOT_CERTIFICATION.csv",
            "D39_SELF_FED_EVALUATION.csv",
            "D39_SHADOW_CANDIDATES.csv",
            "D39_RETENTION_BASELINES.json",
        ]
        for name in d39_result_files:
            copy_file(D39_OUTPUT / name, staging / f"results/{name}", missing)
        copy_file(
            D39_OUTPUT / "shadow_candidates/stable_velocity_residual_update.csv",
            staging / "results/stable_velocity_residual_update.csv",
            missing,
        )

        # Authoritative input copies used by the native run.
        for name in [
            "open_arch_tcp_poses_base_link.csv",
            "open_arch_seed_joints.csv",
        ]:
            copy_file(INPUTS / name, staging / f"results/inputs/{name}", missing)

        # Native MoveIt2 / FK / dynamics / post-Ruckig evidence for the winning
        # route.  Compact summaries are enough for the handoff; source traces
        # remain in the repository and the full runtime log is retained here.
        winner_files = [
            "moveit_quality_report.csv",
            "moveit_fk_tcp_trace.csv",
            "moveit_joint_dynamics_report.csv",
            "moveit_collision_report.csv",
            "moveit_waypoint_joint_trajectory.csv",
            "moveit_smoothed_joint_trajectory.csv",
            "moveit_joint_step_report.csv",
            "moveit_waypoint_joint_step_report.csv",
            "moveit_runtime.log",
            "final_acceptance_summary.json",
            "production_readiness_check.json",
            "audit_goal_requirements_strict.json",
            "ik_continuity_segments_summary.json",
        ]
        for name in winner_files:
            copy_file(D40_WINNER / name, staging / f"evidence/task_space/{name}", missing)

        # Compact summaries from failed routes make the route decision auditable
        # without copying unrelated legacy or bulky outputs.
        for route_dir in sorted(D40_OUTPUT.glob("route*")):
            if route_dir == D40_WINNER or not route_dir.is_dir():
                continue
            for path in route_dir.glob("*failure*summary*.json"):
                copy_file(path, staging / f"evidence/candidates/{route_dir.name}/{path.name}", missing, required=False)
            for path in route_dir.glob("*diagnostic*.json"):
                copy_file(path, staging / f"evidence/candidates/{route_dir.name}/{path.name}", missing, required=False)

        # D40 implementation and directly relevant execution/test sources.
        changed_sources = [
            (ROOT / "tools/stage3_h13_d40_task_space_geometry_locked_causal_cartesian_recovery_execution.py", "changed_files/tools/"),
            (ROOT / "ros2_moveit_bridge/plan_closed_contour_moveit.py", "changed_files/ros2_moveit_bridge/"),
            (ROOT / "tests/test_stage3_h13_d40.py", "changed_files/tests/"),
            (ROOT / "tests/test_stage3_h13_d39.py", "changed_files/tests/"),
            (ROOT / "tools/stage3_h13_d39_causal_task_space_recovery.py", "changed_files/tools/"),
            (ROOT / "scripts/run_internal_wiper_moveit_strict.sh", "changed_files/scripts/"),
            (ROOT / "scripts/run_moveit_strict_validation.sh", "changed_files/scripts/"),
        ]
        for src, dst_dir in changed_sources:
            copy_file(src, staging / dst_dir / src.name, missing)

        # Evidence of the exact validation performed for this handoff.
        pytest_log = run_capture([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_d39.py", "tests/test_stage3_h13_d40.py"])
        write_text(staging / "evidence/tests/pytest_d39_d40.txt", pytest_log)
        if "[exit_code=0]" not in pytest_log:
            raise RuntimeError("packaged pytest validation failed; see evidence/tests/pytest_d39_d40.txt")
        compile_targets = [
            "tools/stage3_h13_d40_task_space_geometry_locked_causal_cartesian_recovery_execution.py",
            "tools/stage3_h13_d39_causal_task_space_recovery.py",
            "ros2_moveit_bridge/plan_closed_contour_moveit.py",
            "tests/test_stage3_h13_d39.py",
            "tests/test_stage3_h13_d40.py",
        ]
        compile_log = run_capture([sys.executable, "-m", "py_compile", *compile_targets])
        write_text(staging / "evidence/tests/py_compile.txt", compile_log)
        if "[exit_code=0]" not in compile_log:
            raise RuntimeError("packaged py_compile validation failed; see evidence/tests/py_compile.txt")

        # Git evidence is deliberately labelled as mixed working-tree evidence:
        # the bridge file was already dirty before D40 and is not a D40-only patch.
        write_text(staging / "evidence/git/git_status_short.txt", run_capture(["git", "status", "--short"]))
        write_text(
            staging / "evidence/git/git_diff_stat.txt",
            run_capture(["git", "diff", "--stat", "--", "ros2_moveit_bridge/plan_closed_contour_moveit.py"]),
        )
        write_text(
            staging / "evidence/git/plan_closed_contour_moveit_mixed_worktree.diff",
            run_capture(["git", "diff", "--", "ros2_moveit_bridge/plan_closed_contour_moveit.py"]),
        )
        write_text(
            staging / "evidence/git/relevant_log.txt",
            run_capture(["git", "log", "-10", "--oneline", "--", "tools/stage3_h13_d40_task_space_geometry_locked_causal_cartesian_recovery_execution.py", "ros2_moveit_bridge/plan_closed_contour_moveit.py"]),
        )

        # Human-readable handoff ledgers.
        write_text(
            staging / "changed_files/CHANGED_FILES_MANIFEST.md",
            """# Changed-file manifest

## D40 additions

- `tools/stage3_h13_d40_task_space_geometry_locked_causal_cartesian_recovery_execution.py`: Route B seed preparation, D39 retention check, D40 evidence materialization, and canonical baseline lock.
- `tests/test_stage3_h13_d40.py`: D40 artifact and evidence-contract tests.

## D40-modified existing source

- `ros2_moveit_bridge/plan_closed_contour_moveit.py`: native FK/Jacobian normal-constrained DLS trajectory mode and `normal_dls_waypoints` planning branch. This file was already dirty before D40; the archive contains the full mixed working file and the mixed-worktree diff, not a fabricated D40-only patch.

## Context-only source

- D39 execution/test files and strict MoveIt launch scripts are included to make the retained baseline and native validation reproducible. They are not represented as newly changed by D40.

No D39 result file was overwritten by the D40 execution.
""",
        )

        write_text(
            staging / "results/D40_METHOD_LEDGER.md",
            """# D40 method ledger

| Route / method | Status | Evidence / conclusion |
|---|---|---|
| Route A: native MoveIt2 FK/Jacobian DLS, normal-constrained v1 | EXECUTED / FAILED | Stalled at waypoint 87; strict local position error 0.9299 mm and normal error 0.0632°. |
| Route A: native MoveIt2 FK/Jacobian DLS, normal-constrained v2 | EXECUTED / FAILED | Stalled at waypoint 88; position error 9.3528 mm and normal error 1.5597°. |
| Route A: native MoveIt2 FK/Jacobian DLS, normal-constrained v3 | EXECUTED / FAILED | Stalled at waypoint 87; position error 8.9594 mm and normal error 1.4910°. |
| Route A: native MoveIt2 FK/Jacobian DLS, position-dominant trust-region v4 | EXECUTED / PASS / WINNER | Full 181-point native pipeline passed; this is D40 Retention Baseline 2. |
| Route B: full-pose seeded MoveIt2 IK from D39 stable prior | EXECUTED / FAILED | Failed at waypoint 67; all branches exceeded the 20° continuity gate. |
| Route B: widened IK beam (beam8) | EXECUTED / FAILED | Failed at waypoint 67; widening did not recover a continuity-valid branch. |
| Route B: initial-anchor seeded IK | EXECUTED / FAILED | Failed at waypoint 67. |
| Route B: initial-anchor roll9 search | EXECUTED / FAILED | Failed at waypoint 67. |
| Route B: authoritative seed-joint control | EXECUTED / PASS / CONTROL | Native certification passed, but it was not selected as the D40 system champion. |
| Route C: differentiable-FK self-fed training | NOT ATTEMPTED in D40 | D40 did not claim new training; D39 self-fed retention remains the B1 baseline. |
| Route D: task-space-native policy/projection | IMPLEMENTED / COMBINED / WINNER | Normal-constrained task-space DLS projection is the promoted D40 geometry recovery method. |
| Beyond-route backend: MoveIt2 RobotState Jacobian binding | INSPECTED / USED | Local upstream binding was inspected and the native Jacobian call was used in the bridge. |

D40 is therefore a native task-space recovery and certification result, not a claim that every proposed route was successful.
""",
        )

        write_text(
            staging / "results/D39_VS_D40_COMPARISON.md",
            """# D39 versus D40 comparison

| Dimension | D39 retained baseline / state | D40 promoted state |
|---|---|---|
| Retention baseline | `D39_RETENTION_BASELINE_1 = stable_velocity_residual_update` | `D40_RETENTION_BASELINE_2 = routeA_DLS_position_dominant_v4` |
| Point count | 181 / 181 | 181 / 181 |
| Self-fed B1 | PASS; future joint reads 0; H32 ≈ 0.131635 rad; p35 q4 ≈ 0.273695 rad | D39 self-fed policy retained as the B1 reference; the D40 task-space projection is not re-labelled as a new self-fed-policy score |
| Native path p95 | 52.557053961281554 mm | 0.028219554641211297 mm |
| Native path max | 52.696673331918014 mm | 0.14126918172561212 mm |
| Native standoff max | 27.786585975634324 mm | 0.10550011330623388 mm |
| Native normal max | 6.1632178604247° | 1.0068129181254613° |
| B2 geometry | unresolved in D39 | solved / PASS |
| B3 full-robot certification | unresolved in D39 | solved / PASS in the native 181-point ON-state scope |
| B4 stability | unresolved in D39 | solved / PASS; post-Ruckig smoothed trajectory certified |
| Canonical state | baseline 2 was NONE | baseline 2 LOCKED; D39 files remained unchanged |

D40 changes the retained geometry/certification state while preserving D39 as the historical B1 retention reference. H32 and p35 q4 are not fabricated as post-projection policy metrics.
""",
        )

        write_text(
            staging / "results/RETENTION_BASELINE_HISTORY.md",
            """# Retention baseline history

1. D39 locked `D39_RETENTION_BASELINE_1 = stable_velocity_residual_update` for B1 self-fed retention.
2. D40 tested full-pose seeded IK variants and initial-anchor/roll/beam variants; each failed the waypoint-67 continuity gate.
3. D40 tested three strict normal-constrained native DLS configurations; v1–v3 stalled at waypoints 87–88.
4. D40 validated `routeA_DLS_position_dominant_v4` through native MoveIt2 PlanningScene, FK, collision, dynamics, and post-Ruckig checks.
5. D40 locked `D40_RETENTION_BASELINE_2_LOCKED` to that route. The direct authoritative seed control remains a passing control route, not the champion.
""",
        )

        write_text(
            staging / "evidence/FAILURE_LEDGER.md",
            """# D40 failure ledger

| Attempt | Failure point | Observed result | Lesson / disposition |
|---|---|---|---|
| Full-pose seeded MoveIt2 IK from D39 stable prior | Waypoint 67 | Every branch exceeded the 20° continuity gate; representative branch deltas included ~143° | Geometry/topology mismatch under full-pose IK; do not promote. |
| Beam8 IK | Waypoint 67 | Widening the branch set did not yield a continuity-valid branch | Search breadth was not the limiting factor; do not promote. |
| Initial-anchor IK | Waypoint 67 | Failed continuity gate | Seed choice alone did not repair the geometry. |
| Initial-anchor roll9 | Waypoint 67 | Failed continuity gate | Tool-axis roll search did not repair the branch transition. |
| Native DLS v1 | Waypoint 87 | Position 0.9299 mm, normal 0.0632° at local stall | Strict local convergence was too brittle for the full path. |
| Native DLS v2 | Waypoint 88 | Position 9.3528 mm, normal 1.5597° | Weighting/step configuration remained trapped; do not promote. |
| Native DLS v3 | Waypoint 87 | Position 8.9594 mm, normal 1.4910° | Alternate weighting remained trapped; do not promote. |

The failed routes are retained as evidence, not hidden. No failed route is represented as a PASS.
""",
        )

        write_text(
            staging / "evidence/BUG_FIX_LEDGER.md",
            """# D40 implementation / bug-fix ledger

1. The existing bridge had no native normal-constrained DLS trajectory mode. D40 added `build_normal_constrained_dls_trajectory` and the `normal_dls_waypoints` planning branch, using MoveIt2 FK/Jacobian evaluation with position-dominant trust-region updates.
2. The first DLS implementation used a local fail-closed guard tighter than the authoritative native quality gate. The internal threshold was aligned to the native 4 mm / 5° gate, while the native report remained the promotion authority; v4 then passed the full native gate.
3. One beam-search retry initially passed an integer where the ROS2 parameter expected a floating-point diversity value. The retry used `5.0`; the route still failed at waypoint 67, so this execution issue did not change the scientific conclusion.
4. The D40 runner materializes seed manifests, D40 summaries, route ledgers, and blocker status. Tests assert the 181-point scope, no future joint reads, collision-label semantics, unavailable CCD/clearance semantics, and locked baseline outputs.
""",
        )

        write_text(
            staging / "references/RESEARCH_LEDGER.md",
            """# Research / citation ledger

External internet research: NOT PERFORMED for D40. No external paper, website, GitHub repository, or online quotation is claimed in this handoff.

Local technical source inspected: `tmp/stage3_h7_8_moveit2_upstream/moveit_py/src/moveit/moveit_core/robot_state/robot_state.cpp`, specifically the MoveIt2 Python binding for `RobotState.get_jacobian`. Existing local bridge APIs were also inspected.

Implementation consequence: D40 uses the native MoveIt2 RobotState FK/Jacobian path for task-space projection and labels collision results `adaptive_discrete_interpolation`; Bullet CCD and clearance remain `not_available` / JSON null where unavailable.
""",
        )

        write_text(
            staging / "evidence/SCOPE_AND_INTEGRITY.md",
            """# Scope and integrity statement

- Authoritative input: the 181-point open-arch pair under `outputs/internal_wiper_moveit_inputs/`.
- State scope: ON-state open-arch planning only. No OFF state, reorientation, retreat, approach, closed-contour transition, GNN, PPO, LSTM, or Transformer was added to the D40 graph.
- Collision semantics: `adaptive_discrete_interpolation`, not strict continuous collision detection.
- Bullet CCD and clearance: unavailable in the reporting backend; represented as `not_available` and JSON nulls, never inferred from collision-free states.
- Validity: the winning result has native MoveIt2 PlanningScene, FK, collision, joint dynamics, and post-Ruckig checks recorded in `evidence/task_space/`.
- No D39 result file was overwritten. The bridge source is a mixed dirty working file and is labelled accordingly.
- No D41 work was started.
""",
        )

        if missing:
            raise RuntimeError("required handoff source(s) missing:\n" + "\n".join(sorted(set(missing))))

        # Package metadata is generated only after all copies and ledgers exist.
        all_files = sorted(p for p in staging.rglob("*") if p.is_file())
        # Scan for concrete credential material, not generic words such as
        # "token" appearing in a prompt or report.
        secret_patterns = [
            re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
            re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
            re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._-]{20,}"),
        ]
        findings: list[str] = []
        for path in all_files:
            try:
                data = path.read_bytes()
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in secret_patterns:
                if pattern.search(text):
                    findings.append(str(path.relative_to(staging)) + ":" + pattern.pattern)
        if findings:
            raise RuntimeError("credential-pattern scan failed: " + "; ".join(findings))

        manifest = []
        for path in all_files:
            data = path.read_bytes()
            manifest.append(
                {
                    "path": str(path.relative_to(staging)).replace("\\", "/"),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        write_text(
            staging / "ARCHIVE_MANIFEST.json",
            json.dumps(
                {
                    "archive_name": archive.name,
                    "created_local": dt.datetime.now().isoformat(timespec="seconds"),
                    "root": "D40_TASK_SPACE_GEOMETRY_LOCKED_CAUSAL_CARTESIAN_RECOVERY_HANDOFF",
                    "file_count": len(manifest) + 1,
                    "credential_pattern_scan": "PASS",
                    "files": manifest,
                },
                indent=2,
            )
            + "\n",
        )

        with tarfile.open(archive, mode="w:xz", preset=9) as tar:
            for path in sorted(p for p in staging.rglob("*") if p.is_file()):
                tar.add(path, arcname=str(path.relative_to(staging)).replace("\\", "/"), recursive=False)

        # Validate the actual archive bytes and required top-level files.
        required_members = {
            "HANDOFF.md",
            "USER_PROMPTS.md",
            "ARCHIVE_MANIFEST.json",
            "results/D40_FINAL_REPORT.md",
            "results/D40_SUMMARY.json",
            "results/D40_RETENTION_BASELINES.json",
            "evidence/task_space/moveit_quality_report.csv",
            "evidence/task_space/final_acceptance_summary.json",
            "evidence/tests/pytest_d39_d40.txt",
            "references/RESEARCH_LEDGER.md",
        }
        with tarfile.open(archive, mode="r:xz") as tar:
            names = set(tar.getnames())
            absent = sorted(required_members - names)
            if absent:
                raise RuntimeError("archive validation missing required members: " + ", ".join(absent))
            for member in required_members:
                extracted = tar.extractfile(member)
                if extracted is None or not extracted.read(64):
                    raise RuntimeError("archive validation found empty required member: " + member)

        xz_executable = shutil.which("xz")
        if xz_executable:
            xz_result = subprocess.run([xz_executable, "-t", str(archive)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            xz_status = f"external xz -t PASS (exit_code={xz_result.returncode})\n{xz_result.stdout}"
            if xz_result.returncode != 0:
                raise RuntimeError("external xz integrity test failed: " + xz_result.stdout)
        else:
            xz_status = "external xz executable unavailable; Python tarfile r:xz integrity validation PASS"
        write_text(staging / "evidence/tests/xz_integrity.txt", xz_status + "\n")

        # A small sidecar is not placed on Desktop; print the facts for the
        # calling shell and keep the Desktop limited to the one formal archive.
        size = archive.stat().st_size
        if size > 20 * 1024 * 1024:
            raise RuntimeError(f"archive exceeds 20 MB: {size} bytes")
        final_count = len(list(DESKTOP.glob("D40_TASK_SPACE_GEOMETRY_LOCKED_CAUSAL_CARTESIAN_RECOVERY_HANDOFF_*.tar.xz")))
        if final_count != 1:
            raise RuntimeError(f"Desktop formal D40 archive count is {final_count}, expected exactly one")
        print(json.dumps({
            "archive": str(archive),
            "bytes": size,
            "megabytes_decimal": round(size / 1_000_000, 3),
            "formal_d40_archives_on_desktop": final_count,
            "xz_integrity": xz_status.strip(),
            "staged_file_count": len(list(p for p in staging.rglob("*") if p.is_file())),
        }, indent=2))
        return 0
    finally:
        # The staging directory is generated by this script and is never a
        # user-provided path.  Remove only that exact directory after the run.
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
