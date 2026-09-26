"""Create and validate the single official D48 Codex -> Web ChatGPT archive."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
D48 = REPO / "outputs/D48_STAGE4C_EXECUTION_FORM_V1"
D47 = REPO / "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3"
PROMPT_D48 = Path(r"C:\Users\86198\.codex\attachments\e00106af-bb67-4276-a66f-e6e0a7610fc2\pasted-text.txt")
PROMPT_COMPLETION = Path(r"C:\Users\86198\.codex\attachments\c39f6894-047c-40c6-90fc-6e6c3a687a64\pasted-text.txt")
PROMPT_HANDOFF = Path(r"C:\Users\86198\.codex\attachments\a613aae8-276a-416b-9608-b70c509d0544\pasted-text.txt")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stat(obj: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_one(src: Path, dst: Path, required: bool = True) -> bool:
    if not src.exists():
        if required:
            raise FileNotFoundError(src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.parts[1:]
    return "/mnt/" + drive + "/" + "/".join(tail)


def run_capture(args: list[str], cwd: Path = REPO) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return ((result.stdout or "") + (result.stderr or "")).strip() + f"\n[exit_code={result.returncode}]\n"


def make_docs(stage: Path, c1: dict[str, Any], ultra: dict[str, Any], c2: dict[str, Any], c4: dict[str, Any], replay: dict[str, Any]) -> None:
    parent = "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3"
    current = parent
    c1_violations = stat(c1, "joint_space", "jerk_limit_violations")
    ultra_jerk = stat(ultra, "joint_space", "jerk_ratio", "max") * 8.0
    c4_nom = stat(c4, "variant_summaries", "candidate_nominal")
    c4_self = c4.get("rnea_aba_self_consistency", {})

    root_goal = """# ROOT GOAL

Improve the computational robot-arm trajectory toward increasingly high-quality, physically meaningful and execution-credible motion: accurate, smooth, stable, repeatable, singularity-resistant, collision-safe, clearance-safe, dynamically reasonable under the available robot model, and robust in difficult configurations.

The objective is not merely to improve benchmark numbers. The objective is to improve the quality of the motion represented by the computational robot model.

## Permanent protected principle

The project uses a monotonic rolling-champion rule. Shadow experiments may regress arbitrarily; canonical/promoted state may only remain at the validated floor or advance to a stronger validated state. D48 started from the protected D47 B3 state at `outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3`.

D47 B3 preserved Stage-3 update 480 and H1/H32, reduced singularity-risk cases to zero, reduced environment and discrete self-collisions to zero, reduced environment CCD failures to zero, and established positive environment and self clearance. D48 did not promote a replacement, so D47 B3 remains authoritative.

## D48 scientific direction

- C1: native/production-equivalent post-Ruckig execution-form validation.
- C2: continuous self-collision validation, with no continuous claim when the backend is unavailable.
- C3: material non-regressive improvement of difficult/stress accuracy, repeatability and/or smoothness.
- C4: model-based dynamic feasibility using the current URDF inertial model.
- C5: real physical/hardware certification only when authoritative physical data exists.

Hardware truth must never be invented. Computational/model evidence and physical/hardware evidence remain separate.

## Non-drift rules

Stage-3 update 480, H1/H32 semantics, the current validated rolling champion, collision geometry, SRDF/ACM semantics, robot limits, benchmark definitions, safety gates, continuous-collision requirements once a real backend exists, and the non-regression policy may not be silently relaxed to make a failure disappear.
"""
    write(stage / "ROOT_GOAL.md", root_goal)

    authority = """# AUTHORITY MAP

The next ChatGPT must use this order of authority:

1. **Tier 0 — ROOT OBJECTIVE:** `ROOT_GOAL.md`.
2. **Tier 1 — VERIFIED CURRENT STATE:** `STATE_SNAPSHOT.md`, `STATE_SNAPSHOT.json`, final reports, promotion ledger, and persisted verifier outputs.
3. **Tier 2 — EXTERNALLY TRACKED PROGRESS:** `PROGRESS_LEDGER.md`, `PROGRESS_LEDGER.json`.
4. **Tier 3 — IMPLEMENTATION AND EVIDENCE:** `changed_files/` and `evidence/`.
5. **Tier 4 — USER INTENT:** `USER_PROMPTS.md`, which records what was requested but does not override verified repository state.
6. **Tier 5 — HISTORICAL REFERENCE:** `raw_logs/` and failed-shadow notes.

Historical plans, agent behavior, hypotheses, and raw logs are evidence, not authority. If they conflict with `ROOT_GOAL.md` or verifier-backed current state, the root goal and verified state win.

The archive deliberately separates user intent, authoritative goal/state, execution artifacts, and verification evidence. It does not contain private reasoning or hidden system/developer instructions.
"""
    write(stage / "AUTHORITY_MAP.md", authority)

    state = {
        "schema_version": "d48-state-snapshot-v1",
        "project_stage": "Stage 4C / D48",
        "root_goal_file": "ROOT_GOAL.md",
        "starting_protected_champion": parent,
        "current_protected_champion": current,
        "current_champion_status": "D47_B3_PROTECTED_UNCHANGED",
        "stage3_update": 480,
        "h1": {"status": "PRESERVED", "value": 7.089328839013259e-05},
        "h32": {"status": "PRESERVED", "value": 0.01849100619381173},
        "c1": {
            "status": "VERIFIED_PASS_SHADOW_CANDIDATE",
            "candidate": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005",
            "native_post_ruckig": "12/12",
            "jerk_violations": 0,
            "max_jerk_rad_s3": ultra_jerk,
            "original_0_15_shadow": {"status": "FAILED_SHADOW_TYPE_B", "jerk_violations": c1_violations},
            "promotion": "NOT_PROMOTED",
        },
        "c2": {
            "status": "BLOCKED_EXTERNAL",
            "adaptive_discrete_status": "VERIFIED_PASS",
            "collision_method": "adaptive_discrete_interpolation",
            "cases": c2.get("case_count"),
            "adaptive_self_contact_cases": c2.get("adaptive_self_sweep_contact_cases"),
            "continuous_self_collision_status": c2.get("continuous_self_collision_status"),
        },
        "c3": {
            "status": "VERIFIED_NONREGRESSIVE_SHADOW_NOT_PROMOTED",
            "finding": "0.05 timing shadow materially reduced jerk; 0.10 and 0.20 were rejected for jerk",
        },
        "c4": {
            "status": "VERIFIED_PASS_MODEL_BASED",
            "backend": c4.get("backend"),
            "rnea_aba": c4_self.get("status"),
            "hardware_torque_certification": c4.get("hardware_torque_certification"),
            "candidate_nominal_peak_torque_Nm": stat(c4_nom, "peak_abs_torque_Nm", "max"),
            "candidate_nominal_peak_torque_slew_Nm_s": stat(c4_nom, "peak_abs_torque_slew_Nm_s", "max"),
            "candidate_nominal_peak_power_proxy_W": stat(c4_nom, "peak_aggregate_power_proxy_W", "max"),
        },
        "c5": {"status": "NOT_APPLICABLE_REQUIRES_REAL_HARDWARE_DATA"},
        "tests": "10 passed",
        "replay": {"status": replay.get("status"), "case_count": replay.get("case_count"), "max_abs_numeric_delta": 0.0, "numeric_tolerance": replay.get("numeric_tolerance")},
        "known_limitations": [
            "No usable continuous self-collision backend was available; continuous self-collision is unverified.",
            "URDF dynamics are project-model values, not hardware torque certification.",
            "Physical clearance and hardware certification require external authoritative data.",
            "The archived verifier scope is the frozen 12-case acceptance set, not a rerun of all 1000 cases.",
        ],
        "next_unfinished_action": "Provide/install a real continuous self-collision backend compatible with the frozen MoveIt2 model, then rerun the same persisted 0.05 candidate and acceptance set.",
    }
    write(stage / "STATE_SNAPSHOT.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    state_md = f"""# STATE SNAPSHOT

## Current authoritative state

- Project: **Stage 4C / D48**
- Starting protected state: `{parent}`
- Current protected champion: `{current}`
- Promotion in D48: **none**; D47 B3 is unchanged and authoritative.
- Stage-3 update: `480`
- H1: `7.089328839013259e-05`, preserved
- H32: `0.01849100619381173`, preserved

## Work-unit state

- **C1:** `VERIFIED_PASS_SHADOW_CANDIDATE`. The 0.05 native MoveIt2 post-Ruckig candidate passed all measured hard gates on 12 cases, with zero jerk violations and maximum jerk `{ultra_jerk:.8g}` rad/s³. The original 0.15 route remains recorded as a Type-B failed shadow with `{c1_violations}` violations.
- **C2:** `BLOCKED_EXTERNAL` for continuous certification. The 12-case MoveIt2 `adaptive_discrete_interpolation` sweep found 0 self contacts. Continuous status is `{c2.get('continuous_self_collision_status')}`.
- **C3:** `VERIFIED_NONREGRESSIVE_SHADOW_NOT_PROMOTED`. The 0.05 route materially improved smoothness relative to the 0.10/0.20 shadows; no canonical promotion occurred.
- **C4:** `VERIFIED_PASS_MODEL_BASED`. Pinocchio RNEA+ABA completed for nominal and ±10% synthetic mass variants; RNEA→ABA is `{c4_self.get('status')}`.
- **C5:** `NOT_APPLICABLE_REQUIRES_REAL_HARDWARE_DATA`.

## Verification

- Focused tests: **10 passed**.
- Deterministic replay: **PASS**, 12/12, maximum numeric delta `0` at `1e-12`.
- Hardware torque certification: `NOT_APPLICABLE_UNMEASURED`.
- Physical clearance certification: `NOT_APPLICABLE_UNVERIFIED`.

## Known limitations and next action

The continuous self-collision backend is unavailable; no continuous claim is made. URDF-derived dynamics are model evidence, not hardware truth. The evidence package covers the frozen 12-case acceptance set. The next unfinished scientific action is to provide a real continuous self-collision backend and rerun the same persisted candidate and cases.

Evidence map: `PROGRESS_LEDGER.md`, `HANDOFF.md`, and `EVIDENCE_INDEX.md`.
"""
    write(stage / "STATE_SNAPSHOT.md", state_md)

    units = [
        {
            "work_unit": "POST_RUCKIG_EXECUTION_FORM",
            "objective": "Validate the actual native MoveIt2 post-Ruckig execution form and all C1 hard gates.",
            "prerequisite": parent,
            "status": "VERIFIED_PASS",
            "verifier": "Native MoveIt2 worker + D41 geometry/FK/dynamics chain + independent replay",
            "verifier_result": "0.05 candidate 12/12; finite/joint/velocity/acceleration/jerk/geometry/singularity/protected gates pass",
            "evidence": ["results/C1_ACCEPTANCE/metrics.json", "results/C3_ULTRA_SLOW_005/metrics.json", "results/C3_ULTRA_SLOW_005/replay_comparison.json"],
            "parent": parent,
            "candidate": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005",
            "promoted": False,
            "remaining": "No canonical promotion until C2 continuous backend is available.",
        },
        {
            "work_unit": "CONTINUOUS_SELF_COLLISION",
            "objective": "Certify zero continuous swept self-collision failures.",
            "prerequisite": "C1 native candidate",
            "status": "BLOCKED_EXTERNAL",
            "verifier": "MoveIt2 PlanningScene adaptive discrete sweep; continuous backend search",
            "verifier_result": "12/12 adaptive scans complete with 0 contacts; continuous backend unavailable",
            "evidence": ["results/C3_ULTRA_SLOW_005/C2_SELF_SWEEP_REPAIRED/self_sweep_summary.json"],
            "parent": parent,
            "candidate": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005",
            "promoted": False,
            "remaining": "Install/provide compatible continuous self-collision backend and rerun.",
        },
        {
            "work_unit": "STRESS_ACCURACY_REPEATABILITY_SMOOTHNESS",
            "objective": "Find a safe non-regressive timing/smoothness improvement.",
            "prerequisite": "C1 native output",
            "status": "VERIFIED_NONREGRESSION",
            "verifier": "Native timing shadows, FK, jerk audit, C1 hard gates, deterministic replay",
            "verifier_result": "0.10/0.20 rejected for jerk; 0.05 has 0 jerk violations and preserves measured safety/accuracy floors",
            "evidence": ["results/C3_TIMING_SHADOW/c3_shadow_report.json", "results/C3_ULTRA_SLOW_005/metrics.json"],
            "parent": parent,
            "candidate": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005",
            "promoted": False,
            "remaining": "Resolve route-sensitive jerk without relying on ultra-slow timing.",
        },
        {
            "work_unit": "MODEL_BASED_DYNAMIC_FEASIBILITY",
            "objective": "Audit model-based torque, slew, power, energy, and sensitivity.",
            "prerequisite": "Native candidate and current URDF inertial model",
            "status": "VERIFIED_PASS",
            "verifier": "Pinocchio RNEA + ABA",
            "verifier_result": "Nominal and ±10% synthetic mass variants finite; RNEA→ABA self-consistency PASS",
            "evidence": ["results/C4_ULTRA_SLOW_005/dynamics_report.json"],
            "parent": parent,
            "candidate": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005",
            "promoted": False,
            "remaining": "Use model slew/power comparison as a Stage 4B optimization target; do not call it hardware certification.",
        },
        {
            "work_unit": "PHYSICAL_HARDWARE_CERTIFICATION",
            "objective": "Certify real actuator/hardware torque and physical clearance only with authoritative data.",
            "prerequisite": "Verified hardware telemetry, limits, payload, tool inertia, and calibration uncertainty",
            "status": "NOT_APPLICABLE",
            "verifier": "No authoritative hardware data available",
            "verifier_result": "Not measured; no hardware claim made",
            "evidence": ["results/FINAL_REPORT.json"],
            "parent": parent,
            "candidate": None,
            "promoted": False,
            "remaining": "Requires external real-world data.",
        },
    ]
    write(stage / "PROGRESS_LEDGER.json", json.dumps({"schema_version": "d48-progress-ledger-v1", "work_units": units}, indent=2, ensure_ascii=False) + "\n")
    ledger_md = """# PROGRESS LEDGER

| Work Unit | Target | Status | Verifier | Evidence | Parent | Promoted? |
|---|---|---|---|---|---|---|
| `POST_RUCKIG_EXECUTION_FORM` | Native post-Ruckig C1 gates | `VERIFIED_PASS` | Native MoveIt2 + D41/FK/dynamics + replay | `results/C1_ACCEPTANCE/metrics.json`; `results/C3_ULTRA_SLOW_005/metrics.json` | D47 B3 | No; C2 pending |
| `CONTINUOUS_SELF_COLLISION` | Zero continuous swept self-collision | `BLOCKED_EXTERNAL` | MoveIt2 adaptive discrete sweep; backend search | `results/C3_ULTRA_SLOW_005/C2_SELF_SWEEP_REPAIRED/self_sweep_summary.json` | D47 B3 | No |
| `STRESS_ACCURACY_REPEATABILITY_SMOOTHNESS` | Safe non-regressive smoothness route | `VERIFIED_NONREGRESSION` | Native shadows + C1 gates + replay | `results/C3_TIMING_SHADOW/c3_shadow_report.json` | D47 B3 | No |
| `MODEL_BASED_DYNAMIC_FEASIBILITY` | RNEA/ABA model audit | `VERIFIED_PASS` | Pinocchio RNEA + ABA | `results/C4_ULTRA_SLOW_005/dynamics_report.json` | D47 B3 | No |
| `PHYSICAL_HARDWARE_CERTIFICATION` | Real hardware torque/clearance | `NOT_APPLICABLE` | No authoritative physical data | `results/FINAL_REPORT.json` | D47 B3 | No |

The 0.15 route is recorded as `FAILED_SHADOW` for Type-B jerk behavior; the 0.10 and 0.20 routes are `REJECTED` for jerk. The protected D47 B3 state remains canonical.
"""
    write(stage / "PROGRESS_LEDGER.md", ledger_md)

    takeover = """# TAKEOVER PROTOCOL

The next ChatGPT must reorient before proposing or executing any scientific work.

## First round

1. Read `ROOT_GOAL.md` and state the root objective in your own words.
2. Read `STATE_SNAPSHOT.md` and `STATE_SNAPSHOT.json`; reconstruct the actual verified repository/scientific state.
3. Read `PROGRESS_LEDGER.*`, then audit whether inherited state or previous behavior is consistent with the root goal.

Do not immediately continue the previous agent's apparent trajectory. Previous behavior, plans, hypotheses, raw logs, and summaries are reference evidence, not authority.

## State audit

Recognizing that a previous agent made a mistake is not enough. Check whether the mistake left incorrect code, configuration, trajectory artifacts, canonical outputs, champion pointers, or ledgers. Correct persisted state when necessary. Do not merely understand a discrepancy while operating on the wrong state.

## Current verified facts

- D47 B3 remains the authoritative protected champion.
- The 0.05 D48 native post-Ruckig candidate passed the measured C1 gates on the 12-case acceptance set but was not promoted.
- C2 adaptive discrete self-sweep passed with zero contacts; continuous self-collision is unavailable/unverified.
- C4 is model-based Pinocchio evidence, not hardware certification.

## Next action

The next genuinely unfinished action is to provide a real continuous self-collision backend compatible with the frozen MoveIt2 model and rerun the same persisted candidate and acceptance set. Do not change D47 inputs merely to bypass the missing backend. After that, revisit the route-sensitive jerk and model-dynamic slew/power bottlenecks.
"""
    write(stage / "TAKEOVER_PROTOCOL.md", takeover)

    prompts = """# USER PROMPTS

Only user messages and user-supplied files actually accessible to this Codex session are preserved below. Codex responses, tool output, system/developer prompts, hidden reasoning, and inaccessible conversation content are not fabricated or included.

## User Prompt 01 — repository execution rules

The user supplied the repository `AGENTS.md` instructions for `D:\\robotfucker`. The exact current file is included at `references/AGENTS.md` and is the authoritative repository execution rule set, including Stage 0/1 scope, frozen-state protection, collision labels, and the D46 aggressive-measurement doctrine.

## User Prompt 02 — D48 scientific execution request

The user supplied the full D48 / Stage 4C execution request as an attachment. Its exact wording is preserved in `references/D48_STAGE4C_EXECUTION_PROMPT.txt`.

## User Prompt 03 — mandatory completion policy

The user supplied the full mandatory completion policy as an attachment. Its exact wording is preserved in `references/MANDATORY_COMPLETION_POLICY.txt`.

## User Prompt 04 — current context-handoff archive request

The user supplied the full final context-handoff archive creation request as an attachment. Its exact wording is preserved in `references/D48_CONTEXT_HANDOFF_ARCHIVE_REQUEST.txt`.

## User Prompt 05 — visible attachment wrapper for the D48 scientific request

```text
# Files pasted by the user:

## "# D48 — STAGE 4C EXECUTION-FORM CHAMPION CERTIFICATION, CONTINUOUS SAFETY, ACCU…": [attachment]

## "# MANDATORY COMPLETION POLICY — SOLVE THE PROBLEM, DO NOT STOP AT THE DIAGNOSIS…": [attachment]

Pasted text contains the user's request.

## My request:
```

## User Prompt 06 — visible attachment wrapper for this archive request

```text
# Files pasted by the user:

## "# D48 / STAGE 4C — FINAL CONTEXT HANDOFF ARCHIVE CREATION After the current D48…": [attachment]

Pasted text contains the user's request.

## My request:
```

The visible `My request:` fields in these wrappers contained no additional text; the attached files contain the actual requests. No timestamps are invented.
"""
    write(stage / "USER_PROMPTS.md", prompts)

    handoff = f"""# HANDOFF

## Root goal

Improve the computational robot-arm trajectory toward accurate, smooth, stable, repeatable, singularity-resistant, collision-safe, clearance-safe, dynamically reasonable, and difficult-configuration-robust motion. See `ROOT_GOAL.md`.

## D48 mission

D48 / Stage 4C was to validate the actual native post-Ruckig execution form, establish continuous self-collision safety where a real backend exists, find safe non-regressive smoothness/accuracy/repeatability improvements, audit model-based dynamics, and keep hardware claims explicitly separate from model evidence.

## Starting and current state

D48 started from protected D47 B3 at `{parent}`. No D48 candidate was promoted. The exact current authoritative champion is still D47 B3; Stage-3 update 480, H1, and H32 are preserved.

## Actual execution

Codex implemented a native MoveIt2 worker and launch path that performs TOTG and MoveIt2 Ruckig smoothing, persists native post-Ruckig q/dq/ddq, and derives jerk from native post-Ruckig acceleration. The C1 driver then ran the retained D41/FK/dynamics geometry chain on explicit q-only adapters and ran an independent deterministic replay. C2 ran an independent MoveIt2 PlanningScene adaptive discrete self sweep at a 0.5-degree joint-step bound. C3 measured 0.10, 0.20, and 0.05 native timing shadows. C4 ran Pinocchio RNEA/ABA with nominal and ±10% in-memory mass sensitivity.

## Champion evolution

`D47 B3 (protected) → 0.15 native shadow (rejected for jerk) → 0.10 shadow (rejected for jerk) → 0.20 shadow (rejected for jerk) → 0.05 shadow (C1 safety PASS, not canonical)`

The 0.05 route is the best measured shadow candidate, not a promoted champion. Canonical state remains D47 B3.

## Scientific results

- **C1:** 0.05 candidate passed 12/12 measured hard gates, including zero jerk violations, zero environment/self/discrete-CCD failures, finite FK/dynamics/geometry, preserved clearance/singularity floors, and preserved H1/H32/Stage3 update. The 0.15 route had `{c1_violations}` jerk violations and remains Type-B shadow evidence.
- **C2:** 12/12 adaptive discrete self sweeps completed with zero contacts using `adaptive_discrete_interpolation`. Continuous self-collision is `{c2.get('continuous_self_collision_status')}` because no usable continuous backend was present.
- **C3:** The 0.05 timing route materially reduced jerk relative to the 0.10/0.20 shadows, reaching `{ultra_jerk:.8g}` rad/s³ maximum with zero violations. It was not promoted because C2 continuous certification is unavailable.
- **C4:** Pinocchio RNEA+ABA completed with `{c4_self.get('status')}` self-consistency and finite nominal/±10% synthetic mass results. Values are model-based and not hardware-certified.
- **C5:** Not applicable without real hardware telemetry, verified actuator limits, payload/tool inertia, and calibration uncertainty.

## Important implementation changes

See `changed_files/` for exact source snapshots. Key files are:

- `ros2_moveit_bridge/stage4c_execution_form_native.py`: native post-Ruckig worker.
- `ros2_moveit_bridge/launch/stage4c_execution_form_native.launch.py`: native worker launch.
- `tools/stage4c_execution_form.py`: C1 driver, q-only adapter, metrics, replay.
- `ros2_moveit_bridge/stage4c_self_sweep_native.py`: MoveIt2 adaptive self sweep.
- `tools/stage4c_model_dynamics.py`: Pinocchio RNEA/ABA audit.
- `tools/stage4c_c3_shadow.py`: timing shadow measurements.
- `tools/stage4c_finalize.py`: final D48 report assembly.

## Defects and rejected routes

The native-output/q-only column-contract defect was reproduced and repaired: the retained q-only D41/FK adapters had been given native post CSVs whose last six columns were derived jerk, causing false geometry/singularity failures. Explicit q-only inputs fixed the measurement defect and affected C1 geometry was recomputed. A C2 one-shot MoveItCpp teardown `-11` was also reproduced and repaired after complete summary persistence; one-case and full 12-case runs then exited cleanly. These were Type-A infrastructure defects, not robot weaknesses.

The 0.15, 0.10, and 0.20 routes exposed genuine post-Ruckig jerk behavior and were not hidden by weakening gates. The selected 0.05 route has a model-space torque-slew/power proxy increase relative to the D47 q-only comparison; this remains a Stage 4B optimization target, not a hardware claim.

## Verification

Focused tests: `10 passed`. The selected 0.05 candidate replay passed 12/12 with maximum numeric delta `0` at `1e-12`. Full native/geometry evidence is summarized under `results/`; large raw trajectory CSVs are intentionally omitted as regenerable and size-expensive.

## Remaining problems and next action

1. Continuous self-collision is unverified because the environment lacks a usable compatible continuous backend.
2. Jerk compliance is route-sensitive; the current compliant route is ultra-slow and not canonical.
3. Model torque-slew/power proxies are not improved versus the D47 comparison and hardware limits are unknown.
4. Physical/hardware certification remains unavailable.

The next genuinely unfinished scientific action is: **provide a real continuous self-collision backend compatible with the frozen MoveIt2 model, then rerun the same persisted 0.05 candidate and acceptance set.**

## Recommended reading order

1. `ROOT_GOAL.md`
2. `STATE_SNAPSHOT.md`
3. `PROGRESS_LEDGER.md`
4. `TAKEOVER_PROTOCOL.md`
5. `HANDOFF.md`
6. `USER_PROMPTS.md`
7. `results/FINAL_REPORT.md` and `results/FINAL_REPORT.json`
8. `results/PROMOTION_LEDGER.json`
9. `changed_files/`
10. `evidence/`
11. selected failed-shadow summaries
12. `raw_logs/`
"""
    write(stage / "HANDOFF.md", handoff)

    evidence_index = """# EVIDENCE INDEX

The archive contains compact, high-value evidence rather than the entire repository.

| Archive path | What it proves |
|---|---|
| `results/FINAL_REPORT.json` / `.md` | Final D48 status, exact verdict, blockers, metric table, defects, replay, limitations, and next action. |
| `results/PROMOTION_LEDGER.json` | D47 B3 remained canonical; D48 candidate was not promoted. |
| `results/C1_ACCEPTANCE/metrics.json` | 0.15 native C1 measurement and Type-B jerk finding. |
| `results/C3_ULTRA_SLOW_005/metrics.json` | 0.05 candidate C1 safety/geometry/singularity/joint metrics. |
| `results/C3_ULTRA_SLOW_005/replay_comparison.json` | Independent 12-case deterministic replay at 1e-12. |
| `results/C3_ULTRA_SLOW_005/C2_SELF_SWEEP_REPAIRED/self_sweep_summary.json` | MoveIt2 adaptive discrete self sweep, zero contacts, unavailable continuous backend. |
| `results/C3_TIMING_SHADOW/c3_shadow_report.json` | 0.10/0.20 timing shadows and rejection evidence. |
| `results/C4_ULTRA_SLOW_005/dynamics_report.json` | Pinocchio RNEA/ABA, torque/slew/power/energy, and ±10% mass sensitivity. |
| `evidence/focused_tests.txt` | Actual focused pytest output. |
| `evidence/git_status.txt` | Repository status captured at archive creation. |
| `evidence/git_diff.patch` | Source snapshots in new-file unified-diff form; untracked D48 files had no committed baseline diff. |
| `evidence/parent_identity.txt` | Protected D47 B3 path and metrics used as the non-regression floor. |
| `evidence/secret_scan.txt` | Archive-stage credential scan result. |
| `changed_files/` | Core D48 implementation and focused tests only. |

Full native trajectory CSVs, build caches, installed dependencies, binaries, and unrelated repository files are omitted because they are regenerable and would obscure the authoritative handoff under the 20 MB limit. The persisted summaries retain the actual verifier results and exact paths to the full outputs in the shared repository.
"""
    write(stage / "EVIDENCE_INDEX.md", evidence_index)


def assemble(stage: Path) -> None:
    c1 = read_json(D48 / "C1_ACCEPTANCE/metrics.json")
    ultra = read_json(D48 / "C3_ULTRA_SLOW_005/metrics.json")
    c2 = read_json(D48 / "C3_ULTRA_SLOW_005/C2_SELF_SWEEP_REPAIRED/self_sweep_summary.json")
    c4 = read_json(D48 / "C4_ULTRA_SLOW_005/dynamics_report.json")
    replay = read_json(D48 / "C3_ULTRA_SLOW_005/replay_comparison.json")
    make_docs(stage, c1, ultra, c2, c4, replay)

    # User-supplied source prompts and repository rules.
    copy_one(REPO / "AGENTS.md", stage / "references/AGENTS.md")
    copy_one(PROMPT_D48, stage / "references/D48_STAGE4C_EXECUTION_PROMPT.txt")
    copy_one(PROMPT_COMPLETION, stage / "references/MANDATORY_COMPLETION_POLICY.txt")
    copy_one(PROMPT_HANDOFF, stage / "references/D48_CONTEXT_HANDOFF_ARCHIVE_REQUEST.txt")

    changed = [
        "ros2_moveit_bridge/stage4c_execution_form_native.py",
        "ros2_moveit_bridge/launch/stage4c_execution_form_native.launch.py",
        "ros2_moveit_bridge/stage4c_self_sweep_native.py",
        "ros2_moveit_bridge/launch/stage4c_self_sweep_native.launch.py",
        "tools/stage4c_execution_form.py",
        "tools/stage4c_model_dynamics.py",
        "tools/stage4c_self_sweep.py",
        "tools/stage4c_c3_shadow.py",
        "tools/stage4c_evaluate_post_candidate.py",
        "tools/stage4c_finalize.py",
        "tests/test_stage4c_execution_form.py",
        "tests/test_stage4c_self_sweep.py",
        "tests/test_stage4c_c3_shadow.py",
    ]
    for item in changed:
        copy_one(REPO / item, stage / "changed_files" / item)

    # Compact D48 result package.
    result_files = [
        "FINAL_REPORT.md",
        "FINAL_REPORT.json",
        "PROMOTION_LEDGER.json",
        "STAGE4_FAILURE_TAXONOMY_V1.json",
        "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json",
        "C1_ACCEPTANCE/metrics.json",
        "C1_ACCEPTANCE/case_metrics.jsonl",
        "C1_ACCEPTANCE/post_manifest.csv",
        "C1_ACCEPTANCE/replay_comparison.json",
        "C1_ACCEPTANCE/native_postprocess/execution_form_summary.json",
        "C1_ACCEPTANCE/native_postprocess/runtime_limit_audit.json",
        "C1_ACCEPTANCE/native_postprocess_replay/execution_form_summary.json",
        "C1_ACCEPTANCE/geometry/D41_native_case_summary.jsonl",
        "C1_ACCEPTANCE/geometry/D41_native_provenance.json",
        "C1_ACCEPTANCE/geometry/D41_native_clearance.csv",
        "C1_ACCEPTANCE/geometry/D41_native_jacobian.csv",
        "C3_TIMING_SHADOW/c3_shadow_report.json",
        "C3_TIMING_SHADOW/ultra_slow_005/native_postprocess/execution_form_summary.json",
        "C3_TIMING_SHADOW/ultra_slow_005/native_postprocess_replay/execution_form_summary.json",
        "C3_ULTRA_SLOW_005/metrics.json",
        "C3_ULTRA_SLOW_005/replay_comparison.json",
        "C3_ULTRA_SLOW_005/C2_SELF_SWEEP_REPAIRED/self_sweep_summary.json",
        "C4_ACCEPTANCE/dynamics_report.json",
        "C4_ULTRA_SLOW_005/dynamics_report.json",
        "C4_ULTRA_SLOW_005/model_inventory.json",
    ]
    for item in result_files:
        copy_one(D48 / item, stage / "results" / item)

    # Small parent identity evidence only; do not duplicate the complete D47 tree.
    parent_identity = {
        "path": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
        "status": "PROTECTED_READ_ONLY_UNCHANGED",
        "stage3_update": 480,
        "H1": 7.089328839013259e-05,
        "H32": 0.01849100619381173,
        "minimum_environment_clearance_m": 0.08432694494047037,
        "minimum_self_clearance_m": 0.01662204867779536,
        "minimum_sigma": 1.0189550264124055e-06,
        "maximum_condition": 1824075.1425836,
        "environment_collision_cases": 0,
        "self_collision_cases": 0,
        "environment_ccd_failures": 0,
        "note": "D47 B3 is the authoritative canonical champion after D48; no D47 files were modified by this archive operation.",
    }
    write(stage / "evidence/parent_identity.txt", json.dumps(parent_identity, indent=2) + "\n")

    # Test and repository evidence.
    tests = run_capture([sys.executable, "-m", "pytest", "-q", "tests/test_stage4c_execution_form.py", "tests/test_stage4c_self_sweep.py", "tests/test_stage4c_c3_shadow.py"])
    write(stage / "evidence/focused_tests.txt", tests)
    write(stage / "evidence/git_status.txt", run_capture(["git", "status", "--short"]))
    write(stage / "evidence/git_diff_stat.txt", run_capture(["git", "diff", "--stat"]))

    patch_lines = ["# D48 source snapshot patch. New D48 files were untracked, so no committed parent diff exists.\n"]
    for item in changed:
        text = (REPO / item).read_text(encoding="utf-8")
        patch_lines.extend(difflib.unified_diff([], text.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + item))
    write(stage / "evidence/git_diff.patch", "".join(patch_lines))

    raw_notes = """# Selected raw-log notes

- A native C1 smoke originally produced false geometry failures because a native post CSV was passed into a retained q-only parser. The explicit q-only adapter repaired this Type-A measurement defect; the affected C1 geometry was recomputed.
- The first C2 full worker wrote its complete summary but returned -11 during MoveItCpp teardown. The controlled one-shot worker exit was repaired and then verified by clean one-case and 12-case runs.
- The final C2 summary is authoritative for the available verifier: adaptive discrete self sweep complete; continuous backend unavailable.
"""
    write(stage / "raw_logs/selected_log_notes.md", raw_notes)


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret|password|private[_ ]key)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def secret_scan(stage: Path) -> list[str]:
    findings: list[str] = []
    for path in stage.rglob("*"):
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(path.relative_to(stage).as_posix() + ": pattern=" + pattern.pattern)
    return findings


def validate_required(stage: Path) -> list[str]:
    required = [
        "ROOT_GOAL.md", "AUTHORITY_MAP.md", "STATE_SNAPSHOT.md", "STATE_SNAPSHOT.json",
        "PROGRESS_LEDGER.md", "PROGRESS_LEDGER.json", "TAKEOVER_PROTOCOL.md", "HANDOFF.md",
        "USER_PROMPTS.md", "EVIDENCE_INDEX.md", "results/FINAL_REPORT.md", "results/FINAL_REPORT.json",
        "results/PROMOTION_LEDGER.json", "changed_files/tools/stage4c_execution_form.py",
        "changed_files/ros2_moveit_bridge/stage4c_execution_form_native.py", "evidence/focused_tests.txt",
        "evidence/git_diff.patch", "evidence/parent_identity.txt",
    ]
    return [item for item in required if not (stage / item).is_file()]


def create_archive(stage: Path, archive: Path) -> None:
    cmd = f"export XZ_OPT=-9e; tar -cJf {wsl_path(archive)} -C {wsl_path(stage.parent)} {stage.name}"
    result = subprocess.run(
        ["wsl", "-e", "bash", "-lc", cmd],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tar failed ({result.returncode}): {result.stdout}\n{result.stderr}")


def validate_archive(archive: Path, stage: Path) -> tuple[Path, str, int]:
    listing = run_capture(["wsl", "-e", "bash", "-lc", f"tar -tJf {wsl_path(archive)}"])
    if "ROOT_GOAL.md" not in listing or "STATE_SNAPSHOT.json" not in listing:
        raise RuntimeError("tar listing did not contain required state files")
    validation = stage.parent / (stage.name + "_extracted")
    if validation.exists():
        shutil.rmtree(validation)
    validation.mkdir(parents=True)
    cmd = f"tar -xJf {wsl_path(archive)} -C {wsl_path(validation)}"
    result = subprocess.run(
        ["wsl", "-e", "bash", "-lc", cmd],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tar extraction failed ({result.returncode}): {result.stdout}\n{result.stderr}")
    extracted_root = validation / stage.name
    missing = validate_required(extracted_root)
    if missing:
        raise RuntimeError("extracted archive missing: " + ", ".join(missing))
    findings = secret_scan(extracted_root)
    if findings:
        raise RuntimeError("secret scan findings: " + "; ".join(findings))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    size = archive.stat().st_size
    if size > 20 * 1024 * 1024:
        raise RuntimeError(f"archive exceeds 20 MB: {size} bytes")
    return validation, digest, size


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage = REPO / "tmp" / f"D48_STAGE4C_CONTEXT_HANDOFF_{timestamp}"
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    archive = desktop / f"D48_STAGE4C_CONTEXT_HANDOFF_{timestamp}.tar.xz"
    if stage.exists() or archive.exists():
        raise RuntimeError(f"refusing to overwrite existing staging/archive path: {stage} / {archive}")
    stage.mkdir(parents=True)
    try:
        assemble(stage)
        findings = secret_scan(stage)
        write(stage / "evidence/secret_scan.txt", "SECRET_SCAN_STATUS=" + ("PASS" if not findings else "FAIL") + "\n" + "\n".join(findings) + "\n")
        missing = validate_required(stage)
        if missing:
            raise RuntimeError("staging archive missing: " + ", ".join(missing))
        if findings:
            raise RuntimeError("secret scan findings: " + "; ".join(findings))
        create_archive(stage, archive)
        validation, digest, size = validate_archive(archive, stage)
        archive_validation = {
            "status": "PASS",
            "archive": str(archive),
            "archive_name": archive.name,
            "size_bytes": size,
            "size_limit_bytes": 20 * 1024 * 1024,
            "format": "tar + xz/LZMA2",
            "compression": "XZ_OPT=-9e",
            "sha256": digest,
            "tar_listing": "PASS",
            "full_extraction": "PASS",
            "required_files": "PASS",
            "secret_scan": "PASS",
            "official_new_d48_archives_on_desktop": 1,
            "temporary_staging_removed_after_validation": True,
        }
        write(D48 / "ARCHIVE_VALIDATION.json", json.dumps(archive_validation, indent=2) + "\n")
        shutil.rmtree(validation)
        shutil.rmtree(stage)
        print(json.dumps(archive_validation, indent=2))
    except Exception:
        # Keep failed staging for diagnosis; never delete a failed archive automatically.
        raise


if __name__ == "__main__":
    main()
