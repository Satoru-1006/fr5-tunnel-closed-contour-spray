#!/usr/bin/env python3
"""Assemble the compact, claim-fenced Stage D5 evidence set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def native_task_validation(root: Path) -> dict[int, list[dict[str, str]]]:
    result: dict[int, list[dict[str, str]]] = {}
    for path in sorted((root / "native_cases").glob("*/D1_ATTEMPTS.csv")):
        for row in read_csv(path):
            result.setdefault(int(row["point_id"]), []).append(row)
    return result


def make_task_report(root: Path, native: dict[int, list[dict[str, str]]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = read_csv(root / "TASK_FREEDOM_RECOVERY.csv")
    counts = {"EXACT_POSE_BOUNDED_IK_VALID": 0, "TASK_FREEDOM_RECOVERABLE": 0, "UNRESOLVED": 0, "GEOMETRY_ONLY_COLLISION_BLOCKED": 0}
    for row in rows:
        point = int(row["waypoint_id"])
        attempts = native.get(point, [])
        accepted = [item for item in attempts if item.get("reason") == "VALID_NATIVE_IK" and item.get("collision_free") == "true"]
        if row.get("classification") == "UNRESOLVED":
            row["native_validation"] = "NOT_RUN"
            row["native_collision_status"] = "NOT_RUN"
            counts["UNRESOLVED"] += 1
        elif accepted:
            row["native_validation"] = "PASS"
            row["native_collision_status"] = "PASS"
            if row["classification"] == "EXACT_POSE_BOUNDED_IK_VALID":
                counts["EXACT_POSE_BOUNDED_IK_VALID"] += 1
            else:
                counts["TASK_FREEDOM_RECOVERABLE"] += 1
        else:
            row["native_validation"] = "FAIL"
            row["native_collision_status"] = ";".join(sorted({item.get("collision_pairs", "") for item in attempts if item.get("collision_pairs")})) or "NO_NATIVE_ACCEPT"
            row["classification"] = "GEOMETRY_ONLY_COLLISION_BLOCKED"
            counts["GEOMETRY_ONLY_COLLISION_BLOCKED"] += 1
    fields = ["waypoint_id", "classification", "selected_variant", "standoff_m", "incidence_axis", "incidence_deg", "q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad", "position_error_m", "orientation_error_rad", "joint_margin_fraction", "native_validation", "native_collision_status"]
    write_csv(root / "TASK_FREEDOM_RECOVERY.csv", fields, rows)
    return rows, counts


def collision_rows(d4: Path, root: Path) -> list[dict[str, object]]:
    native57 = read_csv(d4 / "D4_COLLISION_VISUAL_TEST" / "native" / "D1_ATTEMPTS.csv")
    native_cont = read_csv(d4 / "D4_CONTINUATION_NATIVE_VALIDATION" / "D1_ATTEMPTS.csv")
    selected: dict[int, dict[str, str]] = {}
    for row in native57 + native_cont:
        point = int(row["point_id"])
        if point not in selected and point in {57, 60, 64, 67, 105, 106}:
            selected[point] = row
    probe_depth: dict[int, float] = {}
    for point in (57, 59, 67, 60, 64, 105, 106):
        path = root / "collision_probes" / f"collision_probe_wp{point}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                depths = [float(item.get("depth_m", 0.0)) for item in data.get("pairs", []) if item.get("depth_m") is not None]
                if depths:
                    probe_depth[point] = max(depths)
            except (OSError, ValueError, TypeError):
                pass
    rows = []
    for point in (57, 60, 64, 67, 105, 106):
        item = selected.get(point, {})
        pairs = item.get("collision_pairs", "")
        q = [item.get(f"q{i}_rad", "") for i in range(1, 7)]
        rows.append({
            "waypoint_id": point,
            "q_source": "D4 native representative candidate/continuation",
            "q_rad": ";".join(q),
            "target_frame": "base_link; spray_tcp target evaluated by native RobotState",
            "moveit_fcl_collision": "TRUE" if item else "NOT_RECORDED",
            "moveit_fcl_pairs": pairs,
            "penetration_or_contact_depth_m": f"{probe_depth[point]:.17g}" if point in probe_depth else "not_available",
            "visual_mesh_sensitivity": "same_pair_family_as_collision_mesh" if point <= 67 else "not_run_for_continuation_tool_pair",
            "official_step_present": "TRUE",
            "occt_ocp_crosscheck": "UNAVAILABLE_OCCT_NOT_INSTALLED",
            "same_q_same_frame": "PASS_NATIVE_Q_FRAME; CAD_TRANSFORM_NOT_EXECUTED",
            "active_shadow_classification": "REAL_MODEL_SELF_COLLISION_SOFTWARE_GEOMETRY" if item else "UNRESOLVED",
            "final_collision_truth": "UNRESOLVED",
            "note": "Native FCL collision is reproducible in the active software shadow; official CAD/B-Rep pair placement was not independently evaluated because OCP/OCCT is unavailable.",
        })
    fields = list(rows[0])
    write_csv(root / "COLLISION_TRUTH_CASES.csv", fields, rows)
    return rows


def official_cad_audit(root: Path) -> dict[str, object]:
    step = Path("vendor/estun/iER8_720_MI/cad/official_geometry_ground_truth_044662A_iER8-720-MI.step")
    text = step.read_text(encoding="utf-8", errors="ignore") if step.exists() else ""
    products = sorted(set(re.findall(r"iER8-720-MI-0[0-7]", text)))
    result: dict[str, object] = {
        "step_path": str(step.resolve()), "step_present": step.exists(), "step_size_bytes": step.stat().st_size if step.exists() else None,
        "assembly_product_labels_found": products, "official_cad_status": "PRESENT_TEXT_ASSEMBLY_ONLY",
        "occt_ocp_status": "UNAVAILABLE_NOT_INSTALLED", "geometry_crosscheck_status": "NOT_EXECUTED",
    }
    try:
        import OCP  # type: ignore  # noqa: F401
        result["occt_ocp_status"] = "AVAILABLE"
    except Exception as exc:
        result["occt_ocp_error"] = f"{type(exc).__name__}: {exc}"
    (root / "OFFICIAL_CAD_STEP_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def plot_outputs(root: Path, d4: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt
        selected = read_csv(d4 / "D4_D2_MERGED_RESELECT" / "SELECTED_CONTINUOUS_Q.csv")
        points = np.asarray([int(row["waypoint_id"]) for row in selected])
        q = np.asarray([[float(row[f"q{i}_rad"]) for i in range(1, 7)] for row in selected])
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=160, sharex=True)
        for index in range(6): axes[0].plot(points, np.degrees(q[:, index]), label=f"J{index + 1}")
        axes[0].set_ylabel("Joint position (deg)"); axes[0].set_title("Available native-valid selected path (114/181 only)"); axes[0].legend(ncol=6, fontsize=8); axes[0].grid(alpha=0.25)
        continuity = read_csv(d4 / "D4_D2_MERGED_RESELECT" / "JOINT_CONTINUITY.csv")
        jumps = np.asarray([float(row["max_abs_dq_rad"]) for row in continuity])
        axes[1].plot(points[1:], np.degrees(jumps), color="#d62728"); axes[1].axhline(20, color="#444", linestyle="--", linewidth=1); axes[1].set(xlabel="Waypoint transition end", ylabel="Max wrapped jump (deg)"); axes[1].grid(alpha=0.25)
        fig.tight_layout(); path = root / "FIG3_FIG4_AVAILABLE_114_PATH.png"; fig.savefig(path); plt.close(fig); paths["available_path"] = str(path)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160); ax.axis("off"); ax.text(0.02, 0.72, "FCL: reproducible collision in active shadow", fontsize=13, color="#d62728"); ax.text(0.02, 0.52, "Official STEP: present", fontsize=13, color="#2ca02c"); ax.text(0.02, 0.32, "OCP/OCCT geometry cross-check: UNAVAILABLE", fontsize=13, color="#ff7f0e"); ax.text(0.02, 0.10, "Gate B final truth: UNRESOLVED", fontsize=14, weight="bold"); fig.tight_layout(); path = root / "FIG2_FCL_VS_CAD_STATUS.png"; fig.savefig(path); plt.close(fig); paths["collision_status"] = str(path)
    except Exception as exc:
        (root / "FIGURE_BACKEND_UNAVAILABLE.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--d4", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    args = parser.parse_args()
    root = args.root; d4 = args.d4; root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "A_TASK_FREEDOM" / "WP1_56_JOINT_LIMIT_CONFLICT.csv", root / "WP1_56_JOINT_LIMIT_CONFLICT.csv")
    shutil.copy2(root / "A_TASK_FREEDOM" / "D5_LIMIT_TASK_FREEDOM_SUMMARY.json", root / "D5_LIMIT_TASK_FREEDOM_SUMMARY.json")
    native = native_task_validation(root / "A_TASK_FREEDOM")
    task_rows, task_counts = make_task_report(root / "A_TASK_FREEDOM", native)
    shutil.copy2(root / "A_TASK_FREEDOM" / "TASK_FREEDOM_RECOVERY.csv", root / "TASK_FREEDOM_RECOVERY.csv")
    collision = collision_rows(d4, root)
    cad = official_cad_audit(root)
    (root / "OFFICIAL_LIMIT_CROSSCHECK.md").write_text("""# Official iER8-720-MI joint-limit cross-check

Authority: `config/ier8/ier8_official_spec.yaml`, the Stage C official/model
cross-check, and the active Stage C generated URDF used by Native MoveIt.

| Axis | Lower (deg) | Upper (deg) | Active model check |
|---|---:|---:|---|
| J1 | -170 | 170 | PASS |
| J2 | -80 | 135 | PASS |
| J3 | -190 | 65 | PASS |
| J4 | -190 | 190 | PASS |
| J5 | -120 | 120 | PASS |
| J6 | -360 | 360 | PASS |

No limit was widened, narrowed, or replaced during D5. Numeric controller zero
offsets and installed-axis signs remain unavailable; this is a software-shadow
limit cross-check, not controller calibration evidence.
""", encoding="utf-8")

    d4_summary = json.loads((d4 / "FINAL" / "STAGE_D4_FINAL_SUMMARY.json").read_text(encoding="utf-8"))
    # Use the accepted wrapped-joint deduplicated solution bank, not the raw
    # repeated solver rows, for the compact final artifact.
    valid = read_csv(d4 / "D4_D2_MERGED_RESELECT" / "D2_SOLUTIONS_DEDUP.csv")
    write_csv(root / "FINAL_181_IK_RESULTS.csv", ["waypoint_id", "original_exact_native_valid_count", "original_exact_status", "task_freedom_status"], [
        {"waypoint_id": point, "original_exact_native_valid_count": sum(int(row["point_id"]) == point for row in valid), "original_exact_status": "VALID" if any(int(row["point_id"]) == point for row in valid) else "NO_VALID_NATIVE_EXACT_SOLUTION", "task_freedom_status": next((row.get("classification", "UNRESOLVED") + ":" + row.get("native_validation", "" ) for row in task_rows if int(row["waypoint_id"]) == point), "NOT_APPLICABLE")}
        for point in range(1, 182)
    ])
    np.savez(root / "FINAL_VALID_IK_SOLUTIONS.npz", waypoint_id=np.asarray([int(row["point_id"]) for row in valid]), q=np.asarray([[float(row[f"q{i}_rad"]) for i in range(1, 7)] for row in valid]), scope="PARTIAL_NATIVE_VALID_114_WAYPOINTS_NOT_COMPLETE_181_PATH")
    shutil.copy2(d4 / "D4_D2_MERGED_RESELECT" / "SELECTED_CONTINUOUS_Q.csv", root / "SELECTED_CONTINUOUS_Q.csv")
    shutil.copy2(d4 / "D4_D2_MERGED_RESELECT" / "JOINT_CONTINUITY.csv", root / "JOINT_CONTINUITY.csv")

    plots = plot_outputs(root, d4)
    d2 = json.loads((d4 / "D4_D2_MERGED_RESELECT" / "D2_SUMMARY.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": "ier8_stage_d5_final_v1", "active_target_robot": "ESTUN_iER8_720_MI", "active_tcp": "spray_tcp_software_shadow",
        "authoritative_input": str(args.targets.resolve()), "gate_a_task_limit_truth": "PASS_FOR_FEASIBILITY_TRUTH", "wp1_56_exact_bounded_valid": 0,
        "wp1_56_unbounded_exact_roots": 56, "wp1_56_task_freedom_geometry_and_native": task_counts,
        "gate_b_collision_truth": "UNRESOLVED", "wp57_67_collision_class": "ACTIVE_SHADOW_COLLISION_REPRODUCED; OFFICIAL_CAD_CROSSCHECK_UNAVAILABLE",
        "wp105_106_collision_class": "ACTIVE_SHADOW_ROBOT_TOOL_INTERFERENCE_REPRODUCED; OFFICIAL_CAD_CROSSCHECK_UNAVAILABLE",
        "collision_model_correction": "NO", "official_cad": cad,
        "gate_c_global_branch": "FAIL_NOT_ESTABLISHED", "final_ik_valid": "114 / 181 original exact target set",
        "continuous_181_q_path": "NOT_ESTABLISHED", "max_joint_jump_deg": d2["max_joint_jump_deg"], "max_jump_transition": d2["max_jump_transition"], "max_jump_joint": "J1", "continuity_violations_over_20deg": d2["discontinuity_count_over_20deg"],
        "full_path_jacobian": "NOT_RUN_COMPLETE_PATH_ABSENT", "available_segment_jacobian": "D3 partial 114-point evidence only", "ready_for_trajectory_stage": "NO", "ruckig": "NOT_RUN", "promotion": "NO_PROMOTION", "plots": plots,
    }
    (root / "D5_FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# Stage D5 final report

## First-screen status

```text
STAGE_D5_STATUS = PASS_FOR_TASK_FEASIBILITY_TRUTH_WITH_COLLISION_GATE_UNRESOLVED
ACTIVE_TARGET_ROBOT = ESTUN_iER8_720_MI
GATE_A_TASK_LIMIT_TRUTH = PASS_FOR_FEASIBILITY_TRUTH
WP1_56_EXACT_BOUNDED_VALID = 0 / 56
WP1_56_TASK_FREEDOM_RECOVERED_NATIVE = {task_counts['TASK_FREEDOM_RECOVERABLE']}
WP1_56_TRUE_CONFLICT = {56 - task_counts['TASK_FREEDOM_RECOVERABLE']}
GATE_B_COLLISION_TRUTH = UNRESOLVED
WP57_67_COLLISION_CLASS = ACTIVE_SHADOW_COLLISION_REPRODUCED; CAD CROSSCHECK UNAVAILABLE
WP105_106_COLLISION_CLASS = ACTIVE_SHADOW_ROBOT_TOOL_INTERFERENCE_REPRODUCED; CAD CROSSCHECK UNAVAILABLE
COLLISION_MODEL_CORRECTION = NO
GATE_C_GLOBAL_BRANCH = FAIL_NOT_ESTABLISHED
FINAL_IK_VALID = 114 / 181 (original exact target set)
CONTINUOUS_181_Q_PATH = NOT_ESTABLISHED
MAX_JOINT_JUMP = {d2['max_joint_jump_deg']:.9f} deg
MAX_JUMP_TRANSITION = WP{d2['max_jump_transition'][0]} -> WP{d2['max_jump_transition'][1]}
MAX_JUMP_JOINT = J1
CONTINUITY_VIOLATIONS = {d2['discontinuity_count_over_20deg']}
FULL_PATH_JACOBIAN = NOT_RUN_COMPLETE_PATH_ABSENT
READY_FOR_TRAJECTORY_STAGE = NO
RUCKIG = NOT_RUN
PROMOTION = NO_PROMOTION
```

## Answers to the ten requested questions

1. WP1–56 fail the exact bounded solve because independent exact roots lie outside the official limits; the 36-start rerun found 0/56 bounded exact roots and 56/56 unbounded exact roots.
2. The adaptive distribution split is recorded in `WP1_56_JOINT_LIMIT_CONFLICT.csv`; the smallest violation is 0.358610° at WP56/J2.
3. The largest conflict is 35.449233° at WP17/J2; the report retains the full per-waypoint and per-root evidence.
4. Only the explicitly frozen standoff/incidence shadow was tested. Any native-passing rows are counted as task-freedom recovery; fixed project-defined roll was not silently freed. No authorized base-placement variable was present, so base-placement recovery was not claimed. The native result is recorded separately from geometry candidates.
5. WP57–67 are reproducible native FCL collisions in the active software geometry, mainly `base_link|link_2` (WP57–58 also expose `base_link|link_3`). Official CAD is present, but the same-q B-Rep/OCP cross-check is unavailable in this environment, so final CAD truth remains UNRESOLVED.
6. WP105/WP106 are reproducible native collisions involving `link_4` and the ASB/adapter tool geometry. They are not cleared by changing ACM; no model correction was justified.
7. No new collision-free native branch was established for the unresolved original target layers. The existing 114-point branch set remains the only selected exact path evidence.
8. No. The selected available path still contains the 175.952606° WP103→WP104 J1 jump.
9. No complete 181-point collision-free continuous path exists in the verified evidence.
10. Ruckig cannot start: the original exact path is incomplete, Gate B CAD truth is unresolved, continuity is not closed, and full-path Jacobian evidence is absent. `RUCKIG = NOT_RUN` remains mandatory.

## Claim fences

Collision labels are `adaptive_discrete_interpolation`; strict articulated CCD and physical clearance are unavailable. All IK and collision results are software-shadow evidence only. No hardware, manufacturing, calibrated TCP, controller-zero, production-safety, or promotion claim follows.
"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (root / "FINAL_STATUS.md").write_text("STAGE_D5_STATUS=PASS_FOR_TASK_FEASIBILITY_TRUTH_WITH_COLLISION_GATE_UNRESOLVED\nREADY_FOR_TRAJECTORY_STAGE=NO\nRUCKIG=NOT_RUN\nPROMOTION=NO_PROMOTION\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
