#!/usr/bin/env python3
"""Merge independently completed Stage 1.6 MoveIt2 ablation batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONDITIONS = (
    "full_constraints",
    "collision_off",
    "position_only",
    "relaxed_orientation",
    "tcp_roll_scan",
    "standoff_scan",
    "tcp_position_perturbation",
    "increased_single_point_budget",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _classify(rows: list[dict[str, str]], waypoint: int) -> dict[str, Any]:
    by_condition = {row["condition"]: row for row in rows if int(row["waypoint"]) == waypoint}
    perturb = by_condition.get("tcp_position_perturbation", {})
    roll = by_condition.get("tcp_roll_scan", {})
    standoff = by_condition.get("standoff_scan", {})
    relaxed = by_condition.get("relaxed_orientation", {})
    position = by_condition.get("position_only", {})
    budget = by_condition.get("increased_single_point_budget", {})
    collision = by_condition.get("collision_off", {})

    if _bool(perturb.get("accepted_candidate_count", "0")) or int(perturb.get("accepted_candidate_count", "0") or 0) > 0:
        primary = "tcp_path_local_perturbation_repairs"
    elif int(roll.get("accepted_candidate_count", "0") or 0) > 0:
        primary = "tcp_roll_freedom_repairs"
    elif int(standoff.get("accepted_candidate_count", "0") or 0) > 0:
        primary = "spray_distance_adjustment_repairs"
    elif int(relaxed.get("accepted_candidate_count", "0") or 0) > 0:
        primary = "relaxed_orientation_repairs"
    elif _bool(collision.get("ik_found", "false")):
        primary = "collision_infeasible"
    elif _bool(budget.get("ik_found", "false")):
        primary = "search_budget_insufficient"
    elif _bool(position.get("ik_found", "false")):
        primary = "position_reachable_but_orientation_or_process_gate_failed"
    else:
        primary = "not_diagnosed_ik_failure"
    return {
        "waypoint": waypoint,
        "classification": primary,
        "full_constraints_failure_stage": by_condition.get("full_constraints", {}).get("failure_stage"),
        "collision_off_ik_found": collision.get("ik_found"),
        "position_only_ik_found": position.get("ik_found"),
        "relaxed_orientation_accepted": relaxed.get("accepted_candidate_count"),
        "roll_scan_accepted": roll.get("accepted_candidate_count"),
        "standoff_scan_accepted": standoff.get("accepted_candidate_count"),
        "tcp_position_perturbation_accepted": perturb.get("accepted_candidate_count"),
        "increased_budget_ik_found": budget.get("ik_found"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/ik_feasibility_ablation_stage1_6"))
    args = parser.parse_args()
    root = args.root.resolve()
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for condition in CONDITIONS:
        path = root / condition / "ablation_matrix.csv"
        if not path.exists():
            missing.append(condition)
            continue
        for row in _read_csv(path):
            row["batch_dir"] = condition
            rows.append(row)
    if missing:
        raise SystemExit(f"missing Stage 1.6 batches: {', '.join(missing)}")
    rows.sort(key=lambda row: (int(row["waypoint"]), CONDITIONS.index(row["condition"])))
    fields = sorted({key for row in rows for key in row})
    with (root / "ablation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    pair_path = root / "pair_67_68" / "67_68_branch_continuity.csv"
    pair_rows = _read_csv(pair_path) if pair_path.exists() else []
    with (root / "67_68_branch_continuity.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in pair_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(pair_rows)

    classifications = [_classify(rows, waypoint) for waypoint in range(87, 95)]
    summary = {
        "stage": "stage_1_6",
        "diagnostic_only": True,
        "authoritative_input_scope": "181-point open-arch pair under outputs/internal_wiper_moveit_inputs",
        "legacy_720_point_inputs_used": False,
        "off_states_or_reorientation_edges_added": False,
        "gnn_or_rl_added": False,
        "collision_check_type": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
        "dynamics_status": "not_run",
        "post_ruckig_status": "not_run",
        "production_joint_step_limit_deg": 20.0,
        "waypoint_87_94_classification": classifications,
        "branch_67_68": pair_rows,
        "interpretation": {
            "position_perturbation_success": "path/pose point needs joint optimization; this is diagnostic and does not alter the production path",
            "roll_scan_success": "TCP roll freedom matters; do not widen the production constraint in place",
            "standoff_scan_success": "spray distance must be jointly optimized with the path",
            "budget_only_success": "search insufficiency, not physical infeasibility",
            "all_conditions_fail": "closer to IK reachability failure, subject to backend/timeout limits",
        },
    }
    (root / "stage1_6_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# Stage 1.6 IK feasibility ablation",
        "",
        "Diagnostic-only evidence for the 181-point ON-state open arch. Collision is `adaptive_discrete_interpolation`; Bullet CCD and clearance are unavailable. Dynamics and post-Ruckig are not run by this single-point diagnostic.",
        "",
        "## 87-94 matrix",
        "",
        "| waypoint | classification | full failure | position-only IK | relaxed orientation accepted | roll accepted | standoff accepted | TCP perturbation accepted | budget IK |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in classifications:
        report_lines.append(
            f"| {item['waypoint']} | {item['classification']} | {item['full_constraints_failure_stage']} | {item['position_only_ik_found']} | {item['relaxed_orientation_accepted']} | {item['roll_scan_accepted']} | {item['standoff_scan_accepted']} | {item['tcp_position_perturbation_accepted']} | {item['increased_budget_ik_found']} |"
        )
    report_lines += ["", "## 67->68 continuity", "", "| condition | reachable source scope | best max joint step (deg) | <=20 deg |", "|---|---|---:|---|"]
    for row in pair_rows:
        report_lines.append(f"| {row['pair_condition']} | {row.get('source_candidate_scope', '')} | {row['best_max_joint_step_deg']} | {row['within_20deg_gate']} |")
    (root / "stage1_6_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "merged", "root": str(root), "matrix_rows": len(rows), "pair_rows": len(pair_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
