from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def _contact_pairs(summary: dict[str, Any]) -> list[str]:
    return [str(item.get("pair", "")) for item in summary.get("contact_pairs", []) if item.get("pair")]


def build_matrix(root: Path) -> dict[str, Any]:
    final_quality = metric_map(root / "outputs/final_quality_report.csv")
    phase110_collision = load_json(root / "outputs/phase110_wide_open_window_ik_probe_summary.json")
    phase110_no_collision = load_json(root / "outputs/phase110_wide_open_nocollision_ik_probe_summary.json")
    phase232_collision = load_json(root / "outputs/phase232_wide_open_window_ik_probe_summary.json")
    phase232_no_collision = load_json(root / "outputs/phase232_wide_open_nocollision_ik_probe_summary.json")
    phase110_splice = load_json(root / "outputs/phase110_open_window_splice_analysis.json")

    items = [
        {
            "id": "formal_collision_gate",
            "status": final_quality.get("collision_status", "missing"),
            "evidence": {
                "collision_count": final_quality.get("collision_count", ""),
                "first_collision_index": final_quality.get("first_collision_index", ""),
                "include_bottom_closure_collision": final_quality.get("include_bottom_closure_collision", ""),
                "source": "outputs/final_quality_report.csv",
            },
            "interpretation": "Current simplified MoveIt collision report passes; this no longer matches the original 145/145 collision failure.",
            "next_action": "Keep as pass for the simplified model, but rerun after measured TCP and real workcell collision geometry.",
        },
        {
            "id": "tool_tcp_gate",
            "status": final_quality.get("tool_tcp_production_status", "missing"),
            "evidence": {
                "tool_tcp_source": final_quality.get("tool_tcp_source", ""),
                "tool_tcp_xyz": final_quality.get("tool_tcp_xyz", ""),
                "source": "outputs/final_quality_report.csv",
            },
            "interpretation": "The current 150 mm TCP is still a placeholder, so production validation cannot pass.",
            "next_action": "Load a measured nozzle TCP YAML and rerun strict validation.",
        },
        {
            "id": "production_speed_gate",
            "status": final_quality.get("tcp_speed_production_status", "missing"),
            "evidence": {
                "fk_tcp_speed_mean_m_s": final_quality.get("fk_tcp_speed_mean_m_s", ""),
                "production_min_tcp_speed_m_s": final_quality.get("production_min_tcp_speed_m_s", ""),
                "source": "outputs/final_quality_report.csv",
            },
            "interpretation": "Current formal report clears the production minimum TCP speed gate.",
            "next_action": "Keep monitoring speed after TCP/path changes.",
        },
        {
            "id": "phase110_branch_patch",
            "status": "blocked_by_collision_and_fk_quality",
            "evidence": {
                "collision_on_status": phase110_collision.get("status", "missing"),
                "collision_on_first_failure_source_phase": phase110_collision.get("first_failure_source_phase", ""),
                "collision_on_contacts": _contact_pairs(phase110_collision),
                "no_collision_ik_search_status": phase110_no_collision.get("ik_search_status", "missing"),
                "no_collision_max_joint_step_deg": phase110_no_collision.get("max_joint_step_deg", ""),
                "no_collision_moveit_quality_status": phase110_no_collision.get("moveit_quality_status", ""),
                "no_collision_fk_standoff_error_max_abs_mm": phase110_no_collision.get(
                    "fk_standoff_error_max_abs_mm",
                    "",
                ),
                "best_direct_splice_status": phase110_splice.get("best_splice_trial", {}).get("status", ""),
                "best_direct_splice_max_step_deg": phase110_splice.get("best_splice_trial", {}).get(
                    "max_step_deg",
                    "",
                ),
            },
            "interpretation": (
                "The local phase-110 branch is continuous only as a diagnostic branch. Direct splice fails, wider "
                "collision-on entry collides, and no-collision completion fails coating FK quality."
            ),
            "next_action": "Change tool pose/path clearance or rerun with measured TCP; do not promote the local branch as a patch.",
        },
        {
            "id": "phase232_bottom_closure_branch",
            "status": "blocked_by_collision_then_search_cost",
            "evidence": {
                "collision_on_status": phase232_collision.get("status", "missing"),
                "collision_on_first_failure_source_phase": phase232_collision.get("first_failure_source_phase", ""),
                "collision_on_contacts": _contact_pairs(phase232_collision),
                "no_collision_ik_search_status": phase232_no_collision.get("ik_search_status", "missing"),
                "no_collision_completed_ok_count": phase232_no_collision.get("completed_ok_count", ""),
                "no_collision_timeout_before_local_index": phase232_no_collision.get(
                    "timeout_before_local_index",
                    "",
                ),
                "no_collision_best_bottleneck_step_deg": phase232_no_collision.get(
                    "best_bottleneck_step_deg_max_ok",
                    "",
                ),
            },
            "interpretation": (
                "The bottom-closure transition has an early collision/clearance barrier. Once collision is disabled, "
                "the probe progresses much farther but remains search-cost limited."
            ),
            "next_action": "Prioritize clearance/path/TCP changes before spending more effort on wider IK search.",
        },
    ]
    blocking = [
        item["id"]
        for item in items
        if item["status"] not in {"pass"} and item["id"] not in {"formal_collision_gate", "production_speed_gate"}
    ]
    return {
        "schema_version": 1,
        "purpose": "Root-cause matrix for the current 150 mm placeholder TCP MoveIt2 strict-validation state.",
        "overall_status": final_quality.get("status", "missing"),
        "root_cause_summary": (
            "Collision and production speed are pass in the current simplified formal report, but production remains "
            "blocked by placeholder TCP and by IK continuity/clearance interactions around phase 110 and bottom closure."
        ),
        "blocking_items": blocking,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current IK/collision root-cause matrix.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ik_root_cause_matrix.json")
    args = parser.parse_args()

    payload = build_matrix(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote IK root-cause matrix to {args.out}")


if __name__ == "__main__":
    main()
