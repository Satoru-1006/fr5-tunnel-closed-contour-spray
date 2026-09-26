from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def metric_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _item_by_id(payload: dict[str, Any], item_id: str) -> dict[str, Any]:
    for item in payload.get("items", []):
        if item.get("id") == item_id:
            return item
    return {}


def build_action_plan(root: Path) -> dict[str, Any]:
    final_quality = metric_map(root / "outputs/final_quality_report.csv")
    root_cause = load_json(root / "outputs/ik_root_cause_matrix.json")
    goal_audit = load_json(root / "outputs/goal_resolution_audit.json")

    phase110 = _item_by_id(root_cause, "phase110_branch_patch")
    phase232 = _item_by_id(root_cause, "phase232_bottom_closure_branch")
    tcp_waived = final_quality.get("tool_tcp_acceptance_status") == "waived"

    actions = [
        {
            "id": "load_measured_nozzle_tcp",
            "priority": 1,
            "status": "waived_for_current_scope" if tcp_waived else "waiting_for_external_measurement",
            "why": "The current validation scope explicitly waives the 150 mm placeholder TCP; measured TCP remains optional future evidence.",
            "evidence": {
                "tool_tcp_source": final_quality.get("tool_tcp_source", ""),
                "tool_tcp_xyz": final_quality.get("tool_tcp_xyz", ""),
                "tool_tcp_production_status": final_quality.get("tool_tcp_production_status", ""),
                "tool_tcp_measured_by": final_quality.get("tool_tcp_measured_by", ""),
                "tool_tcp_measured_date": final_quality.get("tool_tcp_measured_date", ""),
                "tool_tcp_calibration_method": final_quality.get("tool_tcp_calibration_method", ""),
                "tool_tcp_acceptance_status": final_quality.get("tool_tcp_acceptance_status", ""),
                "template": "ros2_moveit_bridge/config/tool_tcp_calibration_template.yaml",
            },
            "next_command": (
                "TOOL_TCP_CALIBRATION_YAML=<measured_tcp.yaml> "
                "scripts/run_moveit_strict_validation.sh"
            ),
            "acceptance": [
                "final_quality_report.csv records tool_tcp_source as measured_* or production_*.",
                "The measured TCP YAML includes non-empty measured_by, measured_date in YYYY-MM-DD format, and calibration_method.",
                "tool_tcp_production_status becomes pass.",
            ],
        },
        {
            "id": "rerun_full_strict_validation_after_tcp",
            "priority": 2,
            "status": "not_required_for_current_scope" if tcp_waived else "blocked_by_measured_tcp",
            "why": "Measured TCP rerun is not required for the current scoped validation, but remains the correct future step if the real nozzle TCP replaces the placeholder.",
            "evidence": {
                "strict_audit_status": final_quality.get("strict_audit_status", ""),
                "current_status": final_quality.get("status", ""),
                "collision_status": final_quality.get("collision_status", ""),
                "tcp_speed_production_status": final_quality.get("tcp_speed_production_status", ""),
                "joint_continuity_status": final_quality.get("joint_continuity_status", ""),
            },
            "acceptance": [
                "outputs/audit_goal_requirements_strict.json overall_status is pass.",
                "outputs/final_quality_report.csv status is pass.",
                "collision_status, tcp_speed_production_status, tool_tcp_production_status, and joint_continuity_status are all pass.",
                "python tools/check_production_readiness.py exits with status 0.",
            ],
        },
        {
            "id": "monitor_segmented_reorientation_stops",
            "priority": 3,
            "status": "pass" if final_quality.get("joint_continuity_status") == "pass" else "requires_segmented_process_fix",
            "why": "Large IK branch changes are now explicit spray-off reorientation stops; only spray/process segments are held to the 20 deg continuity gate.",
            "evidence": {
                "trajectory_execution_mode": final_quality.get("trajectory_execution_mode", ""),
                "process_joint_continuity_status": final_quality.get("process_joint_continuity_status", ""),
                "process_max_joint_step_deg": final_quality.get("process_max_joint_step_deg", ""),
                "reorientation_transition_count": final_quality.get("reorientation_transition_count", ""),
                "reorientation_transition_max_interpolated_step_deg": final_quality.get(
                    "reorientation_transition_max_interpolated_step_deg",
                    "",
                ),
                "reorientation_transition_status": final_quality.get("reorientation_transition_status", ""),
                "phase110_diagnostic_context": phase110.get("evidence", {}),
                "phase232_diagnostic_context": phase232.get("evidence", {}),
            },
            "acceptance": [
                "final_quality_report.csv joint_continuity_status is pass.",
                "process_joint_continuity_status is pass with process_max_joint_step_deg <= 20 deg.",
                "reorientation_transition_status is pass and spray_off_transition_status is pass.",
            ],
        },
        {
            "id": "avoid_unbounded_bruteforce_ik",
            "priority": 4,
            "status": "policy_for_next_probes",
            "why": "Previous diversity and wider-beam probes timed out or collided before improving the production gate.",
            "evidence": {
                "root_cause_blocking_items": root_cause.get("blocking_items", []),
                "goal_blocked_items": goal_audit.get("blocked_items", []),
            },
            "acceptance": [
                "Any new IK-search expansion has a bounded timeout and diagnostic CSV.",
                "New probes must test a specific clearance/TCP/path hypothesis instead of only increasing beam width.",
            ],
        },
    ]

    return {
        "schema_version": 1,
        "purpose": "Priority action plan for moving the current fail state toward production strict validation.",
        "overall_status": final_quality.get("status", "missing"),
        "current_passed_gates": [
            gate
            for gate in ("collision_status", "tcp_speed_production_status")
            if final_quality.get(gate) == "pass"
        ],
        "current_blocking_gates": [
            gate
            for gate in ("tool_tcp_production_status", "joint_continuity_status")
            if final_quality.get(gate) != "pass"
            and not (gate == "tool_tcp_production_status" and tcp_waived)
        ],
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prioritized validation action plan.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/validation_action_plan.json")
    args = parser.parse_args()

    payload = build_action_plan(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote validation action plan to {args.out}")


if __name__ == "__main__":
    main()
