from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


GROUP_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "runtime_bridge_and_strict_gates",
        (
            "ros2_moveit_bridge/plan_closed_contour_moveit.py",
            "ros2_moveit_bridge/launch/fr5_spray_plan_only.launch.py",
            "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
            "ros2_moveit_bridge/config/tool_tcp_calibration_template.yaml",
            "scripts/run_moveit_strict_validation.sh",
            "scripts/verify_current_goal_state.py",
            "tools/audit_goal_requirements.py",
            "tools/build_commit_readiness_plan.py",
            "tools/check_production_readiness.py",
            "tools/build_validation_handoff_manifest.py",
            "tools/publish_final_outputs.py",
            "tools/load_tool_tcp_calibration.py",
            "tools/summarize_moveit_ik_failure.py",
            "tools/summarize_ik_search_diagnostics.py",
            "tools/slice_tcp_pose_window.py",
            "tools/analyze_ik_branch_splice.py",
            "tools/build_ik_root_cause_matrix.py",
            "tools/build_goal_resolution_audit.py",
            "tools/build_validation_action_plan.py",
        ),
    ),
    (
        "tests",
        (
            "tests/test_moveit_bridge_static.py",
            "tests/test_closed_contour.py",
        ),
    ),
    (
        "documentation",
        (
            "README.md",
            "PROJECT_REPORT_README.md",
            "ros2_moveit_bridge/README.md",
            "CHANGESET_SUMMARY.md",
            "README_",
            '"README_',
            ".gitignore",
            "scripts/generate_final_figures.py",
        ),
    ),
    (
        "formal_evidence",
        (
            "outputs/audit_goal_requirements_strict.json",
            "outputs/final_quality_report.csv",
            "outputs/final_acceptance_summary.json",
            "outputs/moveit_joint_step_report.csv",
            "outputs/moveit_ik_search_diagnostics.csv",
            "outputs/moveit_waypoint_joint_trajectory.csv",
            "outputs/moveit_waypoint_joint_step_report.csv",
            "outputs/ik_continuity_segments_summary.json",
            "outputs/segmented_process_summary.json",
            "outputs/segmented_process_plan.csv",
            "outputs/smooth_reorientation_transitions.csv",
            "outputs/validation_handoff_manifest.json",
            "outputs/validation_action_plan.json",
            "outputs/production_readiness_check.json",
            "outputs/commit_readiness_plan.json",
            "outputs/current_goal_state_verification.json",
        ),
    ),
    (
        "probe_evidence",
        (
            "outputs/beam",
            "outputs/bottom_closure_collision_probe_summary.json",
            "outputs/contact_verbose_ik_failure_summary.json",
            "outputs/goal_resolution_audit.json",
            "outputs/ik_beam",
            "outputs/ik_continuity_20deg_probe_summary.json",
            "outputs/ik_diversity_b4_probe_summary.json",
            "outputs/ik_diversity_probe_summary.json",
            "outputs/ik_geometry_probe_summary.json",
            "outputs/ik_root_cause_matrix.json",
            "outputs/ik_roll9_probe_summary.json",
            "outputs/ik_timeout_auto_summary_probe_summary.json",
            "outputs/ik_timeout_diagnostics_probe_summary.json",
            "outputs/phase_window_",
            "outputs/phase110_open_window_splice_analysis.json",
            "outputs/phase110_open_window_ik_probe_summary.json",
            "outputs/phase110_wide_open_nocollision_ik_probe_summary.json",
            "outputs/phase110_wide_open_window_ik_probe_summary.json",
            "outputs/phase110_window_ik_probe_summary.json",
            "outputs/phase210_window_ik_probe_summary.json",
            "outputs/phase_window_probe_plan_summary.json",
            "outputs/phase232_window_ik_probe_summary.json",
            "outputs/phase232_wide_open_nocollision_ik_probe_summary.json",
            "outputs/phase232_wide_open_window_ik_probe_summary.json",
            "outputs/phase237_window_ik_probe_summary.json",
            "outputs/speed_0031_probe_summary.json",
        ),
    ),
    (
        "offline_diagnostic_artifacts",
        (
            "outputs/closed_contour_section.png",
            "outputs/coverage_heatmap.png",
            "outputs/dynamics_jerk.png",
            "outputs/dynamics_main.png",
            "outputs/ik_waypoints.csv",
            "outputs/path_3d.png",
            "outputs/tcp_poses.csv",
            "outputs/tcp_poses_base_link.csv",
        ),
    ),
    (
        "review_separately_preexisting_or_mixed",
        (
            "src/metrics.py",
            "outputs/quality_report.csv",
        ),
    ),
]


def blocking_items(quality: dict[str, str], audit: dict) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if quality.get("status") != "pass":
        items.append({"id": "final_quality_not_pass", "evidence": "outputs/final_quality_report.csv status is not pass."})
    if str(audit.get("overall_status", "")) != "pass":
        items.append({"id": "strict_audit_not_pass", "evidence": "outputs/audit_goal_requirements_strict.json is not pass."})
    if quality.get("collision_status") != "pass":
        items.append({"id": "collision_required", "evidence": "Current reports do not pass the MoveIt collision gate."})
    if quality.get("tcp_speed_production_status") != "pass":
        items.append({"id": "tcp_speed_required", "evidence": "Current reports do not pass the configured TCP speed gate."})
    tcp_waived = quality.get("tool_tcp_acceptance_status") == "waived"
    if quality.get("tool_tcp_production_status") != "pass" and not tcp_waived:
        items.append({"id": "measured_tool_tcp_required", "evidence": "TCP is neither measured/production nor explicitly waived."})
    if quality.get("joint_continuity_status") != "pass":
        items.append({"id": "joint_continuity_required", "evidence": "Current reports still fail the accepted joint-continuity gate."})
    return items


def metric_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_status(root: Path) -> list[dict[str, str]]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    entries = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2].strip()
        path = line[3:].replace("\\", "/")
        entries.append({"status": status, "path": path})
    return entries


def classify_path(path: str) -> str:
    for group, patterns in GROUP_RULES:
        for pattern in patterns:
            if path == pattern or path.startswith(pattern):
                return group
    return "unclassified"


def grouped_changes(entries: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups = {group: [] for group, _ in GROUP_RULES}
    groups["unclassified"] = []
    for entry in entries:
        groups[classify_path(entry["path"])].append(entry)
    return {group: values for group, values in groups.items() if values}


def build_manifest(root: Path, quality_csv: Path, audit_json: Path) -> dict:
    quality = metric_map(quality_csv)
    audit = load_json(audit_json)
    entries = git_status(root)
    return {
        "schema_version": 1,
        "purpose": "Machine-readable handoff for the current MoveIt2 strict-validation worktree.",
        "overall_status": quality.get("status", "missing"),
        "strict_audit_status": str(audit.get("overall_status", "missing")),
        "current_gates": {
            "collision_status": quality.get("collision_status", "missing"),
            "tcp_speed_production_status": quality.get("tcp_speed_production_status", "missing"),
            "tool_tcp_production_status": quality.get("tool_tcp_production_status", "missing"),
            "joint_continuity_status": quality.get("joint_continuity_status", "missing"),
            "waypoint_post_joint_continuity_match_status": quality.get(
                "waypoint_post_joint_continuity_match_status",
                "missing",
            ),
        },
        "key_metrics": {
            "tool_tcp_xyz": quality.get("tool_tcp_xyz", ""),
            "tool_tcp_rpy": quality.get("tool_tcp_rpy", ""),
            "tool_tcp_source": quality.get("tool_tcp_source", ""),
            "tool_tcp_measured_by": quality.get("tool_tcp_measured_by", ""),
            "tool_tcp_measured_date": quality.get("tool_tcp_measured_date", ""),
            "tool_tcp_calibration_method": quality.get("tool_tcp_calibration_method", ""),
            "fk_tcp_speed_mean_m_s": quality.get("fk_tcp_speed_mean_m_s", ""),
            "max_joint_step_deg": quality.get("max_joint_step_deg", ""),
            "max_joint_step_joint": quality.get("max_joint_step_joint", ""),
            "max_ik_joint_step_deg": quality.get("max_ik_joint_step_deg", "not_recorded_legacy_run"),
            "production_joint_step_limit_deg": quality.get("production_joint_step_limit_deg", ""),
            "joint_step_exceeding_count": quality.get("joint_step_exceeding_count", ""),
            "joint_step_exceeding_transition_count": quality.get("joint_step_exceeding_transition_count", ""),
            "trajectory_execution_mode": quality.get("trajectory_execution_mode", ""),
            "process_joint_continuity_status": quality.get("process_joint_continuity_status", ""),
            "process_max_joint_step_deg": quality.get("process_max_joint_step_deg", ""),
            "reorientation_transition_count": quality.get("reorientation_transition_count", ""),
            "reorientation_transition_max_interpolated_step_deg": quality.get(
                "reorientation_transition_max_interpolated_step_deg",
                "",
            ),
            "reorientation_transition_status": quality.get("reorientation_transition_status", ""),
            "collision_count": quality.get("collision_count", ""),
            "first_collision_index": quality.get("first_collision_index", ""),
            "include_bottom_closure_collision": quality.get("include_bottom_closure_collision", ""),
        },
        "blocking_items": blocking_items(quality, audit),
        "changed_file_count": len(entries),
        "changed_files": grouped_changes(entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a machine-readable validation handoff manifest.")
    parser.add_argument("--quality-csv", type=Path, default=ROOT / "outputs/final_quality_report.csv")
    parser.add_argument("--audit-json", type=Path, default=ROOT / "outputs/audit_goal_requirements_strict.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/validation_handoff_manifest.json")
    args = parser.parse_args()

    payload = build_manifest(ROOT, args.quality_csv, args.audit_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote validation handoff manifest to {args.out}")


if __name__ == "__main__":
    main()
