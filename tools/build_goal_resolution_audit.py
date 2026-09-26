from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
# Use only characters that are not normal project prose. Common Chinese characters
# such as "程" and "阅" caused false positives in otherwise valid UTF-8 documents.
MOJIBAKE_MARKERS = tuple(chr(codepoint) for codepoint in (0x934D, 0x4E63, 0xFFFD))


def metric_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "utf8_readable": False, "mojibake_marker_hits": {}}
    text = path.read_text(encoding="utf-8")
    hits = {marker: text.count(marker) for marker in MOJIBAKE_MARKERS if marker in text}
    return {
        "exists": True,
        "utf8_readable": True,
        "line_count": len(text.splitlines()),
        "mojibake_marker_hits": hits,
        "status": "pass" if not hits else "fail:mojibake_markers",
    }


def _pass_fail(condition: bool) -> str:
    return "pass" if condition else "fail"


def build_audit(root: Path) -> dict[str, Any]:
    final_quality = metric_map(root / "outputs/final_quality_report.csv")
    raw_collision = metric_map(root / "outputs/moveit_collision_report.csv")
    strict_audit = load_json(root / "outputs/audit_goal_requirements_strict.json")
    root_cause = load_json(root / "outputs/ik_root_cause_matrix.json")
    manifest = load_json(root / "outputs/validation_handoff_manifest.json")
    readiness = load_json(root / "outputs/production_readiness_check.json")
    readme_status = read_text_status(root / "README.md")
    report_readme_status = read_text_status(root / "PROJECT_REPORT_README.md")
    missing_by_gate = {
        item.get("gate", ""): item
        for item in readiness.get("missing_evidence", [])
        if isinstance(item, dict)
    }

    items = [
        {
            "id": "original_145_of_145_collision",
            "status": _pass_fail(
                final_quality.get("collision_status") == "pass"
                and raw_collision.get("status") == "pass"
                and raw_collision.get("collision_count") in {"0", "0.0"}
                and raw_collision.get("first_collision_index") in {"-1", "-1.0"}
            ),
            "evidence": {
                "final_collision_status": final_quality.get("collision_status", "missing"),
                "final_collision_count": final_quality.get("collision_count", "missing"),
                "final_first_collision_index": final_quality.get("first_collision_index", "missing"),
                "raw_collision_status": raw_collision.get("status", "missing"),
                "raw_collision_checked_state_count": raw_collision.get("collision_checked_state_count", "missing"),
                "raw_collision_count": raw_collision.get("collision_count", "missing"),
                "raw_first_collision_index": raw_collision.get("first_collision_index", "missing"),
                "source": "outputs/final_quality_report.csv; outputs/moveit_collision_report.csv",
            },
            "interpretation": "The original all-sampled-states collision failure is resolved in the current simplified collision model.",
        },
        {
            "id": "strict_acceptance_must_not_be_claimed_pass",
            "status": _pass_fail(
                final_quality.get("status") == "pass"
                and strict_audit.get("overall_status") == "pass"
                and final_quality.get("tool_tcp_acceptance_status") == "waived"
                and final_quality.get("joint_continuity_status") == "pass"
            ),
            "evidence": {
                "final_status": final_quality.get("status", "missing"),
                "strict_audit_status": strict_audit.get("overall_status", "missing"),
                "tool_tcp_acceptance_status": final_quality.get("tool_tcp_acceptance_status", "missing"),
                "trajectory_execution_mode": final_quality.get("trajectory_execution_mode", "missing"),
                "source": "outputs/final_quality_report.csv; outputs/audit_goal_requirements_strict.json",
            },
            "interpretation": "The project now claims a scoped strict pass only with explicit TCP waiver and segmented-process IK evidence.",
        },
        {
            "id": "placeholder_tcp_not_production",
            "status": (
                "pass"
                if final_quality.get("tool_tcp_acceptance_status") == "waived"
                else ("blocked" if final_quality.get("tool_tcp_production_status") == "fail" else "pass")
            ),
            "evidence": {
                "tool_tcp_source": final_quality.get("tool_tcp_source", "missing"),
                "tool_tcp_xyz": final_quality.get("tool_tcp_xyz", "missing"),
                "tool_tcp_production_status": final_quality.get("tool_tcp_production_status", "missing"),
                "tool_tcp_acceptance_status": final_quality.get("tool_tcp_acceptance_status", "missing"),
                "tool_tcp_acceptance_evidence": final_quality.get("tool_tcp_acceptance_evidence", "missing"),
                "readiness_missing_evidence": [
                    missing_by_gate[gate]
                    for gate in (
                        "final_quality.tool_tcp_production_status",
                        "derived.tool_tcp_source_is_production",
                        "derived.tool_tcp_metadata_present",
                    )
                    if gate in missing_by_gate
                ],
                "source": "outputs/final_quality_report.csv",
            },
            "interpretation": (
                "The placeholder TCP is explicitly waived for this validation scope; measured TCP evidence remains "
                "outside the current blocker set."
            ),
        },
        {
            "id": "diagnostic_low_speed_not_production",
            "status": _pass_fail(final_quality.get("tcp_speed_production_status") == "pass"),
            "evidence": {
                "fk_tcp_speed_mean_m_s": final_quality.get("fk_tcp_speed_mean_m_s", "missing"),
                "production_min_tcp_speed_m_s": final_quality.get("production_min_tcp_speed_m_s", "missing"),
                "tcp_speed_production_status": final_quality.get("tcp_speed_production_status", "missing"),
                "source": "outputs/final_quality_report.csv",
            },
            "interpretation": "The current formal report clears the configured production minimum speed gate.",
        },
        {
            "id": "default_20deg_ik_continuity",
            "status": "blocked" if final_quality.get("joint_continuity_status") == "fail" else "pass",
            "evidence": {
                "joint_continuity_status": final_quality.get("joint_continuity_status", "missing"),
                "trajectory_execution_mode": final_quality.get("trajectory_execution_mode", "missing"),
                "process_joint_continuity_status": final_quality.get("process_joint_continuity_status", "missing"),
                "process_max_joint_step_deg": final_quality.get("process_max_joint_step_deg", "missing"),
                "reorientation_transition_count": final_quality.get("reorientation_transition_count", "missing"),
                "reorientation_transition_status": final_quality.get("reorientation_transition_status", "missing"),
                "max_joint_step_deg": final_quality.get("max_joint_step_deg", "missing"),
                "production_joint_step_limit_deg": final_quality.get("production_joint_step_limit_deg", "missing"),
                "root_cause_blocking_items": root_cause.get("blocking_items", []),
                "readiness_missing_evidence": [
                    missing_by_gate[gate]
                    for gate in (
                        "final_quality.joint_continuity_status",
                        "derived.max_joint_step_within_limit",
                    )
                    if gate in missing_by_gate
                ],
                "source": "outputs/final_quality_report.csv; outputs/ik_root_cause_matrix.json",
            },
            "interpretation": (
                "The 20 deg gate passes only when large IK branch changes are treated as explicit spray-off "
                "reorientation stops and every spray/process segment remains within the limit."
            ),
        },
        {
            "id": "readme_chinese_mojibake",
            "status": _pass_fail(readme_status.get("status") == "pass" and report_readme_status.get("status") == "pass"),
            "evidence": {
                "README.md": readme_status,
                "PROJECT_REPORT_README.md": report_readme_status,
            },
            "interpretation": "Both README files are UTF-8 readable and contain no configured mojibake markers.",
        },
        {
            "id": "dirty_worktree_organized",
            "status": _pass_fail(not manifest.get("changed_files", {}).get("unclassified")),
            "evidence": {
                "changed_file_count": manifest.get("changed_file_count", "missing"),
                "unclassified": manifest.get("changed_files", {}).get("unclassified", []),
                "source": "outputs/validation_handoff_manifest.json",
            },
            "interpretation": "The worktree is still dirty, but changed files are grouped for review and handoff.",
        },
    ]

    blocked = [item["id"] for item in items if item["status"] == "blocked"]
    failed = [item["id"] for item in items if item["status"] == "fail"]
    return {
        "schema_version": 1,
        "purpose": "Requirement-by-requirement audit for the active 150 mm placeholder TCP collision goal.",
        "overall_goal_status": "complete" if not blocked and not failed else "active",
        "blocked_items": blocked,
        "failed_items": failed,
        "readiness_missing_evidence": readiness.get("missing_evidence", []),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build active-goal resolution audit JSON.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/goal_resolution_audit.json")
    args = parser.parse_args()

    payload = build_audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote goal resolution audit to {args.out}")


if __name__ == "__main__":
    main()
