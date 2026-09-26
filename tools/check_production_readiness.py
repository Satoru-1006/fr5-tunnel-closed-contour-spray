from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MIN_TCP_SPEED_M_S = 0.003


def metric_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _gate(name: str, actual: str, expected: str = "pass") -> dict[str, str]:
    return {
        "gate": name,
        "expected": expected,
        "actual": actual or "missing",
        "status": "pass" if actual == expected else "fail",
    }


def _gate_ok(name: str, ok: bool, actual: str, expected: str = "pass") -> dict[str, str]:
    return {
        "gate": name,
        "expected": expected,
        "actual": actual or "missing",
        "status": "pass" if ok else "fail",
    }


def _derived_gate(name: str, ok: bool, actual: str, expected: str) -> dict[str, str]:
    return {
        "gate": name,
        "expected": expected,
        "actual": actual or "missing",
        "status": "pass" if ok else "fail",
    }


def _float_value(values: dict[str, str], key: str) -> float | None:
    try:
        return float(values.get(key, ""))
    except ValueError:
        return None


def _production_tcp_source(source: str) -> bool:
    source_lower = source.strip().lower()
    return bool(source_lower) and not any(token in source_lower for token in ("assumed", "placeholder", "required"))


def _tcp_scope_waived(quality: dict[str, str]) -> bool:
    return (
        quality.get("tool_tcp_acceptance_status", "").strip().lower() == "waived"
        and quality.get("tool_tcp_source", "").strip().lower() == "assumed_150mm_placeholder"
    )


def _iso_date(value: str) -> bool:
    try:
        dt.date.fromisoformat(value.strip())
        return True
    except ValueError:
        return False


def _missing_evidence(blocking_gates: list[dict[str, str]]) -> list[dict[str, str]]:
    guidance = {
        "final_quality.status": (
            "A regenerated outputs/final_quality_report.csv with status=pass.",
            "Rerun or republish validation after remaining blocking gates are fixed or explicitly waived.",
        ),
        "strict_audit.overall_status": (
            "A regenerated outputs/audit_goal_requirements_strict.json with overall_status=pass.",
            "Rerun tools/audit_goal_requirements.py through scripts/run_moveit_strict_validation.sh.",
        ),
        "collision_report.status": (
            "A MoveIt collision report with status=pass.",
            "Fix collision geometry/TCP/path and rerun collision validation.",
        ),
        "final_quality.collision_status": (
            "Final quality report carrying collision_status=pass from the raw collision report.",
            "Republish final outputs from the latest collision report.",
        ),
        "final_quality.tcp_speed_production_status": (
            "Final quality report with tcp_speed_production_status=pass.",
            "Rerun strict validation at or above the configured production minimum TCP speed.",
        ),
        "final_quality.tool_tcp_production_status": (
            "Final quality report with tool_tcp_production_status=pass or tool_tcp_acceptance_status=waived.",
            "Load a measured TCP YAML, or keep the explicit placeholder TCP waiver in this validation scope.",
        ),
        "final_quality.joint_continuity_status": (
            "Final quality report with joint_continuity_status=pass.",
            "Resolve branch/path/clearance issues or provide segmented spray-off reorientation evidence.",
        ),
        "final_quality.waypoint_post_joint_continuity_match_status": (
            "Final quality report proving pre-Ruckig waypoint and post-Ruckig joint continuity diagnostics match.",
            "Regenerate waypoint and post-Ruckig joint step reports from the same strict run.",
        ),
        "derived.tool_tcp_source_is_production": (
            "A non-placeholder measured/production tool_tcp_source, or an explicit placeholder TCP waiver.",
            "Use a measured TCP calibration source, or keep tool_tcp_acceptance_status=waived for this scoped run.",
        ),
        "derived.tool_tcp_metadata_present": (
            "Non-empty measured TCP metadata, or an explicit placeholder TCP waiver.",
            "Populate measured TCP YAML metadata, or keep tool_tcp_acceptance_status=waived for this scoped run.",
        ),
        "derived.tcp_speed_meets_minimum": (
            "fk_tcp_speed_mean_m_s greater than or equal to production_min_tcp_speed_m_s.",
            "Increase target TCP speed or fix time parameterization while keeping dynamics within limits.",
        ),
        "derived.max_joint_step_within_limit": (
            "process_max_joint_step_deg or max_joint_step_deg less than or equal to the production limit.",
            "Keep process segments under the 20 deg gate and move branch changes into smooth spray-off transitions.",
        ),
        "derived.collision_count_zero": (
            "Raw collision report with collision_count=0.",
            "Fix collision geometry/TCP/path and rerun MoveIt collision validation.",
        ),
        "derived.no_first_collision_index": (
            "Raw collision report with first_collision_index<0.",
            "Fix collision geometry/TCP/path and rerun MoveIt collision validation.",
        ),
    }
    result = []
    for gate in blocking_gates:
        required, action = guidance.get(
            gate["gate"],
            ("Evidence satisfying this gate.", "Inspect the gate and regenerate strict validation artifacts."),
        )
        result.append(
            {
                "gate": gate["gate"],
                "actual": gate["actual"],
                "required_evidence": required,
                "next_action": action,
            }
        )
    return result


def build_readiness_report(
    final_quality_csv: Path,
    strict_audit_json: Path,
    collision_report_csv: Path,
) -> dict[str, Any]:
    quality = metric_map(final_quality_csv)
    strict_audit = load_json(strict_audit_json)
    collision = metric_map(collision_report_csv)
    tcp_waived = _tcp_scope_waived(quality)

    gates = [
        _gate("final_quality.status", quality.get("status", "")),
        _gate("strict_audit.overall_status", str(strict_audit.get("overall_status", ""))),
        _gate("collision_report.status", collision.get("status", "")),
        _gate("final_quality.collision_status", quality.get("collision_status", "")),
        _gate("final_quality.tcp_speed_production_status", quality.get("tcp_speed_production_status", "")),
        _gate_ok(
            "final_quality.tool_tcp_production_status",
            quality.get("tool_tcp_production_status", "") == "pass" or tcp_waived,
            (
                quality.get("tool_tcp_production_status", "")
                if not tcp_waived
                else "waived:" + quality.get("tool_tcp_source", "")
            ),
            "pass or waived",
        ),
        _gate("final_quality.joint_continuity_status", quality.get("joint_continuity_status", "")),
        _gate("final_quality.waypoint_post_joint_continuity_match_status", quality.get("waypoint_post_joint_continuity_match_status", "")),
    ]
    tcp_speed = _float_value(quality, "fk_tcp_speed_mean_m_s")
    speed_min = _float_value(quality, "production_min_tcp_speed_m_s")
    if speed_min is None:
        speed_min = PRODUCTION_MIN_TCP_SPEED_M_S
    max_joint_step = _float_value(quality, "max_joint_step_deg")
    process_max_joint_step = _float_value(quality, "process_max_joint_step_deg")
    joint_limit = _float_value(quality, "production_joint_step_limit_deg")
    if joint_limit is None:
        joint_limit = _float_value(quality, "process_joint_step_limit_deg")
    segmented_mode = (
        quality.get("trajectory_execution_mode", "").strip()
        == "segmented_process_with_smooth_reorientation_stops"
    )
    segmented_joint_ok = (
        segmented_mode
        and quality.get("process_joint_continuity_status") == "pass"
        and quality.get("process_collision_status") == "pass"
        and quality.get("process_dynamics_status") == "pass"
        and quality.get("reorientation_transition_status") == "pass"
        and quality.get("reorientation_transition_collision_status") == "pass"
        and quality.get("reorientation_transition_dynamics_status") == "pass"
        and quality.get("stop_boundary_zero_velocity_acceleration_status") == "pass"
        and quality.get("next_segment_entry_status") == "pass"
        and quality.get("no_gap_or_overlap_status") == "pass"
        and quality.get("spray_off_transition_status") in {"pass", "not_required"}
    )
    effective_joint_step = process_max_joint_step if segmented_joint_ok else max_joint_step
    collision_count = _float_value(collision, "collision_count")
    first_collision_index = _float_value(collision, "first_collision_index")
    derived_gates = [
        _derived_gate(
            "derived.tool_tcp_source_is_production",
            _production_tcp_source(quality.get("tool_tcp_source", "")) or tcp_waived,
            quality.get("tool_tcp_source", ""),
            "non-placeholder measured/production source or waived placeholder",
        ),
        _derived_gate(
            "derived.tool_tcp_metadata_present",
            tcp_waived
            or (
                bool(quality.get("tool_tcp_measured_by", "").strip())
                and bool(quality.get("tool_tcp_calibration_method", "").strip())
                and _iso_date(quality.get("tool_tcp_measured_date", ""))
            ),
            "measured_by="
            + quality.get("tool_tcp_measured_by", "")
            + "; measured_date="
            + quality.get("tool_tcp_measured_date", "")
            + "; calibration_method="
            + quality.get("tool_tcp_calibration_method", ""),
            "non-empty measured TCP metadata or waived placeholder",
        ),
        _derived_gate(
            "derived.tcp_speed_meets_minimum",
            tcp_speed is not None and speed_min is not None and tcp_speed >= speed_min,
            quality.get("fk_tcp_speed_mean_m_s", ""),
            f">={speed_min if speed_min is not None else PRODUCTION_MIN_TCP_SPEED_M_S:g}",
        ),
        _derived_gate(
            "derived.max_joint_step_within_limit",
            effective_joint_step is not None and joint_limit is not None and effective_joint_step <= joint_limit,
            (
                quality.get("process_max_joint_step_deg", "")
                if segmented_joint_ok
                else quality.get("max_joint_step_deg", "")
            ),
            f"<={quality.get('production_joint_step_limit_deg', 'missing')} deg",
        ),
        _derived_gate(
            "derived.segmented_process_acceptance",
            segmented_joint_ok if segmented_mode else quality.get("joint_continuity_status") == "pass",
            "; ".join(
                f"{key}={quality.get(key, '')}"
                for key in (
                    "process_joint_continuity_status",
                    "process_collision_status",
                    "process_dynamics_status",
                    "spray_off_transition_status",
                    "reorientation_transition_collision_status",
                    "reorientation_transition_dynamics_status",
                    "stop_boundary_zero_velocity_acceleration_status",
                    "next_segment_entry_status",
                    "no_gap_or_overlap_status",
                )
            ),
            "all segmented-process gates pass, or continuous mode joint continuity passes",
        ),
        _derived_gate(
            "derived.collision_count_zero",
            collision_count is not None and collision_count == 0.0,
            collision.get("collision_count", ""),
            "0",
        ),
        _derived_gate(
            "derived.no_first_collision_index",
            first_collision_index is not None and first_collision_index < 0.0,
            collision.get("first_collision_index", ""),
            "<0",
        ),
    ]
    gates.extend(derived_gates)
    blocking = [gate for gate in gates if gate["status"] != "pass"]
    return {
        "schema_version": 1,
        "purpose": "Single production-readiness gate for the MoveIt2 strict-validation evidence chain.",
        "overall_status": "pass" if not blocking else "fail",
        "inputs": {
            "final_quality_csv": str(final_quality_csv),
            "strict_audit_json": str(strict_audit_json),
            "collision_report_csv": str(collision_report_csv),
        },
        "key_metrics": {
            "tool_tcp_source": quality.get("tool_tcp_source", ""),
            "tool_tcp_xyz": quality.get("tool_tcp_xyz", ""),
            "tool_tcp_measured_by": quality.get("tool_tcp_measured_by", ""),
            "tool_tcp_measured_date": quality.get("tool_tcp_measured_date", ""),
            "tool_tcp_calibration_method": quality.get("tool_tcp_calibration_method", ""),
            "tool_tcp_acceptance_status": quality.get("tool_tcp_acceptance_status", ""),
            "tool_tcp_acceptance_evidence": quality.get("tool_tcp_acceptance_evidence", ""),
            "fk_tcp_speed_mean_m_s": quality.get("fk_tcp_speed_mean_m_s", ""),
            "production_min_tcp_speed_m_s": quality.get("production_min_tcp_speed_m_s", ""),
            "max_joint_step_deg": quality.get("max_joint_step_deg", ""),
            "trajectory_execution_mode": quality.get("trajectory_execution_mode", ""),
            "process_joint_continuity_status": quality.get("process_joint_continuity_status", ""),
            "process_max_joint_step_deg": quality.get("process_max_joint_step_deg", ""),
            "reorientation_transition_count": quality.get("reorientation_transition_count", ""),
            "reorientation_transition_max_interpolated_step_deg": quality.get(
                "reorientation_transition_max_interpolated_step_deg", ""
            ),
            "reorientation_transition_status": quality.get("reorientation_transition_status", ""),
            "production_joint_step_limit_deg": quality.get("production_joint_step_limit_deg", ""),
            "collision_count": collision.get("collision_count", quality.get("collision_count", "")),
            "first_collision_index": collision.get("first_collision_index", quality.get("first_collision_index", "")),
        },
        "gates": gates,
        "blocking_gates": blocking,
        "missing_evidence": _missing_evidence(blocking),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether strict MoveIt2 evidence is production-ready.")
    parser.add_argument("--final-quality-csv", type=Path, default=ROOT / "outputs/final_quality_report.csv")
    parser.add_argument("--strict-audit-json", type=Path, default=ROOT / "outputs/audit_goal_requirements_strict.json")
    parser.add_argument("--collision-report-csv", type=Path, default=ROOT / "outputs/moveit_collision_report.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/production_readiness_check.json")
    args = parser.parse_args()

    report = build_readiness_report(args.final_quality_csv, args.strict_audit_json, args.collision_report_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote production readiness check to {args.out}")
    print(f"overall_status={report['overall_status']}")
    if report["blocking_gates"]:
        print("blocking_gates=" + ",".join(gate["gate"] for gate in report["blocking_gates"]))
    raise SystemExit(0 if report["overall_status"] == "pass" else 1)


if __name__ == "__main__":
    main()
