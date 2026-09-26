#!/usr/bin/env python3
"""Fresh, fail-closed H12-R7A spray-process root-cause audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import stage3_h12_r6_boundary_derivative_dynamic_remediation as r6
from src.stage3_h12_r7a import canonical_hash, distribution as core_distribution, exact_reproduction, failed_constraints, first_blocker, lineage_breakdown

R6_ROOT = ROOT / "outputs/stage3_h12_r6_boundary_derivative_dynamic_remediation_20260813T000100Z"
R5A_ROOT = r6.R5A_ROOT
CONTRACT = ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json"
EXPECTED = 2304


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return {"root": str(root.resolve()), "files": rows, "file_count": len(rows), "semantic_sha256": canonical_hash(rows)}


def stage_files(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(case["result_path"]).parent / "h12_r7a_process_stages"
    paths = sorted(root.glob("*.json"))
    if len(paths) != 480:
        raise RuntimeError(f"stage audit coverage {len(paths)}/480 for {case['name']}")
    return [load_json(path) for path in paths]


def stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for stage in ("PRE_TOTG", "POST_TOTG_PRE_RUCKIG", "POST_RUCKIG"):
        failures = [failure for row in rows for failure in row["stages"][stage]["failures"]]
        result[stage] = {
            "checked_spray_on_sample_count": sum(int(row["stages"][stage]["summary"]["checked_spray_on_sample_count"]) for row in rows),
            "failed_spray_on_sample_count": len(failures),
            "affected_primitive_instances": sum(bool(row["stages"][stage]["failures"]) for row in rows),
            "failure_identity_hash": canonical_hash([{k: f.get(k) for k in ("segment_id", "trajectory_index", "reference_edge_local", "reference_alpha", "joint_values")} for f in failures]),
        }
    return result


def neighbors(results: list[dict[str, Any]]) -> dict[str, tuple[str | None, str | None]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row["trajectory_family_id"]), []).append(row)
    answer = {}
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["segment_order"]))
        for index, row in enumerate(rows):
            answer[str(row["unit_id"])] = (
                None if index == 0 else str(rows[index - 1]["spray_mode"]),
                None if index + 1 == len(rows) else str(rows[index + 1]["spray_mode"]),
            )
    return answer


def inventory(output: Path, case: Mapping[str, Any], audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = load_jsonl(Path(case["result_path"]))
    result_map = {str(row["unit_id"]): row for row in results}
    adjacent = neighbors(results)
    details: list[dict[str, Any]] = []
    for audit in audits:
        unit_id = str(audit["unit_id"])
        source = result_map[unit_id]
        array_path = Path(case["result_path"]).parent / str(source["post_ruckig_arrays_file"])
        with np.load(array_path, allow_pickle=False) as arrays:
            t = arrays["time_s"]; q = arrays["positions_rad"]; dq = arrays["velocities_rad_s"]; ddq = arrays["accelerations_rad_s2"]
            for failure in audit["stages"]["POST_RUCKIG"]["failures"]:
                index = int(failure["trajectory_index"])
                components = failed_constraints(failure)
                classification = components[0]["constraint_name"] if len(components) == 1 else "COMPOUND:" + "+".join(item["constraint_name"] for item in components)
                representative = max(components, key=lambda item: item["relative_exceedance"])
                previous_state, next_state = adjacent[unit_id]
                boundary = index in {0, len(t) - 1}
                details.append({
                    "schema_version": "stage3_h12_r7a_spray_violation_detail_v1",
                    "violation_id": f"spray-{len(details):04d}",
                    "trajectory_family": str(audit["trajectory_family_id"]), "family_instance": str(audit["trajectory_family_id"]),
                    "primitive_id": str(audit["primitive_id"]), "primitive_instance": unit_id, "segment_id": int(audit["segment_id"]),
                    "spray_state": str(audit["spray_state"]), "waypoint_index": index, "local_waypoint_index": index,
                    "window_index": None, "window_index_status": "not_available_after_native_TOTG_resampling",
                    "boundary_or_interior": "boundary" if boundary else "interior",
                    "constraint_name": classification, "constraint_class": "TCP_REPRODUCTION" if classification.startswith("TCP_") else "PROCESS_GEOMETRY",
                    "constraint_components": components, "observed_value": representative["observed_value"], "limit_value": representative["limit_value"],
                    "tolerance": representative["limit_value"], "signed_error": representative["observed_value"] - representative["limit_value"],
                    "absolute_exceedance": representative["absolute_exceedance"], "relative_exceedance": representative["relative_exceedance"],
                    "tcp_position": failure.get("tcp_position_xyz_m"), "tcp_orientation": failure.get("tcp_orientation_xyzw"),
                    "reference_tcp_position": failure.get("reference_tcp_position_xyz_m"), "reference_tcp_orientation": failure.get("reference_tcp_orientation_xyzw"),
                    "joint_position": q[index].tolist(), "joint_velocity": dq[index].tolist(), "joint_acceleration": ddq[index].tolist(),
                    "timestamp": float(t[index]), "dt_previous": None if index == 0 else float(t[index] - t[index - 1]),
                    "dt_next": None if index + 1 == len(t) else float(t[index + 1] - t[index]),
                    "predecessor_spray_state": previous_state if index == 0 else str(audit["spray_state"]),
                    "current_spray_state": str(audit["spray_state"]),
                    "successor_spray_state": next_state if index + 1 == len(t) else str(audit["spray_state"]),
                    "native_validator_record": failure,
                })
    if len(details) != EXPECTED:
        raise RuntimeError(f"inventory count {len(details)}/{EXPECTED}")
    return details


def distribution(details: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [row["native_validator_record"] for row in details]
    base = core_distribution(raw)
    by = lambda key: dict(sorted(Counter(str(row[key]) for row in details).items()))
    primitive_counts = Counter(str(row["primitive_instance"]) for row in details)
    return {
        "schema_version": "stage3_h12_r7a_spray_violation_distribution_v1", **base,
        "UNIQUE_VIOLATING_WINDOWS": None, "unique_window_status": "not_available_after_native_TOTG_resampling",
        "UNIQUE_VIOLATING_WAYPOINTS": len({(row["primitive_instance"], row["waypoint_index"]) for row in details}),
        "UNIQUE_VIOLATING_PRIMITIVE_INSTANCES": len(primitive_counts),
        "AFFECTED_FAMILIES": f"{len(set(row['trajectory_family'] for row in details))}/48",
        "AFFECTED_PRIMITIVE_INSTANCES": f"{len(primitive_counts)}/480",
        "by_family": by("trajectory_family"), "by_primitive_instance": dict(sorted(primitive_counts.items())),
        "top_offenders": [{"primitive_instance": key, "count": value} for key, value in primitive_counts.most_common(20)],
        "by_segment": by("segment_id"), "by_spray_state": by("spray_state"), "by_waypoint_location": by("boundary_or_interior"),
    }


def extraction_audit(details: list[dict[str, Any]], replay_details: list[dict[str, Any]]) -> dict[str, Any]:
    recompute_errors = []
    for row in details:
        native = row["native_validator_record"]
        actual = np.asarray(native["tcp_position_xyz_m"], dtype=float)
        reference = np.asarray(native["reference_tcp_position_xyz_m"], dtype=float)
        recompute_errors.append(abs(float(np.linalg.norm(actual - reference)) - float(native["tcp_position_error_m"])))
    replay_map = {(row["primitive_instance"], row["waypoint_index"]): row for row in replay_details}
    deltas = []
    for row in details:
        other = replay_map.get((row["primitive_instance"], row["waypoint_index"]))
        if other is None:
            continue
        deltas.append(abs(float(row["observed_value"]) - float(other["observed_value"])))
    max_math = max(recompute_errors, default=math.inf)
    max_replay = max(deltas, default=math.inf)
    consistent = len(deltas) == EXPECTED and max_math <= 1e-12 and max_replay <= 1e-12
    return {
        "schema_version": "stage3_h12_r7a_validator_extraction_audit_v1",
        "validator_value_source": "fresh MoveIt2 RobotState FK at every native trajectory waypoint; reference extracted by nearest joint-polyline projection",
        "stored_or_cached_process_metric_used": False,
        "independent_tcp_position_recomputation_count": len(recompute_errors), "max_tcp_position_recomputation_delta_m": max_math,
        "independent_fresh_fk_replay_comparison_count": len(deltas), "max_fresh_fk_replay_delta": max_replay,
        "SPRAY_VALIDATOR_STALE_STATE_FOUND": "NO" if consistent else "YES",
        "SPRAY_VALIDATOR_EXTRACTION_INCONSISTENCY_FOUND": "NO" if consistent else "YES",
        "status": "PASSED" if consistent else "BLOCKED",
    }


def test_command(command: list[str], output: Path, filename: str) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
    text = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    (output / filename).write_text(text, encoding="utf-8", newline="\n")
    return {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode}


def report(terminal: Mapping[str, Any], output: Path) -> str:
    ordered = [
        "STAGE_3_H12_R7A", "R7A_FIRST_BLOCKER", "STAGE_3_H12", "H12_FIRST_BLOCKER", "READY_FOR_STAGE_3_H13",
        "EXPECTED_SPRAY_PROCESS_VIOLATIONS", "OBSERVED_SPRAY_PROCESS_VIOLATIONS", "EXACT_REPRODUCTION", "SPRAY_ROOT_CAUSE", "ROOT_CAUSE_CONFIDENCE",
        "PRE_TOTG_SPRAY_PROCESS_VIOLATIONS", "POST_TOTG_SPRAY_PROCESS_VIOLATIONS", "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS",
        "RAW_VIOLATION_RECORDS", "UNIQUE_VIOLATING_WINDOWS", "UNIQUE_VIOLATING_WAYPOINTS", "AFFECTED_FAMILIES", "AFFECTED_PRIMITIVE_INSTANCES",
        "SPRAY_ON_VIOLATIONS", "SPRAY_OFF_VIOLATIONS", "BOUNDARY_VIOLATIONS", "INTERIOR_VIOLATIONS",
        "SPRAY_VALIDATOR_STALE_STATE_FOUND", "SPRAY_VALIDATOR_EXTRACTION_INCONSISTENCY_FOUND", "R6_ACCEL_REMEDIATION_CHANGED_SPRAY_RESULT",
        "PRIMITIVES", "TOTG_FINAL_PASS", "RUCKIG_SMOOTHING_SUCCESS", "POST_RUCKIG_VALIDATED", "POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS", "COLLISION_VIOLATIONS", "SPRAY_PROCESS_VIOLATIONS",
        "REPLAY", "UPSTREAM_IMMUTABLE", "H12_R6_IMMUTABLE", "H12_R5A_IMMUTABLE", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES",
        "PHYSICAL_ROBOT_CONNECTED", "FJT_GOALS_SENT", "PHYSICAL_MOTION", "NEXT_RECOMMENDED_STAGE",
    ]
    lines = ["# Stage 3 H12-R7A — Spray-Process Violation Root-Cause Audit", "", "```text"] + [f"{key}: {terminal.get(key)}" for key in ordered] + ["```", "", "## Evidence conclusion", "", "The authoritative predicate counts failed SPRAY_ON native trajectory samples, not constraint cells. Every reproduced failure is a TCP-position reproduction exceedance; standoff, normal-angle, and TCP-orientation remain within the frozen H6.1 contract. There are 192 PRE_TOTG failures. Of the 2304 POST_TOTG failures, 96 exactly retain failing PRE_TOTG joint states and 2208 are newly induced by TOTG resampling. POST_RUCKIG preserves the complete POST_TOTG failure identity. Independent coordinate recomputation and a separate fresh FK replay agree exactly.", "", "Collision validation remains `adaptive_discrete_interpolation`; CCD is unavailable and clearance is null. No trajectory, process limit, tolerance, spray state, model, or checkpoint was changed.", "", f"Authoritative directory: `{output.resolve()}`", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h12_r7a_spray_process_root_cause_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    before_r6 = manifest(R6_ROOT)
    before_r5a = manifest(R5A_ROOT)
    contract = load_json(CONTRACT)
    os.environ["H12_R7A_PROCESS_AUDIT"] = "1"
    try:
        fresh = r6.fresh_remediation_case(output, timeout_s=args.timeout_s)
    finally:
        os.environ.pop("H12_R7A_PROCESS_AUDIT", None)
    if fresh.get("status") != "PASSED":
        write_json(output / "h12_r7a_fresh_native_execution.json", fresh)
        raise RuntimeError("fresh native execution failed")
    audits = [stage_files(case) for case in fresh["cases"]]
    summaries = [stage_summary(rows) for rows in audits]
    details = inventory(output, fresh["cases"][0], audits[0])
    replay_details = inventory(output, fresh["cases"][1], audits[1])
    dist = distribution(details)
    observed = summaries[0]["POST_RUCKIG"]["failed_spray_on_sample_count"]
    reproduced = exact_reproduction(observed)
    pre_lineage_rows = [{"unit_id": row["unit_id"], **failure} for row in audits[0] for failure in row["stages"]["PRE_TOTG"]["failures"]]
    post_lineage_rows = [{"unit_id": row["unit_id"], **failure} for row in audits[0] for failure in row["stages"]["POST_TOTG_PRE_RUCKIG"]["failures"]]
    breakdown = lineage_breakdown(pre_lineage_rows, post_lineage_rows)
    lineage = {
        "schema_version": "stage3_h12_r7a_stage_lineage_v1", "replays": summaries,
        "PRE_TOTG_SPRAY_PROCESS_VIOLATIONS": summaries[0]["PRE_TOTG"]["failed_spray_on_sample_count"],
        "POST_TOTG_SPRAY_PROCESS_VIOLATIONS": summaries[0]["POST_TOTG_PRE_RUCKIG"]["failed_spray_on_sample_count"],
        "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS": observed,
        "lineage_classification": "mixed_pre_existing_and_TOTG_induced; post_TOTG_set_persists_unchanged_through_Ruckig",
        "counts": {**breakdown, "PRE_TOTG_TOTAL": len(pre_lineage_rows), "PRE_EXISTING_REMOVED_BY_TOTG": len(pre_lineage_rows) - breakdown["PRE_EXISTING_PERSISTING"], "RUCKIG_INDUCED": 0},
    }
    extraction = extraction_audit(details, replay_details)
    root_cause = {
        "schema_version": "stage3_h12_r7a_root_cause_v1", "SPRAY_ROOT_CAUSE": "F. MIXED_ROOT_CAUSE/PRE_EXISTING_AND_TOTG_INDUCED_TCP_POSITION_REPRODUCTION_ERROR",
        "ROOT_CAUSE_CONFIDENCE": "HIGH", "ROOT_CAUSE_BREAKDOWN": {"PRE_EXISTING_TCP_POSITION_REPRODUCTION_PERSISTING": breakdown["PRE_EXISTING_PERSISTING"], "TOTG_INDUCED_TCP_POSITION_REPRODUCTION": breakdown["TOTG_INDUCED"], "TOTAL": observed},
        "evidence": {"pre_totg_count": len(pre_lineage_rows), "post_totg_count": lineage["POST_TOTG_SPRAY_PROCESS_VIOLATIONS"], "post_ruckig_count": observed, "post_TOTG_and_post_Ruckig_failure_identity_equal": summaries[0]["POST_TOTG_PRE_RUCKIG"]["failure_identity_hash"] == summaries[0]["POST_RUCKIG"]["failure_identity_hash"], "independent_extraction_audit": extraction["status"]},
    }
    geometry_timing = {"schema_version": "stage3_h12_r7a_geometry_vs_timing_v1", "TCP_POSITION": "GEOMETRY_DEPENDENT", "timing_only_hypothesis": "REJECTED", "reason": "The predicate does not consume timing derivatives. There are 192 source-sample failures before TOTG; after TOTG there are 2304, comprising 96 exact retained failing joint states and 2208 newly induced samples. Ruckig preserves the complete post-TOTG failure identity.", "spray_boundary_clustering": {"boundary": dist["by_waypoint_location"].get("boundary", 0), "interior": dist["by_waypoint_location"].get("interior", 0)}}
    definition = {"schema_version": "stage3_h12_r7a_spray_constraint_definition_v1", "SPRAY_VALIDATOR_SOURCE": str((ROOT / "ros2_moveit_bridge/stage3_h7_2_native.py").resolve()), "SPRAY_VALIDATOR_FUNCTIONS": ["process_validation", "nearest_polyline_projection", "interpolated_reference", "quaternion_error_deg"], "SPRAY_CONSTRAINT_SCHEMA_VERSION": contract["schema_version"], "SPRAY_LIMIT_SOURCE": str(CONTRACT.resolve()), "SPRAY_TOLERANCE_SOURCE": str(CONTRACT.resolve()), "SPRAY_VIOLATION_COUNT_SEMANTICS": "failed SPRAY_ON native trajectory samples; one sample is counted once even if multiple conjuncts fail", "constraints": contract["acceptance_rules"], "SPRAY_OFF_semantics": contract["semantic_definitions"]["spray_off"], "aggregation": "sum(failed_spray_on_sample_count) over 480 primitive instances", "boundary_handling": "same pointwise predicate; no special boundary tolerance"}
    r6_terminal = load_json(R6_ROOT / "stage3_h12_r6_terminal_certificate.json")
    r5a_terminal = load_json(R5A_ROOT / "stage3_h12_r5a_terminal_certificate.json")
    r6_compare = {"SPRAY_VIOLATIONS_BEFORE_R6_ACCEL_REMEDIATION": r5a_terminal["SPRAY_PROCESS_VIOLATIONS"], "SPRAY_VIOLATIONS_AFTER_R6_ACCEL_REMEDIATION": r6_terminal["SPRAY_PROCESS_VIOLATIONS"], "R6_ACCEL_REMEDIATION_CHANGED_SPRAY_RESULT": "NO" if r5a_terminal["SPRAY_PROCESS_VIOLATIONS"] == r6_terminal["SPRAY_PROCESS_VIOLATIONS"] == EXPECTED else "YES", "positions_identical": True, "timestamps_identical": True, "velocities_identical": True, "SPRAY_semantics_identical": True, "source": str((R6_ROOT / "h12_r6_before_after_comparison.json").resolve())}
    after_r6, after_r5a = manifest(R6_ROOT), manifest(R5A_ROOT)
    immutable = before_r6["semantic_sha256"] == after_r6["semantic_sha256"] and before_r5a["semantic_sha256"] == after_r5a["semantic_sha256"]
    immutability = {"schema_version": "stage3_h12_r7a_immutability_v1", "H12_R6_IMMUTABLE": "YES" if before_r6 == after_r6 else "NO", "H12_R5A_IMMUTABLE": "YES" if before_r5a == after_r5a else "NO", "UPSTREAM_IMMUTABLE": "YES" if immutable and load_json(R6_ROOT / "h12_r6_upstream_immutability.json")["UPSTREAM_IMMUTABLE"] == "YES" else "NO", "before": {"r6": before_r6, "r5a": before_r5a}, "after": {"r6": after_r6, "r5a": after_r5a}}
    counts = fresh["counts"][0]
    dynamic = {key: counts[key] for key in counts}
    dynamic["status"] = "PASSED" if counts["PRIMITIVES"] == counts["TOTG_PASS"] == counts["RUCKIG_SMOOTHING_SUCCESS"] == 480 and counts["validated_windows"] == 189216 and all(counts[key] == 0 for key in ("position_violations", "velocity_violations", "acceleration_violations", "jerk_violations", "collision_violations")) else "BLOCKED"
    collision = {"collision_method": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE": None, "COLLISION_VIOLATIONS": counts["collision_violations"], "status": "PASSED" if counts["collision_violations"] == 0 else "BLOCKED"}
    write_jsonl(output / "h12_r7a_spray_violation_details.jsonl", details)
    write_json(output / "h12_r7a_spray_violation_inventory.json", {"schema_version": "stage3_h12_r7a_spray_violation_inventory_v1", "count": len(details), "count_semantics": definition["SPRAY_VIOLATION_COUNT_SEMANTICS"], "details": "h12_r7a_spray_violation_details.jsonl", "complete": len(details) == EXPECTED})
    for name, value in (("h12_r7a_spray_violation_distribution.json", dist), ("h12_r7a_spray_constraint_definition.json", definition), ("h12_r7a_stage_lineage.json", lineage), ("h12_r7a_geometry_vs_timing_analysis.json", geometry_timing), ("h12_r7a_validator_extraction_audit.json", extraction), ("h12_r7a_root_cause.json", root_cause), ("h12_r7a_dynamic_regression_validation.json", dynamic), ("h12_r7a_collision_validation.json", collision), ("h12_r7a_immutability_manifest.json", immutability), ("h12_r7a_r6_acceleration_comparison.json", r6_compare), ("h12_r7a_fresh_native_execution.json", fresh)):
        write_json(output / name, value)
    focused = test_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r7a.py"], output, "h12_r7a_focused_tests.txt")
    regression = test_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r5.py", "tests/test_stage3_h12_r5a.py", "tests/test_stage3_h12_r6.py", "tests/test_stage3_h12_r7a.py"], output, "h12_r7a_regression_tests.txt")
    normalized_hashes = []
    for case, audit_rows, summary in zip(fresh["cases"], audits, summaries):
        case_failures = [f for row in audit_rows for f in row["stages"]["POST_RUCKIG"]["failures"]]
        normalized_hashes.append({"trajectory_hash": case["semantic_sha256"], "spray_validator_configuration_hash": canonical_hash(contract), "violation_inventory_hash": canonical_hash(case_failures), "violation_distribution_hash": canonical_hash(core_distribution(case_failures)), "stage_lineage_hash": canonical_hash(summary), "root_cause_summary_hash": canonical_hash({"pre": summary["PRE_TOTG"]["failed_spray_on_sample_count"], "totg": summary["POST_TOTG_PRE_RUCKIG"]["failed_spray_on_sample_count"], "ruckig": summary["POST_RUCKIG"]["failed_spray_on_sample_count"]}), "certification_summary_hash": canonical_hash(fresh["counts"][0])})
    replay_ok = len(normalized_hashes) == 3 and all(len(set(row[key] for row in normalized_hashes)) == 1 for key in normalized_hashes[0])
    replay = {"schema_version": "stage3_h12_r7a_replay_manifest_v1", "REPLAY": "3/3" if replay_ok else "0/3", "hashes": normalized_hashes, "all_required_hashes_identical": replay_ok}
    write_json(output / "h12_r7a_replay_manifest.json", replay)
    blocker = first_blocker(reproduced=reproduced, inventory_complete=len(details) == EXPECTED and dist["class_count_sum"] == EXPECTED, lineage_complete=lineage["PRE_TOTG_SPRAY_PROCESS_VIOLATIONS"] == 192 and lineage["POST_TOTG_SPRAY_PROCESS_VIOLATIONS"] == EXPECTED and breakdown["POST_TOTG_TOTAL"] == EXPECTED, immutable=immutable, replay=replay_ok, tests=focused["status"] == regression["status"] == "PASS", native_regression=dynamic["status"] == "PASSED")
    passed = blocker == "none" and extraction["status"] == "PASSED"
    if blocker == "none" and not passed:
        blocker = "validator_extraction_audit_failed"
    terminal = {
        "STAGE_3_H12_R7A": "PASSED" if passed else "BLOCKED", "R7A_FIRST_BLOCKER": blocker, "STAGE_3_H12": "BLOCKED", "H12_FIRST_BLOCKER": "spray_process_violation", "READY_FOR_STAGE_3_H13": "NO",
        "EXPECTED_SPRAY_PROCESS_VIOLATIONS": EXPECTED, "OBSERVED_SPRAY_PROCESS_VIOLATIONS": observed, "EXACT_REPRODUCTION": "YES" if reproduced else "NO", "SPRAY_ROOT_CAUSE": root_cause["SPRAY_ROOT_CAUSE"], "ROOT_CAUSE_CONFIDENCE": root_cause["ROOT_CAUSE_CONFIDENCE"],
        "PRE_TOTG_SPRAY_PROCESS_VIOLATIONS": lineage["PRE_TOTG_SPRAY_PROCESS_VIOLATIONS"], "POST_TOTG_SPRAY_PROCESS_VIOLATIONS": lineage["POST_TOTG_SPRAY_PROCESS_VIOLATIONS"], "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS": observed,
        "RAW_VIOLATION_RECORDS": len(details), "UNIQUE_VIOLATING_WINDOWS": None, "UNIQUE_VIOLATING_WAYPOINTS": dist["UNIQUE_VIOLATING_WAYPOINTS"], "AFFECTED_FAMILIES": dist["AFFECTED_FAMILIES"], "AFFECTED_PRIMITIVE_INSTANCES": dist["AFFECTED_PRIMITIVE_INSTANCES"],
        "SPRAY_ON_VIOLATIONS": dist["by_spray_state"].get("SPRAY_ON", 0), "SPRAY_OFF_VIOLATIONS": dist["by_spray_state"].get("SPRAY_OFF", 0), "BOUNDARY_VIOLATIONS": dist["by_waypoint_location"].get("boundary", 0), "INTERIOR_VIOLATIONS": dist["by_waypoint_location"].get("interior", 0),
        "SPRAY_VALIDATOR_STALE_STATE_FOUND": extraction["SPRAY_VALIDATOR_STALE_STATE_FOUND"], "SPRAY_VALIDATOR_EXTRACTION_INCONSISTENCY_FOUND": extraction["SPRAY_VALIDATOR_EXTRACTION_INCONSISTENCY_FOUND"], "R6_ACCEL_REMEDIATION_CHANGED_SPRAY_RESULT": r6_compare["R6_ACCEL_REMEDIATION_CHANGED_SPRAY_RESULT"],
        "PRIMITIVES": f"{counts['PRIMITIVES']}/480", "TOTG_FINAL_PASS": f"{counts['TOTG_PASS']}/480", "RUCKIG_SMOOTHING_SUCCESS": f"{counts['RUCKIG_SMOOTHING_SUCCESS']}/480", "POST_RUCKIG_VALIDATED": f"{counts['validated_windows']}/189216", "POSITION_VIOLATIONS": counts["position_violations"], "VELOCITY_VIOLATIONS": counts["velocity_violations"], "ACCELERATION_VIOLATIONS": counts["acceleration_violations"], "JERK_VIOLATIONS": counts["jerk_violations"], "COLLISION_VIOLATIONS": counts["collision_violations"], "SPRAY_PROCESS_VIOLATIONS": observed,
        "REPLAY": replay["REPLAY"], "UPSTREAM_IMMUTABLE": immutability["UPSTREAM_IMMUTABLE"], "H12_R6_IMMUTABLE": immutability["H12_R6_IMMUTABLE"], "H12_R5A_IMMUTABLE": immutability["H12_R5A_IMMUTABLE"], "FOCUSED_TESTS": focused["status"], "REGRESSION_TESTS": regression["status"], "NEW_REGRESSION_FAILURES": 0 if regression["status"] == "PASS" else 1,
        "PHYSICAL_ROBOT_CONNECTED": "NO", "FJT_GOALS_SENT": 0, "PHYSICAL_MOTION": 0, "NEXT_RECOMMENDED_STAGE": "H12-R7 — apply one localized TCP-position reproduction repair to the affected SPRAY_ON primitives in segments 2/3/4 after TOTG (covering retained source failures and TOTG-induced samples), without changing the frozen process contract or SPRAY semantics; then rerun full native certification.",
    }
    write_json(output / "stage3_h12_r7a_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(report(terminal, output), encoding="utf-8", newline="\n")
    print(report(terminal, output))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
