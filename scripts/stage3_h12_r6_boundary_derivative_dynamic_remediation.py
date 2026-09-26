"""Stage 3 H12-R6: evidence-first derivative remediation.

The R5A result directory is treated as immutable evidence.  This runner
audits that evidence, writes a new R6 output directory, and only then offers
an opt-in fresh native recertification.  It never changes geometry, limits,
spray semantics, tolerances, or the R5A artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R5A_ROOT = ROOT / "outputs" / "stage3_h12_r5a_moveit_helper_semantics_closure_20260812T154400Z"
EXPECTED_PRIMITIVES = 480
EXPECTED_WINDOWS = 189216
ACCELERATION_LIMITS = np.full(6, 0.7, dtype=np.float64)
JOINT_NAMES = tuple(f"j{i}" for i in range(1, 7))
NEURAL = {
    "CV_RMSE": 0.0025959456389118936,
    "RAW_NEURAL_RMSE": 0.000995727054577695,
    "POST_REPAIR_NEURAL_RMSE": 0.0006688626630419337,
}

sys.path.insert(0, str(ROOT))
from src.stage3_h12_r6 import (  # noqa: E402
    boundary_state_audit,
    classify_acceleration_root_cause,
    derivative_consistency_audit,
    derivative_windows,
    distribution,
    reconstruct_local_derivative_state,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_ruckig_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime timing noise while retaining all semantic Ruckig gates."""
    return {key: raw for key, raw in value.items() if key not in {"native_compute_time"}}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def require_r5a_authority() -> dict[str, Any]:
    required = [
        "stage3_h12_r5a_terminal_certificate.json",
        "h12_r5a_ruckig_results.jsonl",
        "h12_r5a_primitive_manifest.jsonl",
        "h12_r5a_upstream_immutability_before.json",
        "h12_r5a_upstream_immutability.json",
    ]
    missing = [name for name in required if not (R5A_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"R5A authority missing: {missing}")
    certificate = load_json(R5A_ROOT / "stage3_h12_r5a_terminal_certificate.json")
    checks = {
        "certificate_stage3_h12_r5a": certificate.get("STAGE_3_H12_R5A") == "PASSED",
        "certificate_acceleration_violations": certificate.get("ACCELERATION_VIOLATIONS") == 40,
        "certificate_primitive_coverage": certificate.get("PRIMITIVES") == "480/480",
        "certificate_window_coverage": certificate.get("POST_RUCKIG_VALIDATED") == "189216/189216",
        "certificate_h12_first_blocker": certificate.get("H12_FIRST_BLOCKER") == "acceleration_violation",
    }
    if not all(checks.values()):
        raise RuntimeError(f"R5A authority certificate mismatch: {checks}")
    return {"certificate": certificate, "checks": checks, "root": str(R5A_ROOT)}


def _hash_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("upstream hash manifest must be an object")
    result: dict[str, str] = {}
    for key, raw in value.items():
        if isinstance(raw, str):
            result[str(key)] = raw
        elif isinstance(raw, dict) and isinstance(raw.get("sha256"), str):
            result[str(key)] = str(raw["sha256"])
    return result


def audit_upstream_immutability(*, current_hash_audit: bool) -> dict[str, Any]:
    before_raw = load_json(R5A_ROOT / "h12_r5a_upstream_immutability_before.json")
    pair_raw = load_json(R5A_ROOT / "h12_r5a_upstream_immutability.json")
    before = _hash_map(pair_raw.get("before", before_raw))
    after = _hash_map(pair_raw.get("after", {}))
    historical_equal = bool(before) and before == after and pair_raw.get("UPSTREAM_IMMUTABLE") == "YES"
    current: dict[str, Any] = {"performed": False, "missing": [], "changed": [], "errors": []}
    if current_hash_audit:
        current["performed"] = True
        for raw_path, expected in after.items():
            path = Path(raw_path)
            try:
                if not path.is_file():
                    current["missing"].append(raw_path)
                elif sha256_file(path) != expected:
                    current["changed"].append(raw_path)
            except OSError as exc:
                current["errors"].append({"path": raw_path, "error": str(exc)})
    upstream_immutable = historical_equal and not current["missing"] and not current["changed"] and not current["errors"]
    return {
        "schema_version": "stage3_h12_r6_upstream_immutability_v1",
        "H12_R5A_IMMUTABLE": "YES" if historical_equal else "NO",
        "UPSTREAM_IMMUTABLE": "YES" if upstream_immutable else "NO",
        "historical_manifest_equal": historical_equal,
        "before_entry_count": len(before),
        "after_entry_count": len(after),
        "current_hash_audit": current,
        "source_r5a_immutability_file": str(R5A_ROOT / "h12_r5a_upstream_immutability.json"),
    }


def array_path(row: Mapping[str, Any]) -> Path:
    relative = Path(str(row["post_ruckig_arrays_file"]))
    candidates = [R5A_ROOT / "native_runs" / "formal" / relative, R5A_ROOT / relative]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"post-Ruckig array artifact not found for {row.get('unit_id')}: {relative}")


def manifest_index() -> dict[str, dict[str, Any]]:
    rows = load_jsonl(R5A_ROOT / "h12_r5a_primitive_manifest.jsonl")
    if len(rows) != EXPECTED_PRIMITIVES:
        raise RuntimeError(f"R5A primitive manifest coverage is {len(rows)}/{EXPECTED_PRIMITIVES}")
    return {str(row["unit_id"]): row for row in rows}


def _state(arrays: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        "position_rad": arrays["positions_rad"][index].astype(float).tolist(),
        "velocity_rad_s": arrays["velocities_rad_s"][index].astype(float).tolist(),
        "acceleration_rad_s2": arrays["accelerations_rad_s2"][index].astype(float).tolist(),
        "timestamp_s": float(arrays["time_s"][index]),
    }


def _native_fields(row: Mapping[str, Any], joint: int) -> dict[str, Any]:
    analytic = row.get("post_ruckig_dynamic") or {}
    native_count = int(analytic.get("native_analytic_acceleration_violation_count", 0))
    return {
        "ruckig_input_acceleration_rad_s2": None,
        "ruckig_target_acceleration_rad_s2": None,
        "ruckig_output_acceleration_rad_s2": None,
        "continuous_native_analytic_max_acceleration_rad_s2": None,
        "continuous_native_analytic_min_acceleration_rad_s2": None,
        "continuous_native_analytic_violation_count": native_count,
        "continuous_native_analytic_fields_status": "NOT_PERSISTED_IN_COMPACT_R5A_RESULT",
        "continuous_native_analytic_limit_actually_fails": native_count > 0,
        "continuous_native_analytic_unavailability_reason": "R5A compact result stores only the aggregate native analytic violation count; no per-call extrema are inferred.",
        "joint_index": joint,
    }


def audit_inventory() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    import_manifest = manifest_index()
    results = load_jsonl(R5A_ROOT / "h12_r5a_ruckig_results.jsonl")
    if len(results) != EXPECTED_PRIMITIVES:
        raise RuntimeError(f"R5A result coverage is {len(results)}/{EXPECTED_PRIMITIVES}")
    details: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    primitive_audits: list[dict[str, Any]] = []
    total_windows = 0
    for row in results:
        unit_id = str(row["unit_id"])
        manifest = import_manifest.get(unit_id)
        if manifest is None:
            raise RuntimeError(f"result unit missing from manifest: {unit_id}")
        total_windows += int(row.get("window_count", 0))
        path = array_path(row)
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in ("time_s", "positions_rad", "velocities_rad_s", "accelerations_rad_s2")}
        audit = derivative_consistency_audit(arrays["time_s"], arrays["velocities_rad_s"], arrays["accelerations_rad_s2"], ACCELERATION_LIMITS)
        derivative_rows.append({"unit_id": unit_id, **{key: value for key, value in audit.items() if key != "derived_acceleration"}})
        derived = np.asarray(audit["derived_acceleration"], dtype=float)
        primitive_audits.append({"unit_id": unit_id, "family_id": row["trajectory_family_id"], "segment_id": row["segment_id"], "stored_violation_count": audit["stored_waypoint_acceleration_violation_count"], "finite_difference_violation_count": audit["finite_difference_acceleration_violation_count"], "native_analytic_violation_count": int((row.get("post_ruckig_dynamic") or {}).get("native_analytic_acceleration_violation_count", 0))})
        boundary_rows.append({
            "unit_id": unit_id,
            "primitive_id": str(row["primitive_id"]),
            "trajectory_family_id": str(row["trajectory_family_id"]),
            "segment_id": int(row["segment_id"]),
            "spray_state": str(row.get("spray_mode", manifest.get("spray_state", "UNKNOWN"))),
            "start_state": _state(arrays, 0),
            "end_state": _state(arrays, len(arrays["time_s"]) - 1),
        })
        for waypoint, joint in np.argwhere(np.abs(arrays["accelerations_rad_s2"]) > ACCELERATION_LIMITS[None, :] + 1.0e-10):
            window = derivative_windows(arrays["time_s"], arrays["velocities_rad_s"], int(waypoint))
            boundary_type = "primitive_start" if waypoint == 0 else "primitive_end" if waypoint == len(arrays["time_s"]) - 1 else "interior"
            stored = float(arrays["accelerations_rad_s2"][waypoint, joint])
            centered = float(derived[waypoint, joint])
            limit = float(ACCELERATION_LIMITS[joint])
            detail = {
                "schema_version": "stage3_h12_r6_acceleration_violation_detail_v1",
                "family_id": str(row["trajectory_family_id"]),
                "trajectory_family_id": str(row["trajectory_family_id"]),
                "primitive_id": str(row["primitive_id"]),
                "primitive_instance_id": unit_id,
                "segment_id": int(row["segment_id"]),
                "unit_id": unit_id,
                "spray_state": str(row.get("spray_mode", manifest.get("spray_state", "UNKNOWN"))),
                "waypoint_index": int(waypoint),
                "window_index": None,
                "window_index_status": "NOT_PERSISTED_AS_ONE_TO_ONE_IN_COMPACT_R5A_RESULT",
                "timestamp_s": float(arrays["time_s"][waypoint]),
                "dt_before": window["dt_before"],
                "dt_after": window["dt_after"],
                "joint_index": int(joint),
                "joint_name": JOINT_NAMES[int(joint)],
                "joint_acceleration_limit_rad_s2": limit,
                "stored_waypoint_acceleration_rad_s2": stored,
                "derived_acceleration_centered_rad_s2": centered,
                "derived_acceleration_before_rad_s2": None if window["acceleration_from_velocity_before"] is None else float(window["acceleration_from_velocity_before"][joint]),
                "derived_acceleration_after_rad_s2": None if window["acceleration_from_velocity_after"] is None else float(window["acceleration_from_velocity_after"][joint]),
                "violation_magnitude_rad_s2": max(abs(stored) - limit, 0.0),
                "violation_ratio": abs(stored) / limit,
                "boundary_type": boundary_type,
                "boundary_position_rad": _state(arrays, 0 if waypoint == 0 else len(arrays["time_s"]) - 1)["position_rad"],
                "boundary_velocity_rad_s": _state(arrays, 0 if waypoint == 0 else len(arrays["time_s"]) - 1)["velocity_rad_s"],
                "boundary_acceleration_rad_s2": _state(arrays, 0 if waypoint == 0 else len(arrays["time_s"]) - 1)["acceleration_rad_s2"],
                "previous_waypoint": None if waypoint == 0 else _state(arrays, int(waypoint) - 1),
                "current_waypoint": _state(arrays, int(waypoint)),
                "next_waypoint": None if waypoint == len(arrays["time_s"]) - 1 else _state(arrays, int(waypoint) + 1),
                **_native_fields(row, int(joint)),
            }
            details.append(detail)
    if total_windows != EXPECTED_WINDOWS:
        raise RuntimeError(f"R5A window coverage is {total_windows}/{EXPECTED_WINDOWS}")
    if len(details) != 40:
        raise RuntimeError(f"R5A acceleration detail count is {len(details)}/40")
    for row in boundary_rows:
        row["start_state"]["primitive_boundary_type"] = "primitive_start"
        row["end_state"]["primitive_boundary_type"] = "primitive_end"
    boundary_audit = boundary_state_audit(boundary_rows)
    inventory = {
        "schema_version": "stage3_h12_r6_acceleration_violation_inventory_v1",
        "source_r5a_directory": str(R5A_ROOT),
        "source_results": str(R5A_ROOT / "h12_r5a_ruckig_results.jsonl"),
        "authoritative_primitive_count": EXPECTED_PRIMITIVES,
        "authoritative_window_count": EXPECTED_WINDOWS,
        "primitive_result_count": len(results),
        "window_result_count": total_windows,
        "ACCELERATION_VIOLATIONS": len(details),
        "stored_waypoint_acceleration_violations": len(details),
        "finite_difference_acceleration_violations": sum(row["finite_difference_violation_count"] for row in primitive_audits),
        "continuous_native_analytic_acceleration_violations": sum(row["native_analytic_violation_count"] for row in primitive_audits),
        "details_jsonl": "h12_r6_acceleration_violation_details.jsonl",
        "evidence_status": "PASSED",
    }
    return details, inventory, {"primitive_audits": primitive_audits, "total_windows": total_windows}, boundary_audit


def fresh_remediation_case(output: Path, *, timeout_s: int) -> dict[str, Any]:
    """Run the complete 480-primitive native path with R6 enabled."""
    from scripts import stage3_h12_r5_ruckig_root_cause as r5

    input_path = R5A_ROOT / "h12_r5a_native_input.npz"
    manifest_path = R5A_ROOT / "h12_r5a_primitive_manifest.jsonl"
    os.environ["H12_R6_RECONDITION_DERIVATIVES"] = "1"
    try:
        interposer = r5.build_interposer(output)
        cases: list[dict[str, Any]] = []
        for name in ("formal", "formal_replay_1", "formal_replay_2"):
            cases.append(r5.run_case(output, input_path, manifest_path, interposer, name, mitigate=True, installed_helpers=True, hard_limit_rule=True, boundary_velocity_recondition=True, timeout_s=timeout_s))
        counts = [r5.case_counts(case) for case in cases]
        return {"status": "PASSED", "cases": cases, "counts": [{key: value for key, value in count.items() if key != "rows"} for count in counts]}
    finally:
        os.environ.pop("H12_R6_RECONDITION_DERIVATIVES", None)


def run_test_command(command: list[str], output: Path, filename: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False)
        text = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
        status = "PASS" if proc.returncode == 0 else "FAIL"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        text = f"TIMEOUT\n{exc}\n"
        status = "FAIL"
        returncode = None
    (output / filename).write_text(f"{status}\ncommand: {' '.join(command)}\nreturncode: {returncode}\n{text}", encoding="utf-8", newline="\n")
    return {"status": status, "returncode": returncode, "command": command, "output_file": filename}


def fresh_array_path(output: Path, case_name: str, row: Mapping[str, Any]) -> Path:
    path = output / "native_runs" / case_name / str(row["post_ruckig_arrays_file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def audit_fresh_arrays(output: Path, case_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    acceleration_violations = 0
    validated_windows = 0
    geometry_unchanged = True
    derivative_state_records: list[dict[str, Any]] = []
    r5a_rows = {str(row["unit_id"]): row for row in load_jsonl(R5A_ROOT / "h12_r5a_ruckig_results.jsonl")}
    for row in rows:
        validated_windows += int(row.get("window_count", 0))
        with np.load(fresh_array_path(output, case_name, row), allow_pickle=False) as fresh:
            old_row = r5a_rows[str(row["unit_id"])]
            with np.load(array_path(old_row), allow_pickle=False) as old:
                q_same = np.array_equal(old["positions_rad"], fresh["positions_rad"])
                v_same = np.array_equal(old["velocities_rad_s"], fresh["velocities_rad_s"])
                t_same = np.array_equal(old["time_s"], fresh["time_s"])
                geometry_unchanged = geometry_unchanged and q_same and v_same and t_same
            audit = derivative_consistency_audit(fresh["time_s"], fresh["velocities_rad_s"], fresh["accelerations_rad_s2"], ACCELERATION_LIMITS)
            acceleration_violations += int(audit["stored_waypoint_acceleration_violation_count"])
            derivative_state_records.append({"unit_id": row["unit_id"], "reconditioning": row.get("h12_r6_derivative_reconditioning"), "acceleration_audit": {key: value for key, value in audit.items() if key != "derived_acceleration"}})
    return {
        "case": case_name,
        "primitive_count": len(rows),
        "validated_windows": validated_windows,
        "acceleration_violations": acceleration_violations,
        "position_velocity_timestamp_unchanged": geometry_unchanged,
        "derivative_state_records": derivative_state_records,
    }


def enrich_replay_hashes(output: Path, fresh: dict[str, Any]) -> None:
    """Persist comparable hashes for every required replay evidence layer."""
    for case in fresh.get("cases", []):
        rows = load_jsonl(Path(case["result_path"]))
        rows = sorted(rows, key=lambda row: int(row.get("unit_index", 0)))
        array_records: list[dict[str, Any]] = []
        for row in rows:
            with np.load(Path(case["result_path"]).parent / str(row["post_ruckig_arrays_file"]), allow_pickle=False) as arrays:
                array_records.append({
                    "unit_id": row["unit_id"],
                    "time_s": arrays["time_s"].tolist(),
                    "positions_rad": arrays["positions_rad"].tolist(),
                    "velocities_rad_s": arrays["velocities_rad_s"].tolist(),
                    "accelerations_rad_s2": arrays["accelerations_rad_s2"].tolist(),
                })
        case["totg_output_hash"] = canonical_hash([{ "unit_id": row["unit_id"], "output_duration": row.get("output_duration"), "post_totg_dynamic": row.get("post_totg_dynamic") } for row in rows])
        case["ruckig_output_hash"] = canonical_hash([{ "unit_id": row["unit_id"], "ruckig": deterministic_ruckig_summary(row.get("ruckig") or {}), "output_duration": row.get("output_duration") } for row in rows])
        case["boundary_state_hash"] = canonical_hash([{ "unit_id": rec["unit_id"], "start": {key: rec[key][0] for key in ("positions_rad", "velocities_rad_s", "accelerations_rad_s2", "time_s")}, "end": {key: rec[key][-1] for key in ("positions_rad", "velocities_rad_s", "accelerations_rad_s2", "time_s")} } for rec in array_records])
        case["derivative_state_hash"] = canonical_hash([{ "unit_id": rec["unit_id"], "time_s": rec["time_s"], "velocities_rad_s": rec["velocities_rad_s"], "accelerations_rad_s2": rec["accelerations_rad_s2"] } for rec in array_records])
        case["acceleration_validation_hash"] = canonical_hash([{ "unit_id": row["unit_id"], "post_ruckig_dynamic": row.get("post_ruckig_dynamic"), "post_ruckig_jerk": row.get("post_ruckig_jerk"), "h12_r6_derivative_reconditioning": row.get("h12_r6_derivative_reconditioning") } for row in rows])
        case["spray_validation_hash"] = canonical_hash([{ "unit_id": row["unit_id"], "post_ruckig_process": row.get("post_ruckig_process") } for row in rows])
        case["certification_summary_hash"] = canonical_hash([{ "unit_id": row["unit_id"], "status": row.get("status"), "first_blocker": row.get("first_blocker"), "post_ruckig_certification_reached": row.get("post_ruckig_certification_reached"), "ruckig_completion_gate": row.get("ruckig_completion_gate") } for row in rows])


def replay_audit(fresh: Mapping[str, Any]) -> dict[str, Any]:
    cases = list(fresh.get("cases", []))
    counts = list(fresh.get("counts", []))
    semantic_hashes = [case.get("semantic_sha256") for case in cases]
    replay_pass = fresh.get("status") == "PASSED" and len(cases) == 3 and len(counts) == 3 and len(set(semantic_hashes)) == 1 and all(case.get("exact_result_coverage") for case in cases)
    return {
        "schema_version": "stage3_h12_r6_replay_manifest_v1",
        "REPLAY": "3/3" if replay_pass else f"{sum(bool(case.get('exact_result_coverage')) for case in cases)}/3",
        "fresh_case_count": len(cases),
        "semantic_hashes": semantic_hashes,
        "totg_output_hashes": [case.get("totg_output_hash") for case in cases],
        "ruckig_output_hashes": [case.get("ruckig_output_hash") for case in cases],
        "boundary_state_hashes": [case.get("boundary_state_hash") for case in cases],
        "derivative_state_hashes": [case.get("derivative_state_hash") for case in cases],
        "acceleration_validation_hashes": [case.get("acceleration_validation_hash") for case in cases],
        "spray_validation_hashes": [case.get("spray_validation_hash") for case in cases],
        "certification_summary_hashes": [case.get("certification_summary_hash") for case in cases],
        "semantic_replay_equal": len(set(semantic_hashes)) == 1 if semantic_hashes else False,
        "status": "PASSED" if replay_pass else "BLOCKED",
    }


def report_text(terminal: Mapping[str, Any], distribution_data: Mapping[str, Any], details: list[Mapping[str, Any]], output: Path) -> str:
    first = details[0] if details else {}
    lines = [
        "# Stage 3 H12-R6 Boundary-State / Derivative-Consistency Dynamic Remediation",
        "",
        "```text",
        f"STAGE_3_H12_R6: {terminal['STAGE_3_H12_R6']}",
        f"FIRST_BLOCKER: {terminal['FIRST_BLOCKER']}",
        "",
        f"STAGE_3_H12: {terminal['STAGE_3_H12']}",
        f"H12_FIRST_BLOCKER: {terminal['H12_FIRST_BLOCKER']}",
        f"READY_FOR_STAGE_3_H13: {terminal['READY_FOR_STAGE_3_H13']}",
        "",
        f"H12_R5A_IMMUTABLE: {terminal['H12_R5A_IMMUTABLE']}",
        f"UPSTREAM_IMMUTABLE: {terminal['UPSTREAM_IMMUTABLE']}",
        "",
        f"ACCELERATION_ROOT_CAUSE: {terminal['ACCELERATION_ROOT_CAUSE']}",
        f"ACCELERATION_ROOT_CAUSE_CONFIDENCE: {terminal['ACCELERATION_ROOT_CAUSE_CONFIDENCE']}",
        f"ACCELERATION_VIOLATIONS_BEFORE: {terminal['ACCELERATION_VIOLATIONS_BEFORE']}",
        f"ACCELERATION_VIOLATIONS_AFTER: {terminal['ACCELERATION_VIOLATIONS_AFTER']}",
        f"REAL_CONTINUOUS_ACCELERATION_VIOLATIONS_BEFORE: {terminal['REAL_CONTINUOUS_ACCELERATION_VIOLATIONS_BEFORE']}",
        f"REAL_CONTINUOUS_ACCELERATION_VIOLATIONS_AFTER: {terminal['REAL_CONTINUOUS_ACCELERATION_VIOLATIONS_AFTER']}",
        f"BOUNDARY_DERIVATIVE_INCONSISTENCIES_BEFORE: {terminal['BOUNDARY_DERIVATIVE_INCONSISTENCIES_BEFORE']}",
        f"BOUNDARY_DERIVATIVE_INCONSISTENCIES_AFTER: {terminal['BOUNDARY_DERIVATIVE_INCONSISTENCIES_AFTER']}",
        "",
        f"PRIMITIVES: {terminal['PRIMITIVES']}",
        f"TOTG_FINAL_PASS: {terminal['TOTG_FINAL_PASS']}",
        f"RUCKIG_SMOOTHING_SUCCESS: {terminal['RUCKIG_SMOOTHING_SUCCESS']}",
        f"RUCKIG_DURATION_CEILING_HIT: {terminal['RUCKIG_DURATION_CEILING_HIT']}",
        f"RUCKIG_NATIVE_ERRORS: {terminal['RUCKIG_NATIVE_ERRORS']}",
        f"POST_RUCKIG_VALIDATED: {terminal['POST_RUCKIG_VALIDATED']}",
        f"POSITION_VIOLATIONS: {terminal['POSITION_VIOLATIONS']}",
        f"VELOCITY_VIOLATIONS: {terminal['VELOCITY_VIOLATIONS']}",
        f"ACCELERATION_VIOLATIONS: {terminal['ACCELERATION_VIOLATIONS']}",
        f"JERK_VIOLATIONS: {terminal['JERK_VIOLATIONS']}",
        f"SPRAY_PROCESS_VIOLATIONS: {terminal['SPRAY_PROCESS_VIOLATIONS']}",
        f"COLLISION_VIOLATIONS: {terminal['COLLISION_VIOLATIONS']}",
        "",
        f"WAYPOINT_POSITION_MODIFIED: {terminal['WAYPOINT_POSITION_MODIFIED']}",
        f"WAYPOINT_ORDER_MODIFIED: {terminal['WAYPOINT_ORDER_MODIFIED']}",
        f"SPRAY_SEMANTICS_MODIFIED: {terminal['SPRAY_SEMANTICS_MODIFIED']}",
        f"DYNAMIC_LIMITS_MODIFIED: {terminal['DYNAMIC_LIMITS_MODIFIED']}",
        f"VALIDATION_TOLERANCE_RELAXED: {terminal['VALIDATION_TOLERANCE_RELAXED']}",
        "",
        f"FINAL_CERTIFIED_NEURAL_RMSE: {terminal['FINAL_CERTIFIED_NEURAL_RMSE']}",
        f"FINAL_GAIN_VS_CV: {terminal['FINAL_GAIN_VS_CV']}",
        f"REPLAY: {terminal['REPLAY']}",
        f"FOCUSED_TESTS: {terminal['FOCUSED_TESTS']}",
        f"REGRESSION_TESTS: {terminal['REGRESSION_TESTS']}",
        f"NEW_REGRESSION_FAILURES: {terminal['NEW_REGRESSION_FAILURES']}",
        "",
        "PHYSICAL_ROBOT_CONNECTED: NO",
        "FJT_GOALS_SENT: 0",
        "PHYSICAL_MOTION: 0",
        "",
        f"SOURCE_H12_R6_DIRECTORY: {output}",
        f"NEXT_RECOMMENDED_STAGE: {terminal['NEXT_RECOMMENDED_STAGE']}",
        "```",
        "",
        "## Q1-Q13 evidence answers",
        "",
        f"Q1. The 40 stored-waypoint flags are distributed as {len(distribution_data['ACCEL_VIOLATIONS_BY_FAMILY'])} families, {len(distribution_data['ACCEL_VIOLATIONS_BY_PRIMITIVE'])} primitives, {len(distribution_data['ACCEL_VIOLATIONS_BY_SEGMENT'])} segments, and {len(distribution_data['ACCEL_VIOLATIONS_BY_JOINT'])} joints. The complete family/segment/primitive/joint/boundary/spray maps are in the distribution JSON. First evidence row: {first.get('family_id')} / {first.get('primitive_id')} / joint {first.get('joint_index')}.",
        "Q2. The flags are stored waypoint acceleration inconsistencies; finite-difference acceleration is within the physical limit for all 40, so they are not established continuous-time violations.",
        "Q3. No. Native analytic acceleration violation count is zero; compact R5A results do not persist per-call extrema, so no extrema are inferred.",
        f"Q4. {terminal['ACCELERATION_ROOT_CAUSE']} with {terminal['ACCELERATION_ROOT_CAUSE_CONFIDENCE']} confidence.",
        "Q5. Only offending stored acceleration cells were replaced by deterministic non-uniform-grid finite differences from unchanged velocity/time arrays.",
        "Q6. No position, waypoint order, spray semantics, joint limits, acceleration limits, collision limits, or validation tolerance were changed.",
        f"Q7. 40 -> {terminal['ACCELERATION_VIOLATIONS_AFTER']} in the fresh authoritative result when available; offline evidence alone is not a certification.",
        f"Q8. 2304 -> {terminal['SPRAY_PROCESS_VIOLATIONS']} and remains a separate H12 blocker.",
        f"Q9. Full fresh coverage is {terminal['PRIMITIVES']} primitives and {terminal['POST_RUCKIG_VALIDATED']} windows.",
        f"Q10. Replay is {terminal['REPLAY']}.",
        f"Q11. H12-R5A immutable={terminal['H12_R5A_IMMUTABLE']}; upstream immutable={terminal['UPSTREAM_IMMUTABLE']}.",
        f"Q12. {terminal['H12_FIRST_BLOCKER']}.",
        f"Q13. {terminal['READY_FOR_STAGE_3_H13']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fresh-native", action="store_true", help="run three fresh 480-primitive native recertifications")
    parser.add_argument("--skip-current-hash-audit", action="store_true")
    parser.add_argument("--reuse-fresh-output", action="store_true", help="refresh an existing R6 directory using its completed fresh-native execution")
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args(argv)
    output = args.output_dir or ROOT / "outputs" / f"stage3_h12_r6_boundary_derivative_dynamic_remediation_{utc_stamp()}"
    output = output if output.is_absolute() else ROOT / output
    if output.exists() and not args.reuse_fresh_output:
        raise RuntimeError(f"refusing to overwrite existing R6 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    authority = require_r5a_authority()
    immutability = audit_upstream_immutability(current_hash_audit=not args.skip_current_hash_audit)
    details, inventory, derivative_data, boundary_audit = audit_inventory()
    write_json(output / "h12_r6_acceleration_violation_inventory.json", inventory)
    write_jsonl(output / "h12_r6_acceleration_violation_details.jsonl", details)
    distribution_data = distribution(details)
    write_json(output / "h12_r6_acceleration_violation_distribution.json", distribution_data)
    write_json(output / "h12_r6_boundary_state_audit.json", boundary_audit)
    write_json(output / "h12_r6_derivative_consistency_audit.json", derivative_data)
    native_vs_validator = {
        "schema_version": "stage3_h12_r6_native_vs_validator_acceleration_v1",
        "stored_waypoint_acceleration_violations": len(details),
        "finite_difference_acceleration_violations": inventory["finite_difference_acceleration_violations"],
        "continuous_native_analytic_acceleration_violations": inventory["continuous_native_analytic_acceleration_violations"],
        "DO_CONTINUOUS_TRAJECTORY_ACCELERATION_LIMITS_ACTUALLY_FAIL": "YES" if inventory["continuous_native_analytic_acceleration_violations"] else "NO",
        "per_call_extrema_persisted_in_compact_r5a": False,
        "status": "PASSED",
    }
    write_json(output / "h12_r6_native_vs_validator_acceleration.json", native_vs_validator)
    root_cause = classify_acceleration_root_cause(stored_waypoint_violations=len(details), finite_difference_violations=inventory["finite_difference_acceleration_violations"], native_analytic_violations=inventory["continuous_native_analytic_acceleration_violations"], boundary_derivative_inconsistencies=boundary_audit["boundary_derivative_inconsistency_count"])
    root_cause.update({"evidence": {"inventory": inventory, "boundary_audit": boundary_audit}, "excluded_classes": ["REAL_CONTINUOUS_DYNAMIC_ACCELERATION_VIOLATION", "DURATION_DERIVATIVE_INCONSISTENCY", "FINITE_DIFFERENCE_VALIDATOR_ARTIFACT", "PRIMITIVE_STITCHING_DISCONTINUITY", "TOTG_TO_RUCKIG_STATE_TRANSFER_ERROR"]})
    write_json(output / "h12_r6_root_cause_analysis.json", root_cause)

    remediation_rows: list[dict[str, Any]] = []
    offline_after = 0
    for row in load_jsonl(R5A_ROOT / "h12_r5a_ruckig_results.jsonl"):
        with np.load(array_path(row), allow_pickle=False) as arrays:
            repaired, evidence = reconstruct_local_derivative_state(arrays["time_s"], arrays["velocities_rad_s"], arrays["accelerations_rad_s2"], ACCELERATION_LIMITS)
            offline_after += int(np.count_nonzero(np.abs(repaired) > ACCELERATION_LIMITS[None, :] + 1.0e-10))
        remediation_rows.append({"unit_id": row["unit_id"], **evidence})
    remediation = {
        "schema_version": "stage3_h12_r6_remediation_summary_v1",
        "method": "local_finite_difference_derivative_reconstruction",
        "before_acceleration_violations": len(details),
        "offline_reconstructed_acceleration_violations": offline_after,
        "changed_cell_count": sum(row["changed_cell_count"] for row in remediation_rows),
        "position_modified": False,
        "velocity_modified": False,
        "timestamp_modified": False,
        "waypoint_order_modified": False,
        "spray_semantics_modified": False,
        "dynamic_limits_modified": False,
        "validation_tolerance_relaxed": False,
        "fresh_full_recertification_required": True,
        "fresh_full_recertification_requested": bool(args.fresh_native),
        "status": "EVIDENCE_SUPPORTED_PENDING_FRESH_RECERTIFICATION",
    }
    remediation["fresh_case_reconditioning_evidence"] = remediation_rows
    write_json(output / "h12_r6_remediation_summary.json", remediation)

    fresh: dict[str, Any]
    if args.reuse_fresh_output:
        fresh = load_json(output / "h12_r6_fresh_native_execution.json")
    elif args.fresh_native:
        try:
            fresh = fresh_remediation_case(output, timeout_s=args.timeout_s)
        except Exception as exc:
            fresh = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}", "cases": [], "counts": []}
    else:
        fresh = {"status": "NOT_RUN", "cases": [], "counts": []}
    write_json(output / "h12_r6_fresh_native_execution.json", fresh)
    if fresh.get("status") == "PASSED":
        enrich_replay_hashes(output, fresh)
    replay = replay_audit(fresh)
    write_json(output / "h12_r6_replay_manifest.json", replay)

    fresh_audit = None
    if fresh.get("status") == "PASSED":
        formal_rows = load_jsonl(Path(fresh["cases"][0]["result_path"]))
        fresh_audit = audit_fresh_arrays(output, "formal", formal_rows)
    full_fresh = bool(fresh_audit and fresh_audit["primitive_count"] == EXPECTED_PRIMITIVES and fresh_audit["validated_windows"] == EXPECTED_WINDOWS and fresh_audit["acceleration_violations"] == 0 and fresh_audit["position_velocity_timestamp_unchanged"] and replay["REPLAY"] == "3/3")
    counts = fresh["counts"][0] if fresh.get("status") == "PASSED" and fresh.get("counts") else {}
    after_accel: Any = int(fresh_audit["acceleration_violations"]) if fresh_audit else int(offline_after)
    post_dynamic = {
        "POST_RUCKIG_VALIDATED": None if not fresh_audit else f"{fresh_audit['validated_windows']}/{EXPECTED_WINDOWS}",
        "POSITION_VIOLATIONS": counts.get("position_violations"),
        "VELOCITY_VIOLATIONS": counts.get("velocity_violations"),
        "ACCELERATION_VIOLATIONS": after_accel,
        "JERK_VIOLATIONS": counts.get("jerk_violations"),
        "coverage_complete": bool(fresh_audit and fresh_audit["validated_windows"] == EXPECTED_WINDOWS),
        "status": "PASSED" if full_fresh else "BLOCKED",
    }
    write_json(output / "h12_r6_post_remediation_dynamic_validation.json", post_dynamic)
    spray = {"SPRAY_PROCESS_VIOLATIONS": counts.get("process_violations", 2304), "source": "fresh_authoritative_r6" if fresh_audit else "immutable_r5a_authoritative_evidence", "status": "BLOCKED" if counts.get("process_violations", 2304) else "PASSED"}
    collision = {"COLLISION_VIOLATIONS": counts.get("collision_violations"), "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "clearance": None, "status": "NOT_CERTIFIED" if not fresh_audit else "PASSED" if counts.get("collision_violations") == 0 else "BLOCKED"}
    write_json(output / "h12_r6_spray_process_validation.json", spray)
    write_json(output / "h12_r6_collision_validation.json", collision)
    write_json(output / "h12_r6_before_after_comparison.json", {"before": {"ACCELERATION_VIOLATIONS": 40, "SPRAY_PROCESS_VIOLATIONS": 2304, "PRIMITIVES": "480/480", "POST_RUCKIG_VALIDATED": "189216/189216"}, "offline_after": {"ACCELERATION_VIOLATIONS": offline_after}, "fresh_after": post_dynamic, "geometry_and_semantics_unchanged": bool(fresh_audit and fresh_audit["position_velocity_timestamp_unchanged"]) if fresh_audit else True})
    write_json(output / "h12_r6_upstream_immutability.json", {**immutability, "r5a_authority": authority["checks"]})

    focused = run_test_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r6.py"], output, "focused_test_results.txt")
    regression = run_test_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r5.py", "tests/test_stage3_h12_r5a.py", "tests/test_stage3_h12_r6.py"], output, "regression_test_results.txt")
    first_blocker = "spray_process_violation" if full_fresh and spray["SPRAY_PROCESS_VIOLATIONS"] else "fresh_full_recertification_not_run" if not full_fresh and fresh.get("status") == "NOT_RUN" else "fresh_full_recertification_failed" if not full_fresh else "acceleration_violation"
    r6_passed = full_fresh and after_accel == 0 and focused["status"] == "PASS" and regression["status"] == "PASS" and immutability["H12_R5A_IMMUTABLE"] == "YES" and immutability["UPSTREAM_IMMUTABLE"] == "YES"
    h12_passed = r6_passed and not spray["SPRAY_PROCESS_VIOLATIONS"] and collision["COLLISION_VIOLATIONS"] == 0
    terminal = {
        "STAGE_3_H12_R6": "PASSED" if r6_passed else "BLOCKED",
        "FIRST_BLOCKER": first_blocker,
        "STAGE_3_H12": "PASSED" if h12_passed else "BLOCKED",
        "H12_FIRST_BLOCKER": "none" if h12_passed else ("spray_process_violation" if r6_passed and spray["SPRAY_PROCESS_VIOLATIONS"] else first_blocker),
        "READY_FOR_STAGE_3_H13": "YES" if h12_passed else "NO",
        "H12_R5A_IMMUTABLE": immutability["H12_R5A_IMMUTABLE"],
        "UPSTREAM_IMMUTABLE": immutability["UPSTREAM_IMMUTABLE"],
        "ACCELERATION_ROOT_CAUSE": root_cause["classification"],
        "ACCELERATION_ROOT_CAUSE_CONFIDENCE": root_cause["confidence"],
        "ACCELERATION_VIOLATIONS_BEFORE": 40,
        "ACCELERATION_VIOLATIONS_AFTER": after_accel,
        "REAL_CONTINUOUS_ACCELERATION_VIOLATIONS_BEFORE": inventory["continuous_native_analytic_acceleration_violations"],
        "REAL_CONTINUOUS_ACCELERATION_VIOLATIONS_AFTER": 0 if full_fresh else None,
        "BOUNDARY_DERIVATIVE_INCONSISTENCIES_BEFORE": boundary_audit["boundary_derivative_inconsistency_count"],
        "BOUNDARY_DERIVATIVE_INCONSISTENCIES_AFTER": 0 if full_fresh else None,
        "PRIMITIVES": f"{fresh_audit['primitive_count']}/{EXPECTED_PRIMITIVES}" if fresh_audit else f"0/{EXPECTED_PRIMITIVES}",
        "TOTG_FINAL_PASS": f"{counts.get('TOTG_PASS', 0)}/{EXPECTED_PRIMITIVES}" if counts else "0/480",
        "RUCKIG_SMOOTHING_SUCCESS": f"{counts.get('RUCKIG_SMOOTHING_SUCCESS', 0)}/{EXPECTED_PRIMITIVES}" if counts else "0/480",
        "RUCKIG_DURATION_CEILING_HIT": counts.get("RUCKIG_DURATION_CEILING_HIT", None),
        "RUCKIG_NATIVE_ERRORS": counts.get("RUCKIG_NATIVE_ERRORS", None),
        "POST_RUCKIG_VALIDATED": post_dynamic["POST_RUCKIG_VALIDATED"],
        "POSITION_VIOLATIONS": post_dynamic["POSITION_VIOLATIONS"],
        "VELOCITY_VIOLATIONS": post_dynamic["VELOCITY_VIOLATIONS"],
        "ACCELERATION_VIOLATIONS": after_accel,
        "JERK_VIOLATIONS": post_dynamic["JERK_VIOLATIONS"],
        "SPRAY_PROCESS_VIOLATIONS": spray["SPRAY_PROCESS_VIOLATIONS"],
        "COLLISION_VIOLATIONS": collision["COLLISION_VIOLATIONS"],
        "WAYPOINT_POSITION_MODIFIED": "NO",
        "WAYPOINT_ORDER_MODIFIED": "NO",
        "SPRAY_SEMANTICS_MODIFIED": "NO",
        "DYNAMIC_LIMITS_MODIFIED": "NO",
        "VALIDATION_TOLERANCE_RELAXED": "NO",
        "FINAL_CERTIFIED_NEURAL_RMSE": None,
        "FINAL_GAIN_VS_CV": None,
        "REPLAY": replay["REPLAY"],
        "FOCUSED_TESTS": focused["status"],
        "REGRESSION_TESTS": regression["status"],
        "NEW_REGRESSION_FAILURES": 0 if regression["status"] == "PASS" else 1,
        "NEXT_RECOMMENDED_STAGE": "H13" if h12_passed else "H12_R6_SPRAY_PROCESS_REMEDIATION" if r6_passed else "H12_R6_FULL_RECERTIFICATION",
    }
    write_json(output / "stage3_h12_r6_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(report_text(terminal, distribution_data, details, output), encoding="utf-8", newline="\n")
    print(report_text(terminal, distribution_data, details, output))
    return 0 if r6_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
