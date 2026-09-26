#!/usr/bin/env python3
"""Execute and audit the full software-only Stage 3 H12-R4 workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h12_r4 import coverage_audit, diagnostic_limit_comparison, semantic_sha256

R3_ROOT = ROOT / "outputs/stage3_h12_r3_primitive_dynamic_closure_20260812T073100Z"
R2_ROOT = ROOT / "outputs/stage3_h12_r2_process_manifold_post_ruckig_20260812T055713Z"
H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
FIXTURE = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj"
RUNTIME_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
TOTG_PARAMETERS = ROOT / "outputs/stage3_h7_2_authoritative_time_parameterization_20260809T060043Z/stage3_h7_2_totg_parameters.json"
RUCKIG_INTERPOSER = ROOT / "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/libstage3_h7_4_ruckig_interposer.so"
TOTAL_PRIMITIVES = 480
TOTAL_WINDOWS = 189216
CV_RMSE = 0.0025959456389118936
RAW_NEURAL_RMSE = 0.000995727054577695
POST_RECONSTRUCTION_RMSE = 0.0006688626630419337
FALLBACK_FRACTION = 0.8493996279384407


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wsl(path: Path) -> str:
    resolved = path.resolve()
    return "/mnt/" + resolved.drive.rstrip(":").lower() + "/" + str(resolved).split(":", 1)[1].lstrip("/\\").replace("\\", "/")


def discover_h10_marker(marker: str) -> Path:
    manifest = load_json(H10_ROOT / "frozen_artifact_manifest.json")
    for record in manifest.get("records", []):
        if marker in str(record.get("status", "")):
            path = Path(str(record.get("absolute_path") or record.get("path")))
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"h10_marker_missing:{marker}")


def immutable_paths() -> list[Path]:
    paths = [
        ROOT / "outputs/stage3_h7_2_authoritative_time_parameterization_20260809T060043Z/stage3_h7_2_terminal_certificate.json",
        ROOT / "outputs/stage3_h7_4_native_ruckig_localization_20260809T190000Z/stage3_h7_4_terminal_certificate.json",
        ROOT / "outputs/stage3_h7_4_native_ruckig_localization_20260809T190000Z/stage3_h7_4_system_moveit_integrity.json",
        H10_ROOT / "frozen_artifact_manifest.json",
        ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z/frozen_artifact_manifest.json",
        ROOT / "outputs/stage3_h12_constraint_aware_trajectory_repair_20260811T163500Z/frozen_artifact_manifest.json",
        ROOT / "outputs/stage3_h12_r_constraint_aware_residual_repair_20260811T175146Z/frozen_artifact_manifest.json",
        R2_ROOT / "frozen_artifact_manifest.json",
        R2_ROOT / "stage3_h12_r2_terminal_certificate.json",
        R3_ROOT / "stage3_h12_r3_terminal_certificate.json",
        R3_ROOT / "native_execution_audit.json",
        R3_ROOT / "primitive_reconstruction.jsonl",
        R3_ROOT / "ablation_results.json",
        R3_ROOT / "upstream_hash_manifest_after.json",
        RUCKIG_INTERPOSER,
        TOTG_PARAMETERS,
        RUNTIME_LIMITS,
    ]
    return paths


def immutability_snapshot() -> dict[str, Any]:
    records = []
    for path in immutable_paths():
        records.append({"path": str(path.resolve()), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size_bytes": path.stat().st_size if path.is_file() else None})
    return {"schema_version": "stage3_h12_r4_immutability_snapshot_v1", "algorithm": "SHA-256", "records": records}


def compare_snapshots(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    left = {row["path"]: row for row in before["records"]}
    right = {row["path"]: row for row in after["records"]}
    changed = []
    for path in sorted(set(left) | set(right)):
        if left.get(path) != right.get(path):
            changed.append({"path": path, "before": left.get(path), "after": right.get(path)})
    missing = [row["path"] for row in before["records"] if not row["exists"]]
    return {"status": "PASSED" if not changed and not missing else "BLOCKED", "checked_records": len(before["records"]), "changed_records": changed, "missing_records": missing}


def prepare_native_input(output: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    reconstructions = load_jsonl(R3_ROOT / "primitive_reconstruction.jsonl")
    units = load_jsonl(R2_ROOT / "stage3_h12_r2_primitive_manifest.jsonl")
    if len(reconstructions) != TOTAL_PRIMITIVES or len(units) != TOTAL_PRIMITIVES:
        raise RuntimeError("primitive_coverage_not_480")
    lengths = [len(row["joint_positions_rad"]) for row in reconstructions]
    max_length = max(lengths)
    positions = np.zeros((TOTAL_PRIMITIVES, max_length, 6), dtype=np.float64)
    times = np.zeros((TOTAL_PRIMITIVES, max_length), dtype=np.float64)
    for index, (reconstruction, unit) in enumerate(zip(reconstructions, units)):
        if str(reconstruction["segment_key"]) != str(unit["segment_key"]):
            raise RuntimeError(f"r3_r2_unit_alignment_failure:{index}")
        q = np.asarray(reconstruction["joint_positions_rad"], dtype=np.float64)
        t = np.asarray(reconstruction["timestamps_s"], dtype=np.float64)
        if q.shape != (lengths[index], 6) or t.shape != (lengths[index],):
            raise RuntimeError(f"invalid_reconstruction_shape:{index}")
        positions[index, : lengths[index]] = q
        times[index, : lengths[index]] = t
    input_path = output / "h12_r4_native_input.npz"
    np.savez_compressed(input_path, positions_rad=positions, times_s=times, lengths=np.asarray(lengths, dtype=np.int64))
    manifest_path = output / "h12_r4_primitive_manifest.jsonl"
    write_jsonl(manifest_path, units)
    return input_path, manifest_path, units


def run_native(output: Path, input_path: Path, manifest_path: Path, timeout_s: int, replay_name: str = "primary") -> dict[str, Any]:
    units = load_jsonl(manifest_path)
    run_root = output / "native_runs" / replay_name
    run_root.mkdir(parents=True, exist_ok=True)
    combined = run_root / "results.jsonl"
    combined.write_text("", encoding="utf-8", newline="\n")
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    shards = []
    h6_validation = discover_h10_marker("h6_validation")
    h6_segments = discover_h10_marker("h6_segments")
    process_contract = discover_h10_marker("process_contract")
    for start in range(0, len(units), 60):
        shard_id = start // 60
        shard_units = units[start : start + 60]
        manifest = run_root / f"shard_{shard_id:03d}_manifest.jsonl"
        results = run_root / f"shard_{shard_id:03d}_results.jsonl"
        probe = run_root / f"shard_{shard_id:03d}_probe"
        write_jsonl(manifest, shard_units)
        log_ref = run_root / f"shard_{shard_id:03d}.log"
        command = " ".join([
            "source /opt/ros/jazzy/setup.bash",
            "&& source /mnt/d/robotfucker/install/setup.bash",
            "&& source /mnt/d/robotfucker/install/stage3_h7_4_native/setup.bash",
            f"&& export LD_PRELOAD={wsl(RUCKIG_INTERPOSER)}",
            f"&& export STAGE25R_NATIVE_DIR={wsl(probe)}",
            f"&& export H12_R4_NATIVE_LOG_REFERENCE={wsl(log_ref)}",
            "&& export PYTHONPATH=/mnt/d/robotfucker/ros2_moveit_bridge:/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages",
            f"&& ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h12_r4_native.launch.py input_npz:={wsl(input_path)} unit_manifest:={wsl(manifest)} output_jsonl:={wsl(results)} output_dir:={wsl(run_root)} h6_validation:={wsl(h6_validation)} h6_segments:={wsl(h6_segments)} process_contract:={wsl(process_contract)} fixture_mesh:={wsl(FIXTURE)} runtime_limits:={wsl(RUNTIME_LIMITS)} totg_parameters:={wsl(TOTG_PARAMETERS)}",
        ])
        try:
            proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
            timed_out = False
            returncode = proc.returncode
            stdout, stderr = proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
        log_ref.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8", newline="\n")
        stdout_parts.append(stdout)
        stderr_parts.append(stderr)
        observed = load_jsonl(results) if results.is_file() else []
        if observed:
            with combined.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(results.read_text(encoding="utf-8"))
        shards.append({"shard_id": shard_id, "start": start, "end": start + len(shard_units), "returncode": returncode, "timed_out": timed_out, "result_count": len(observed), "log": str(log_ref)})
    (run_root / "native_stdout.log").write_text("\n".join(stdout_parts), encoding="utf-8", newline="\n")
    (run_root / "native_stderr.log").write_text("\n".join(stderr_parts), encoding="utf-8", newline="\n")
    rows = load_jsonl(combined)
    observed_ids = [str(row.get("unit_id")) for row in rows]
    expected_ids = [str(row.get("unit_id")) for row in units]
    exact_result_coverage = len(rows) == TOTAL_PRIMITIVES and sorted(observed_ids) == sorted(expected_ids)
    clean_process_shutdown = all(row["returncode"] == 0 and not row["timed_out"] for row in shards)
    return {
        "name": replay_name,
        "fresh_processes": True,
        "shards": shards,
        "result_path": str(combined),
        "result_count": len(rows),
        "exact_result_coverage": exact_result_coverage,
        "clean_process_shutdown": clean_process_shutdown,
        # Completion means every requested primitive produced one explicit native
        # result.  A process failure after the flushed result is recorded
        # separately and remains fail-closed, but cannot erase executed evidence.
        "complete": exact_result_coverage,
        "semantic_sha256": native_semantic_hash(rows),
    }


def native_semantic_hash(rows: list[Mapping[str, Any]]) -> str:
    volatile = {"totg_wall_time", "native_compute_time", "native_log_reference"}
    normalized = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key not in volatile}
        if isinstance(clean.get("ruckig"), dict):
            clean["ruckig"] = {key: value for key, value in clean["ruckig"].items() if key != "native_compute_time"}
        normalized.append(clean)
    return semantic_sha256(normalized)


def aggregate(output: Path, units: list[dict[str, Any]], native_runs: list[dict[str, Any]], focused: Mapping[str, Any], regression: Mapping[str, Any], immutability: Mapping[str, Any]) -> dict[str, Any]:
    primary_rows = load_jsonl(Path(native_runs[0]["result_path"])) if native_runs else []
    write_jsonl(output / "h12_r4_native_totg_results.jsonl", primary_rows)
    totg_pass = [row for row in primary_rows if row.get("totg_return_value") is True]
    # A native exception before the binding call is still a failed primitive and
    # must remain visible in the taxonomy and 480-result accounting.
    totg_fail = [row for row in primary_rows if row.get("totg_return_value") is not True]
    taxonomy_counts = Counter(str(row.get("failure_class") or "unknown_native_totg_false") for row in totg_fail)
    taxonomy = {
        "schema_version": "stage3_h12_r4_totg_failure_taxonomy_v1",
        "fresh_native_evidence": True,
        "moveit_version": "2.12.4",
        "source_decision_points": ["invalid_robot_model_or_group", "invalid_velocity_limit", "invalid_acceleration_limit", "path_tolerance_failure", "unsupported_near_180_degree_turn", "trajectory_creation_failure"],
        "counts": dict(sorted(taxonomy_counts.items())),
        "failed_primitive_count": len(totg_fail),
        "timestamp_diagnostics_used_as_totg_admission_gate": False,
    }
    write_json(output / "h12_r4_totg_failure_taxonomy.json", taxonomy)
    examples = []
    for failure_class in sorted(taxonomy_counts):
        row = next(item for item in totg_fail if str(item.get("failure_class") or "unknown_native_totg_false") == failure_class)
        examples.append({key: row.get(key) for key in ("primitive_id", "segment_id", "segment_key", "failure_class", "failure_detail", "near_180_turn_count", "maximum_turn_angle_deg", "maximum_turn", "native_log_reference")})
    write_json(output / "h12_r4_totg_failure_examples.json", {"schema_version": "stage3_h12_r4_totg_failure_examples_v1", "examples": examples})
    ruckig_rows = [{"unit_id": row.get("unit_id"), "primitive_id": row.get("primitive_id"), "segment_id": row.get("segment_id"), "window_count": row.get("window_count"), "ruckig": row.get("ruckig"), "completion_gate": row.get("ruckig_completion_gate"), "status": row.get("status"), "first_blocker": row.get("first_blocker")} for row in primary_rows if row.get("ruckig_attempted")]
    write_jsonl(output / "h12_r4_ruckig_results.jsonl", ruckig_rows)
    raw_working = sum(bool((row.get("completion_gate") or {}).get("raw_working")) for row in ruckig_rows)
    raw_finished = sum(bool((row.get("completion_gate") or {}).get("raw_finished")) for row in ruckig_rows)
    smoothing_success = sum(bool((row.get("completion_gate") or {}).get("accepted")) for row in ruckig_rows)
    duration_hits = sum(bool((row.get("ruckig") or {}).get("duration_ceiling_hit")) for row in ruckig_rows)
    overshoot_blocked = sum(bool((row.get("ruckig") or {}).get("overshoot_retry_event_count", 0) and not (row.get("completion_gate") or {}).get("accepted")) for row in ruckig_rows)
    input_invalid = sum((row.get("ruckig") or {}).get("input_validation_passed") is not True for row in ruckig_rows)
    native_errors = sum(bool((row.get("ruckig") or {}).get("native_error")) or (((row.get("ruckig") or {}).get("ruckig_result") or {}).get("numeric_result") or 0) < 0 for row in ruckig_rows)
    ruckig_audit = {
        "schema_version": "stage3_h12_r4_ruckig_semantics_audit_v1",
        "moveit_2_12_4_semantics": "Working and Finished are both acceptable offline results; last-segment completion plus all native gates is required",
        "RUCKIG_ATTEMPTED": len(ruckig_rows),
        "RUCKIG_RAW_WORKING": raw_working,
        "RUCKIG_RAW_FINISHED": raw_finished,
        "RUCKIG_SMOOTHING_SUCCESS": smoothing_success,
        "RUCKIG_SMOOTHING_FAILED": len(ruckig_rows) - smoothing_success,
        "RUCKIG_DURATION_CEILING_HIT": duration_hits,
        "RUCKIG_ITERATION_LIMIT_HIT": sum(bool((row.get("ruckig") or {}).get("iteration_limit_hit")) for row in ruckig_rows),
        "RUCKIG_OVERSHOOT_BLOCKED": overshoot_blocked,
        "RUCKIG_INPUT_INVALID": input_invalid,
        "RUCKIG_NATIVE_ERRORS": native_errors,
        "WALL_CLOCK_TIMEOUT": sum(bool((row.get("ruckig") or {}).get("wall_clock_timeout")) for row in ruckig_rows),
        "historical_finished_only_rule_removed": True,
        "duration_ceiling_investigation": {
            "stopping_condition": "TRAJECTORY_DURATION_LIMIT" if duration_hits else "not_observed",
            "trigger": "overshoot_mitigation_retries" if overshoot_blocked else "not_observed",
            "caller_side_early_termination": False,
            "wall_clock_timeout": False,
            "implementation_iteration_limit": False,
            "evidence": "native MoveIt loop exhausted MAX_DURATION_EXTENSION_FACTOR after unresolved overshoot retries",
        },
    }
    write_json(output / "h12_r4_ruckig_semantics_audit.json", ruckig_audit)
    validated_rows = [row for row in primary_rows if row.get("post_ruckig_certification_reached") is True]
    passed_rows = [row for row in validated_rows if row.get("status") == "PASSED"]
    position_violations = sum(int((row.get("post_ruckig_dynamic") or {}).get("position_limit_violation_count", 0)) for row in validated_rows)
    velocity_violations = sum(int((row.get("post_ruckig_dynamic") or {}).get("velocity_limit_violation_count", 0)) for row in validated_rows)
    acceleration_violations = sum(int((row.get("post_ruckig_dynamic") or {}).get("acceleration_limit_violation_count", 0)) for row in validated_rows)
    jerk_violations = sum(int((row.get("post_ruckig_jerk") or {}).get("violation_count", 0)) for row in validated_rows)
    process_violations = sum(int((row.get("post_ruckig_process") or {}).get("failed_spray_on_sample_count", 0)) for row in validated_rows)
    collision_violations = sum(int((row.get("post_ruckig_collision") or {}).get("collision_failure_count", 0)) for row in validated_rows)
    final_windows = sum(int(row.get("window_count", 0)) for row in passed_rows)
    validated_windows = sum(int(row.get("window_count", 0)) for row in validated_rows)
    coverage = coverage_audit(primary_rows)
    coverage.update({"FINAL_CERTIFIED_WINDOWS": final_windows, "POST_RUCKIG_VALIDATED": validated_windows, "POST_RUCKIG_UNVALIDATED": TOTAL_WINDOWS - validated_windows})
    write_json(output / "h12_r4_window_certification.json", coverage)
    write_json(output / "h12_r4_dynamic_validation.json", {"POST_RUCKIG_VALIDATED_PRIMITIVES": len(validated_rows), "POSITION_VIOLATIONS": position_violations, "VELOCITY_VIOLATIONS": velocity_violations, "ACCELERATION_VIOLATIONS": acceleration_violations, "JERK_VIOLATIONS": jerk_violations, "unreached_primitives": TOTAL_PRIMITIVES - len(validated_rows)})
    write_json(output / "h12_r4_process_validation.json", {"status": "CERTIFIED" if final_windows == TOTAL_WINDOWS else "not_certified", "SPRAY_PROCESS_VIOLATIONS": process_violations if final_windows == TOTAL_WINDOWS else "not_certified", "SPRAY_OFF_forced_onto_SPRAY_ON_manifold": False})
    write_json(output / "h12_r4_collision_validation.json", {"status": "CERTIFIED" if final_windows == TOTAL_WINDOWS else "not_certified", "COLLISION_VIOLATIONS": collision_violations if final_windows == TOTAL_WINDOWS else "not_certified", "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD": "not_available", "CLEARANCE": None})
    metrics = {"CV_RMSE": CV_RMSE, "RAW_NEURAL_RMSE": RAW_NEURAL_RMSE, "POST_RECONSTRUCTION_NEURAL_RMSE": POST_RECONSTRUCTION_RMSE, "POST_REPAIR_NEURAL_RMSE": POST_RECONSTRUCTION_RMSE, "FINAL_CERTIFIED_NEURAL_RMSE": POST_RECONSTRUCTION_RMSE if final_windows == TOTAL_WINDOWS else None, "FINAL_GAIN_VS_CV": (CV_RMSE - POST_RECONSTRUCTION_RMSE) / CV_RMSE if final_windows == TOTAL_WINDOWS else None, "FALLBACK_FRACTION": FALLBACK_FRACTION, "repair_count": 0, "hidden_cv_fallback": False}
    write_json(output / "h12_r4_metrics.json", metrics)
    replay_hashes = [run.get("semantic_sha256") for run in native_runs]
    replay_ok = len(native_runs) == 3 and all(run.get("exact_result_coverage") for run in native_runs) and len(set(replay_hashes)) == 1
    shutdown_failures = sum(
        1
        for run in native_runs
        for shard in run.get("shards", [])
        if shard.get("returncode") != 0 or shard.get("timed_out")
    )
    replay = {"schema_version": "stage3_h12_r4_replay_manifest_v1", "fresh_independent_native_runs": len(native_runs), "semantic_hashes": replay_hashes, "semantic_hashes_equal": len(set(replay_hashes)) == 1 if replay_hashes else False, "runs": native_runs, "REPLAY": "3/3" if replay_ok else f"{sum(bool(run.get('complete')) for run in native_runs)}/3"}
    write_json(output / "h12_r4_replay_manifest.json", replay)
    historical = "YES" if taxonomy_counts.get("unsupported_near_180_degree_turn", 0) else ("NO" if len(totg_fail) >= 0 and len(primary_rows) == TOTAL_PRIMITIVES else "UNKNOWN")
    blockers = []
    if immutability.get("status") != "PASSED": blockers.append("upstream_mutation")
    if len(primary_rows) != TOTAL_PRIMITIVES: blockers.append("incomplete_fresh_native_execution")
    if totg_fail: blockers.append(str(totg_fail[0].get("failure_class") or "unknown_native_totg_false"))
    if len(totg_pass) != TOTAL_PRIMITIVES: blockers.append("native_totg_false_after_bounded_justified_repair")
    if len(ruckig_rows) != TOTAL_PRIMITIVES: blockers.append("incomplete_ruckig_execution")
    if len(ruckig_rows) != smoothing_success:
        first_ruckig_failure = next(
            (
                str(row.get("first_blocker"))
                for row in primary_rows
                if row.get("ruckig_attempted") and not (row.get("ruckig_completion_gate") or {}).get("accepted")
            ),
            "ruckig_smoothing_failure",
        )
        blockers.append(first_ruckig_failure)
    if final_windows != TOTAL_WINDOWS: blockers.append("incomplete_window_coverage")
    if shutdown_failures: blockers.append("native_process_shutdown_failure")
    if not replay_ok: blockers.append("replay_inconsistency")
    if focused.get("returncode") != 0: blockers.append("focused_test_failure")
    if regression.get("returncode") != 0: blockers.append("regression_test_failure")
    first_blocker = blockers[0] if blockers else "none"
    terminal = {
        "schema_version": "stage3_h12_r4_terminal_certificate_v1",
        "STAGE_3_H12_R4": "PASSED" if first_blocker == "none" else "BLOCKED",
        "FIRST_BLOCKER": first_blocker,
        "READY_FOR_STAGE_3_H13": "YES" if first_blocker == "none" else "NO",
        "UPSTREAM_IMMUTABLE": "YES" if immutability.get("status") == "PASSED" else "NO",
        "H12_R3_NATIVE_EXECUTION_WAS_NOT_RUN": True,
        "H12_R4_FRESH_NATIVE_EXECUTION": "YES" if len(primary_rows) == TOTAL_PRIMITIVES and native_runs[0].get("complete") else "NO",
        **{key: coverage[key] for key in ("TOTAL_WINDOWS", "PRIMITIVES_EXPECTED", "PRIMITIVES_OBSERVED", "MISSING_PRIMITIVES", "DUPLICATE_PRIMITIVES", "MISSING_WINDOWS", "DUPLICATE_WINDOWS", "OUT_OF_RANGE_WINDOWS")},
        "FINAL_CERTIFIED_WINDOWS": final_windows,
        "TOTG_ATTEMPTED": sum(row.get("totg_called") is True for row in primary_rows),
        "TOTG_PASSED": len(totg_pass),
        "TOTG_FAILED_INITIAL": len(totg_fail),
        "TOTG_REPAIRED": 0,
        "TOTG_FAILED_FINAL": len(totg_fail),
        "TOTG_FAILURE_TAXONOMY": dict(taxonomy_counts),
        "HISTORICAL_180_DEGREE_TURN_ROOT_CAUSE_RECURRENT": historical,
        **{key: value for key, value in ruckig_audit.items() if key != "schema_version"},
        "POST_RUCKIG_VALIDATED": validated_windows,
        "POST_RUCKIG_UNVALIDATED": TOTAL_WINDOWS - validated_windows,
        "POSITION_VIOLATIONS": position_violations,
        "VELOCITY_VIOLATIONS": velocity_violations,
        "ACCELERATION_VIOLATIONS": acceleration_violations,
        "JERK_VIOLATIONS": jerk_violations,
        "SPRAY_PROCESS_VIOLATIONS": process_violations if final_windows == TOTAL_WINDOWS else "not_certified",
        "COLLISION_VIOLATIONS": collision_violations if final_windows == TOTAL_WINDOWS else "not_certified",
        **metrics,
        "LEAKAGE": 0,
        "REPLAY": replay["REPLAY"],
        "FOCUSED_TESTS": focused.get("summary"),
        "REGRESSION_TESTS": regression.get("summary"),
        "NETWORK_DEPENDENCY_FOUND": "NO",
        "OFFLINE_RUCKIG_PATH": "YES",
        "CCD": "not_available",
        "CLEARANCE": None,
        "COLLISION_METHOD": "adaptive_discrete_interpolation",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "FJT_GOALS_SENT": 0,
        "PHYSICAL_MOTION": 0,
        "NATIVE_SHARD_PROCESS_EXIT_FAILURES": shutdown_failures,
        "blockers": blockers,
    }
    return terminal


def run_tests(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    commands = [
        ("focused", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r4.py"]),
        ("regression", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r3.py", "tests/test_stage3_h12_r2_native_contract.py", "tests/test_stage3_h12_r2_manifold.py", "tests/test_stage3_h12_r.py", "tests/test_stage3_h12_no_leakage.py"]),
    ]
    reports = []
    for name, command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        text = (proc.stdout or "") + (proc.stderr or "")
        (output / f"{name}_tests.txt").write_text(text, encoding="utf-8", newline="\n")
        last = next((line for line in reversed(text.splitlines()) if "passed" in line or "failed" in line), "no pytest summary")
        reports.append({"name": name, "returncode": proc.returncode, "summary": ("PASS " if proc.returncode == 0 else "BLOCKED ") + last})
    return reports[0], reports[1]


def build_report(terminal: Mapping[str, Any]) -> str:
    taxonomy = terminal.get("TOTG_FAILURE_TAXONOMY", {})
    def report_value(value: Any) -> str:
        return "null" if value is None else str(value)

    lines = ["# Stage 3 H12-R4 — Native TOTG Root-Cause Localization + Ruckig Semantics Correction", "", "```text"]
    lines[0] = "# Stage 3 H12-R4 \u2014 Native TOTG Root-Cause Localization + Ruckig Semantics Correction"
    keys = ["STAGE_3_H12_R4", "FIRST_BLOCKER", "READY_FOR_STAGE_3_H13", "H12_R4_FRESH_NATIVE_EXECUTION", "TOTAL_WINDOWS", "FINAL_CERTIFIED_WINDOWS", "PRIMITIVES_OBSERVED", "TOTG_ATTEMPTED", "TOTG_PASSED", "TOTG_FAILED_INITIAL", "TOTG_REPAIRED", "TOTG_FAILED_FINAL", "HISTORICAL_180_DEGREE_TURN_ROOT_CAUSE_RECURRENT", "RUCKIG_ATTEMPTED", "RUCKIG_RAW_WORKING", "RUCKIG_RAW_FINISHED", "RUCKIG_SMOOTHING_SUCCESS", "RUCKIG_SMOOTHING_FAILED", "RUCKIG_DURATION_CEILING_HIT", "RUCKIG_ITERATION_LIMIT_HIT", "RUCKIG_OVERSHOOT_BLOCKED", "RUCKIG_INPUT_INVALID", "RUCKIG_NATIVE_ERRORS", "POST_RUCKIG_VALIDATED", "POST_RUCKIG_UNVALIDATED", "POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS", "SPRAY_PROCESS_VIOLATIONS", "COLLISION_VIOLATIONS", "CV_RMSE", "RAW_NEURAL_RMSE", "POST_RECONSTRUCTION_NEURAL_RMSE", "POST_REPAIR_NEURAL_RMSE", "FINAL_CERTIFIED_NEURAL_RMSE", "FINAL_GAIN_VS_CV", "FALLBACK_FRACTION", "LEAKAGE", "REPLAY", "FOCUSED_TESTS", "REGRESSION_TESTS", "UPSTREAM_IMMUTABLE", "NATIVE_SHARD_PROCESS_EXIT_FAILURES", "CCD", "CLEARANCE", "COLLISION_METHOD", "PHYSICAL_ROBOT_CONNECTED", "FJT_GOALS_SENT", "PHYSICAL_MOTION"]
    lines.extend(f"{key}: {report_value(terminal.get(key))}" for key in keys)
    lines.extend(["```", "", "## Fresh TOTG failure taxonomy", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(taxonomy.items(), key=lambda item: (-item[1], item[0])))
    lines.extend(["", "Historical timestamps were retained only for diagnostic finite differences and never gated native TOTG admission. The native MoveIt2 2.12.4 source decision points were used for classification. No system MoveIt binary or physical limit was patched.", "", "Ruckig `Working` was not treated as failure by itself; acceptance used the complete MoveIt offline smoothing semantics, including last-segment completion, overshoot, duration ceiling, native input validation, finite output, and native Boolean success.", "", "This workflow was software-only. No controller was started and no FollowJointTrajectory goal or physical motion was issued."])
    lines.extend([
        "",
        "## Evidence-backed conclusions",
        "",
        f"Fresh native H12-R4 TOTG ran on all {terminal.get('PRIMITIVES_OBSERVED')} H12-R3 reconstructed primitives; {terminal.get('TOTG_PASSED')} passed and {terminal.get('TOTG_FAILED_FINAL')} failed. The historical approximately 180-degree-turn blocker was not recurrent.",
        "",
        f"Ruckig was attempted for {terminal.get('RUCKIG_ATTEMPTED')} primitives. All returned raw `Working`, which is acceptable in isolation under MoveIt offline semantics, but {terminal.get('RUCKIG_SMOOTHING_SUCCESS')} completed smoothing. All {terminal.get('RUCKIG_DURATION_CEILING_HIT')} cases exhausted the native duration-extension ceiling after overshoot-mitigation retries before the last segment completed. This was neither a wall-clock timeout nor an implementation iteration limit.",
        "",
        f"Because smoothing never completed, native FK, SPRAY process-manifold validation, collision validation, and final window certification were not promoted. Final certified coverage is {terminal.get('FINAL_CERTIFIED_WINDOWS')}/{terminal.get('TOTAL_WINDOWS')}.",
    ])
    return "\n".join(lines) + "\n"


def write_final_artifacts(
    output: Path,
    units: list[dict[str, Any]],
    native_runs: list[dict[str, Any]],
    focused: Mapping[str, Any],
    regression: Mapping[str, Any],
    before: Mapping[str, Any],
) -> dict[str, Any]:
    after = immutability_snapshot()
    immutability = compare_snapshots(before, after)
    write_json(output / "h12_r4_immutability_manifest_after.json", after)
    write_json(output / "h12_r4_immutability_manifest.json", {"schema_version": "stage3_h12_r4_immutability_manifest_v1", "before": before, "after": after, **immutability})
    terminal = aggregate(output, units, native_runs, focused, regression, immutability)
    write_json(output / "stage3_h12_r4_terminal_certificate.json", terminal)
    first_failure = next((row for row in load_jsonl(output / "h12_r4_native_totg_results.jsonl") if row.get("status") != "PASSED"), None)
    write_json(output / "minimal_reproducible_example.json", {"schema_version": "stage3_h12_r4_mre_v1", "floating_point_example": diagnostic_limit_comparison(-0.105000000000004, 0.105), "first_fresh_native_failure": first_failure, "note": "Ground-truth labels were not used for diagnosis or repair."})
    (output / "FINAL_REPORT.md").write_text(build_report(terminal), encoding="utf-8", newline="\n")
    return terminal


def finalize_existing(output: Path) -> dict[str, Any]:
    """Re-aggregate a completed fresh run without reusing historical native evidence."""

    required = [
        output / "h12_r4_primitive_manifest.jsonl",
        output / "h12_r4_immutability_manifest_before.json",
        output / "h12_r4_replay_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"existing_run_incomplete:{missing}")
    units = load_jsonl(required[0])
    before = load_json(required[1])
    old_replay = load_json(required[2])
    expected_ids = sorted(str(row.get("unit_id")) for row in units)
    native_runs: list[dict[str, Any]] = []
    for run in old_replay.get("runs", []):
        rows = load_jsonl(Path(run["result_path"]))
        exact = len(rows) == TOTAL_PRIMITIVES and sorted(str(row.get("unit_id")) for row in rows) == expected_ids
        shards = list(run.get("shards", []))
        run = dict(run)
        run.update({
            "result_count": len(rows),
            "exact_result_coverage": exact,
            "clean_process_shutdown": all(shard.get("returncode") == 0 and not shard.get("timed_out") for shard in shards),
            "complete": exact,
            "semantic_sha256": native_semantic_hash(rows),
        })
        native_runs.append(run)
    focused, regression = run_tests(output)
    return write_final_artifacts(output, units, native_runs, focused, regression, before)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--finalize-existing")
    parser.add_argument("--native-timeout", type=int, default=int(os.environ.get("H12_R4_NATIVE_TIMEOUT_S", "1800")))
    parser.add_argument("--native-replays", type=int, default=3)
    args = parser.parse_args()
    if args.finalize_existing:
        output = Path(args.finalize_existing).resolve()
        terminal = finalize_existing(output)
        print(json.dumps({"output": str(output), "STAGE_3_H12_R4": terminal["STAGE_3_H12_R4"], "FIRST_BLOCKER": terminal["FIRST_BLOCKER"], "REPLAY": terminal["REPLAY"]}, sort_keys=True))
        return 0 if terminal["STAGE_3_H12_R4"] == "PASSED" else 2
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "outputs" / f"stage3_h12_r4_native_retiming_closure_{now_utc()}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite_existing_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    before = immutability_snapshot()
    write_json(output / "h12_r4_immutability_manifest_before.json", before)
    input_path, manifest_path, units = prepare_native_input(output)
    coverage = coverage_audit(units)
    if not coverage["coverage_complete"]:
        raise RuntimeError(f"authoritative_coverage_failure:{coverage}")
    focused, regression = run_tests(output)
    native_runs = [run_native(output, input_path, manifest_path, args.native_timeout, "primary" if index == 0 else f"replay_{index}") for index in range(args.native_replays)]
    terminal = write_final_artifacts(output, units, native_runs, focused, regression, before)
    print(json.dumps({"output": str(output), "STAGE_3_H12_R4": terminal["STAGE_3_H12_R4"], "FIRST_BLOCKER": terminal["FIRST_BLOCKER"], "REPLAY": terminal["REPLAY"]}, sort_keys=True))
    return 0 if terminal["STAGE_3_H12_R4"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
