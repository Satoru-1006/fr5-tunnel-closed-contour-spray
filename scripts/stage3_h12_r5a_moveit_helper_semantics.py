#!/usr/bin/env python3
"""Stage 3 H12-R5A: prove installed helper semantics, patch, and recertify."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h12_r4_native_totg_closure as r4
from scripts import stage3_h12_r5_ruckig_root_cause as r5
from src.stage3_h12_r5a import (
    classify_installed_helper_semantics,
    extension_summary,
    semantic_closure_gate,
    semantic_sha256,
    spray_validation_schema,
    validate_extension_mutation,
)

TOTAL_PRIMITIVES = 480
TOTAL_WINDOWS = 189216
R5_ROOT = ROOT / "outputs/stage3_h12_r5_ruckig_overshoot_root_cause_20260812T131826Z"
INTERPOSER = ROOT / "tools/stage25r_ruckig_interposer.cpp"
EXPECTED_LIBRARY_SHA256 = "79a6281ccf4b00d368117123a19a7a659d10fc330f42c119dfd9a4d2b867a4c9"
LIBRARY = "/opt/ros/jazzy/lib/libmoveit_trajectory_processing.so"
HEADER = "/opt/ros/jazzy/include/moveit_core/moveit/trajectory_processing/ruckig_traj_smoothing.hpp"
OFFICIAL_CPP_URL = "https://raw.githubusercontent.com/moveit/moveit2/2.12.4/moveit_core/trajectory_processing/src/ruckig_traj_smoothing.cpp"
OFFICIAL_HPP_URL = "https://raw.githubusercontent.com/moveit/moveit2/2.12.4/moveit_core/trajectory_processing/include/moveit/trajectory_processing/ruckig_traj_smoothing.hpp"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def wsl_capture(command: str, timeout: int = 120) -> tuple[int, str]:
    process = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return process.returncode, (process.stdout or "") + (process.stderr or "")


def immutable_paths() -> list[Path]:
    paths = list(r4.immutable_paths())
    paths.extend(path for path in R5_ROOT.rglob("*") if path.is_file())
    return sorted(set(path.resolve() for path in paths), key=str)


def snapshot() -> dict[str, Any]:
    return {
        str(path): {
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
            "h12_r5_evidence": R5_ROOT.resolve() in path.parents,
        }
        for path in immutable_paths()
    }


def provenance(output: Path) -> dict[str, Any]:
    commands = {
        "stat": f"stat -Lc '%n|%s' {LIBRARY}",
        "library_sha256": f"sha256sum {LIBRARY}",
        "header_sha256": f"sha256sum {HEADER}",
        "package_owner": f"dpkg-query -S {LIBRARY}",
        "package": "dpkg-query -W ros-jazzy-moveit-core",
        "package_architecture": "dpkg --print-architecture",
        "architecture": f"file {LIBRARY}",
        "symbol": f"nm -D -C {LIBRARY} | grep 'RuckigSmoothing::extendTrajectoryDuration'",
        "build_id": f"readelf -n {LIBRARY} | grep 'Build ID'",
        "linked_libraries": f"source /opt/ros/jazzy/setup.bash; ldd {LIBRARY}",
        "version": "grep -m1 '<version>' /opt/ros/jazzy/share/moveit_core/package.xml",
        "header_excerpt": f"grep -n -A12 -B5 'extendTrajectoryDuration' {HEADER}",
    }
    raw: dict[str, dict[str, Any]] = {}
    for name, command in commands.items():
        code, text = wsl_capture(command)
        raw[name] = {"returncode": code, "text": text.strip()}
    first_line = lambda value: value.splitlines()[0].strip() if value.splitlines() else ""
    library_line = first_line(raw["library_sha256"]["text"])
    header_line = first_line(raw["header_sha256"]["text"])
    library_hash = library_line.split()[0] if library_line else None
    header_hash = header_line.split()[0] if header_line else None
    stat_parts = first_line(raw["stat"]["text"]).split("|")
    package_parts = first_line(raw["package"]["text"]).split()
    build_id_text = raw["build_id"]["text"]
    record = {
        "schema_version": "stage3_h12_r5a_installed_moveit_provenance_v1",
        "moveit_version": "2.12.4" if "2.12.4" in raw["version"]["text"] else None,
        "library_path": stat_parts[0] if stat_parts else LIBRARY,
        "library_size": int(stat_parts[1]) if len(stat_parts) > 1 else None,
        "library_sha256": library_hash,
        "elf_build_id": first_line(build_id_text.split("Build ID:", 1)[1]) if "Build ID:" in build_id_text else None,
        "package_name": package_parts[0] if package_parts else None,
        "package_version": package_parts[1] if len(package_parts) > 1 else None,
        "package_architecture": first_line(raw["package_architecture"]["text"]),
        "binary_architecture": first_line(raw["architecture"]["text"]),
        "header_path": HEADER,
        "header_sha256": header_hash,
        "symbol_present": raw["symbol"]["returncode"] == 0 and bool(raw["symbol"]["text"]),
        "exported_symbol": first_line(raw["symbol"]["text"]),
        "linked_libraries": [line for line in raw["linked_libraries"]["text"].splitlines() if "=>" in line or "linux-vdso" in line or "ld-linux" in line],
        "binary_matches_h12_r5": library_hash == EXPECTED_LIBRARY_SHA256,
        "raw_commands": raw,
    }
    record["INSTALLED_BINARY_PROVENANCE"] = "PASSED" if all([
        record["moveit_version"] == "2.12.4",
        record["binary_matches_h12_r5"],
        record["symbol_present"],
        bool(record["header_sha256"]),
    ]) else "BLOCKED"
    write_json(output / "h12_r5a_installed_moveit_provenance.json", record)
    return record


def official_source_audit(output: Path) -> dict[str, Any]:
    files = []
    for url, filename in [(OFFICIAL_CPP_URL, "official_moveit_2_12_4_ruckig_traj_smoothing.cpp"), (OFFICIAL_HPP_URL, "official_moveit_2_12_4_ruckig_traj_smoothing.hpp")]:
        destination = output / filename
        code, text = wsl_capture(f"curl -fsSL {url}", 120)
        if code:
            raise RuntimeError(f"official_source_fetch_failed:{url}")
        destination.write_text(text, encoding="utf-8", newline="\n")
        files.append({"url": url, "path": str(destination), "sha256": sha256_file(destination)})
    cpp = (output / "official_moveit_2_12_4_ruckig_traj_smoothing.cpp").read_text(encoding="utf-8")
    hpp = (output / "official_moveit_2_12_4_ruckig_traj_smoothing.hpp").read_text(encoding="utf-8")
    return {
        "source_files": files,
        "header_documents_every_segment": "Extend the duration of every trajectory segment" in hpp and "size_t num_waypoints" in hpp,
        "cpp_calls_with_waypoint_idx": "extendTrajectoryDuration(duration_extension_factor, waypoint_idx" in cpp.replace("\n", " "),
        "cpp_mutates_waypoint_idx_plus_one": "setWayPointDurationFromPrevious(waypoint_idx + 1" in cpp.replace("\n", " "),
    }


def disassembly_audit(output: Path) -> dict[str, Any]:
    command = (
        f"objdump -d -C {LIBRARY} | sed -n "
        "'/<trajectory_processing::RuckigSmoothing::extendTrajectoryDuration/,/^$/p'"
    )
    code, raw = wsl_capture(command, 120)
    (output / "h12_r5a_extend_duration_disassembly.txt").write_text(raw, encoding="utf-8", newline="\n")
    audit = {
        "schema_version": "stage3_h12_r5a_binary_semantics_audit_v1",
        "tool": "objdump -d -C",
        "returncode": code,
        "symbol_located": "RuckigSmoothing::extendTrajectoryDuration" in raw,
        "DISASSEMBLY_SEMANTICS": "INCONCLUSIVE",
        "role": "auxiliary evidence only; black-box mutation has priority",
        "raw_path": str(output / "h12_r5a_extend_duration_disassembly.txt"),
    }
    write_json(output / "h12_r5a_binary_semantics_audit.json", audit)
    return audit


def run_probe(output: Path, timeout_s: int) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite_existing_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    write_json(output / "h12_r5a_upstream_immutability_before.json", before)
    prov = provenance(output)
    if prov["INSTALLED_BINARY_PROVENANCE"] != "PASSED":
        write_json(output / "stage3_h12_r5a_terminal_certificate.json", {
            "STAGE_3_H12_R5A": "BLOCKED", "FIRST_BLOCKER": "installed_moveit_binary_changed",
            "H12_R5_IMMUTABLE": "YES", "READY_FOR_STAGE_3_H13": "NO",
        })
        return 2
    official = official_source_audit(output)
    disassembly_audit(output)
    source_before = INTERPOSER.read_text(encoding="utf-8")
    write_json(output / "h12_r5a_interposer_patch_manifest.json", {
        "schema_version": "stage3_h12_r5a_interposer_patch_manifest_v1",
        "INTERPOSER_ARGUMENT_BEFORE": "num_waypoints",
        "INTERPOSER_ARGUMENT_AFTER": None,
        "INTERPOSER_CORRECTED": "NO",
        "source_sha256_before": sha256_file(INTERPOSER),
        "pre_patch_call_present": "extendTrajectoryDuration(duration_extension_factor, num_waypoints, num_dof" in source_before,
    })
    interposer = r5.build_interposer(output)
    shutil.copy2(R5_ROOT / "h12_r5_native_input.npz", output / "h12_r5a_native_input.npz")
    shutil.copy2(R5_ROOT / "h12_r5_primitive_manifest.jsonl", output / "h12_r5a_primitive_manifest.jsonl")
    units = load_jsonl(output / "h12_r5a_primitive_manifest.jsonl")
    probe_manifest = output / "h12_r5a_probe_manifest.jsonl"
    write_jsonl(probe_manifest, units[:1])
    os.environ["H12_R5A_HELPER_PROBE"] = "1"
    try:
        case = r5.run_case(output, output / "h12_r5a_native_input.npz", probe_manifest, interposer,
                           "installed_helper_binary_probe", mitigate=True, installed_helpers=True,
                           hard_limit_rule=True, boundary_velocity_recondition=True, timeout_s=timeout_s)
    finally:
        os.environ.pop("H12_R5A_HELPER_PROBE", None)
    raw_path = Path(case["shards"][0]["probe"]) / "h12_r5a_extend_duration_binary_probe_raw.jsonl"
    probes = load_jsonl(raw_path)
    semantics = classify_installed_helper_semantics(probes)
    probe_record = {
        "schema_version": "stage3_h12_r5a_extend_duration_binary_probe_v1",
        "installed_library_path": LIBRARY,
        "installed_library_sha256": prov["library_sha256"],
        "BINARY_SECOND_ARGUMENT_SEMANTICS": semantics,
        "probe_count": len(probes),
        "probes": probes,
        "black_box_behavior_priority": "higher_than_header_documentation",
        "probe_execution": case,
    }
    write_json(output / "h12_r5a_extend_duration_binary_probe.json", probe_record)
    reconciliation = {
        "schema_version": "stage3_h12_r5a_moveit_semantics_reconciliation_v1",
        "INSTALLED_HEADER_DOCUMENTED_SEMANTICS": "global/every-segment wording; second parameter named num_waypoints",
        "OFFICIAL_2_12_4_CPP_IMPLEMENTATION_SEMANTICS": "local failing segment via waypoint_idx; mutates waypoint_idx + 1 duration and target derivatives",
        "INSTALLED_BINARY_OBSERVED_SEMANTICS": semantics,
        "INSTALLED_BINARY_MATCHES_OFFICIAL_CPP_SEMANTICS": "YES" if semantics == "waypoint_idx" and official["cpp_calls_with_waypoint_idx"] and official["cpp_mutates_waypoint_idx_plus_one"] else "NO",
        "R5_PRIOR_SEMANTICS_CLAIM_STATUS": "REFUTED" if semantics == "waypoint_idx" else "UNPROVEN",
        "official_source_audit": official,
    }
    write_json(output / "h12_r5a_moveit_semantics_reconciliation.json", reconciliation)
    if semantics == "UNKNOWN":
        write_json(output / "stage3_h12_r5a_terminal_certificate.json", {
            "STAGE_3_H12_R5A": "BLOCKED", "FIRST_BLOCKER": "installed_helper_semantics_unproven",
            "H12_R5_IMMUTABLE": "YES", "READY_FOR_STAGE_3_H13": "NO",
        })
        return 2
    print(f"BINARY_SECOND_ARGUMENT_SEMANTICS: {semantics}")
    print(f"OUTPUT_DIRECTORY: {output}")
    return 0


def run_tests(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    commands = [
        ("focused", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r5a.py"]),
        ("regression", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r5.py", "tests/test_stage3_h12_r4.py", "tests/test_stage3_h12_r3.py", "tests/test_stage3_h12_r2_native_contract.py", "tests/test_stage3_h12_r2_manifold.py", "tests/test_stage3_h12_r.py", "tests/test_stage3_h12_no_leakage.py"]),
    ]
    reports = []
    for name, command in commands:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        text = (process.stdout or "") + (process.stderr or "")
        (output / f"{name}_tests.txt").write_text(text, encoding="utf-8", newline="\n")
        summary = next((line for line in reversed(text.splitlines()) if "passed" in line or "failed" in line), "no pytest summary")
        reports.append({"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode, "summary": summary})
    write_json(output / "h12_r5a_test_results.json", {"FOCUSED_TESTS": reports[0], "REGRESSION_TESTS": reports[1]})
    return reports[0], reports[1]


def collect_extension_rows(run: Mapping[str, Any], units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in run["shards"]:
        start = int(shard["shard_id"]) * 60
        raw_rows = load_jsonl(Path(shard["probe"]) / "h12_r5a_extension_calls_raw.jsonl")
        for raw in raw_rows:
            invocation = int(raw["trajectory_invocation_index"])
            unit = units[start + invocation]
            row = {**raw, "unit_id": str(unit["unit_id"]), "primitive_id": str(unit["primitive_id"])}
            passed, errors = validate_extension_mutation(row)
            row["helper_mutation_invariant"] = "PASSED" if passed else "FAILED"
            row["helper_mutation_invariant_errors"] = errors
            rows.append(row)
    return rows


def max_observed_overshoot(run: Mapping[str, Any]) -> float:
    maximum = 0.0
    for shard in run["shards"]:
        for row in load_jsonl(Path(shard["probe"]) / "stage25r2_overshoot_events.jsonl"):
            point = row.get("worst_overshoot_in_call")
            if point:
                maximum = max(maximum, float(point.get("abs_overshoot_rad", 0.0)))
    return maximum


def native_output_hash(run: Mapping[str, Any]) -> str:
    rows = load_jsonl(Path(run["result_path"]))
    projection = []
    for row in rows:
        ruckig = dict(row.get("ruckig") or {})
        ruckig.pop("native_compute_time", None)
        projection.append({
            "unit_id": row.get("unit_id"), "ruckig": ruckig,
            "output_duration": row.get("output_duration"), "output_waypoint_count": row.get("output_waypoint_count"),
            "post_ruckig_endpoint_delta_rad": row.get("post_ruckig_endpoint_delta_rad"),
        })
    return semantic_sha256(projection)


def report_text(terminal: Mapping[str, Any], output: Path) -> str:
    ordered = [
        "STAGE_3_H12_R5A", "FIRST_BLOCKER", "STAGE_3_H12", "READY_FOR_STAGE_3_H13", "H12_R5_IMMUTABLE",
        "MOVEIT_VERSION", "MOVEIT_BINARY_SHA256", "BINARY_MATCHES_H12_R5",
        "INSTALLED_HEADER_DOCUMENTED_SEMANTICS", "OFFICIAL_2_12_4_CPP_IMPLEMENTATION_SEMANTICS", "INSTALLED_BINARY_OBSERVED_SEMANTICS",
        "R5_PRIOR_SEMANTICS_CLAIM_STATUS", "INTERPOSER_ARGUMENT_BEFORE", "INTERPOSER_ARGUMENT_AFTER", "INTERPOSER_CORRECTED",
        "FORMAL_INSTALLED_EXTENSION_CALLS", "FORMAL_PRIMITIVES_WITH_EXTENSION_CALLS", "FORMAL_EXTENSION_MUTATION_INVARIANT_FAILURES",
        "PRIMITIVES", "TOTG_FINAL_PASS", "RUCKIG_SMOOTHING_SUCCESS", "RUCKIG_DURATION_CEILING_HIT", "RUCKIG_NATIVE_ERRORS",
        "POST_RUCKIG_VALIDATED", "POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS",
        "SPRAY_PROCESS_VIOLATIONS", "COLLISION_VIOLATIONS", "R5_ACCELERATION_VIOLATIONS", "R5A_ACCELERATION_VIOLATIONS",
        "R5_SPRAY_PROCESS_VIOLATIONS", "R5A_SPRAY_PROCESS_VIOLATIONS", "FINAL_CERTIFIED_NEURAL_RMSE", "FINAL_GAIN_VS_CV",
        "REPLAY", "FOCUSED_TESTS", "REGRESSION_TESTS", "UPSTREAM_IMMUTABLE", "PHYSICAL_ROBOT_CONNECTED", "FJT_GOALS_SENT", "PHYSICAL_MOTION",
        "SOURCE_H12_R5A_DIRECTORY", "NEXT_RECOMMENDED_STAGE",
    ]
    lines = ["# Stage 3 H12-R5A - MoveIt 2.12.4 Helper Semantics Closure", "", "```text"]
    for key in ordered:
        lines.append(f"{key}: {terminal.get(key)}")
    lines.extend([
        "```", "",
        "The installed binary's second argument is an index: the black-box probes changed only `second_argument + 1`, including the target duration and recomputed target derivatives.", "",
        "H12-R5 passed `num_waypoints` where the installed implementation expects `waypoint_idx`; the interposer now passes the proven index and fails closed on any unexpected mutation.", "",
        "The fresh formal runs determine whether the 40 acceleration violations and 2,304 spray-process violations remain; no H12-R6 remediation was attempted here.", "",
        "H12-R5A closes only the helper-semantics defect. H12 remains blocked whenever acceleration or spray-process violations remain, so H13 is not authorized.", "",
        f"Evidence directory: `{output}`", "",
    ])
    return "\n".join(lines)


def run_formal(output: Path, timeout_s: int) -> int:
    probe = load_json(output / "h12_r5a_extend_duration_binary_probe.json")
    reconciliation = load_json(output / "h12_r5a_moveit_semantics_reconciliation.json")
    prov = provenance(output)
    if probe["BINARY_SECOND_ARGUMENT_SEMANTICS"] != "waypoint_idx":
        raise RuntimeError("installed_helper_semantics_not_waypoint_idx")
    source = INTERPOSER.read_text(encoding="utf-8")
    corrected = "extendTrajectoryDuration(duration_extension_factor, waypoint_idx, num_dof" in source
    instrumented = "second_argument_passed" in source and "h12_r5a_extension_calls_raw.jsonl" in source
    if not corrected or not instrumented:
        raise RuntimeError("interposer_not_corrected_or_instrumented")
    manifest = load_json(output / "h12_r5a_interposer_patch_manifest.json")
    manifest.update({
        "INTERPOSER_ARGUMENT_AFTER": "waypoint_idx", "INTERPOSER_CORRECTED": "YES",
        "source_sha256_after": sha256_file(INTERPOSER), "mutation_invariant_instrumented": instrumented,
        "correction_authorized_by_binary_probe": True,
    })
    write_json(output / "h12_r5a_interposer_patch_manifest.json", manifest)
    focused, regression = run_tests(output)
    interposer = r5.build_interposer(output)
    input_path = output / "h12_r5a_native_input.npz"
    manifest_path = output / "h12_r5a_primitive_manifest.jsonl"
    units = load_jsonl(manifest_path)
    formal_runs = [
        r5.run_case(output, input_path, manifest_path, interposer, "formal" if index == 0 else f"formal_replay_{index}",
                    mitigate=True, installed_helpers=True, hard_limit_rule=True,
                    boundary_velocity_recondition=True, timeout_s=timeout_s)
        for index in range(3)
    ]
    counts = r5.case_counts(formal_runs[0])
    rows = counts.pop("rows")
    shutil.copy2(Path(formal_runs[0]["result_path"]), output / "h12_r5a_ruckig_results.jsonl")
    all_extension_rows = [collect_extension_rows(run, units) for run in formal_runs]
    write_jsonl(output / "h12_r5a_extension_calls.jsonl", all_extension_rows[0])
    summary = extension_summary(all_extension_rows[0])
    write_json(output / "h12_r5a_extension_summary.json", summary)
    invariant_failures = [row for row in all_extension_rows[0] if row["helper_mutation_invariant"] != "PASSED"]
    write_json(output / "h12_r5a_helper_mutation_invariants.json", {
        "HELPER_MUTATION_INVARIANT": "PASSED" if not invariant_failures else "FAILED",
        "HELPER_MUTATION_INVARIANT_FAILURES": len(invariant_failures), "failures": invariant_failures,
    })
    write_json(output / "h12_r5a_totg_recertification.json", {"TOTG_FINAL_PASS": counts["TOTG_PASS"], "TOTG_FINAL_FAIL": TOTAL_PRIMITIVES - counts["TOTG_PASS"]})
    write_json(output / "h12_r5a_ruckig_final_certification.json", {key: counts[key] for key in ("RUCKIG_ATTEMPTED", "RUCKIG_RAW_WORKING", "RUCKIG_RAW_FINISHED", "RUCKIG_SMOOTHING_SUCCESS", "RUCKIG_DURATION_CEILING_HIT", "RUCKIG_NATIVE_ERRORS")})
    write_json(output / "h12_r5a_post_ruckig_dynamic_validation.json", {
        "POST_RUCKIG_VALIDATED": counts["validated_windows"], "POSITION_VIOLATIONS": counts["position_violations"],
        "VELOCITY_VIOLATIONS": counts["velocity_violations"], "ACCELERATION_VIOLATIONS": counts["acceleration_violations"],
        "JERK_VIOLATIONS": counts["jerk_violations"],
    })
    write_json(output / "h12_r5a_spray_process_validation.json", spray_validation_schema(counts["validated_windows"], counts["process_violations"]))
    write_json(output / "h12_r5a_collision_validation.json", {
        "COVERAGE_STATUS": "COMPLETE" if counts["validated_windows"] == TOTAL_WINDOWS else "INCOMPLETE",
        "VALIDATION_RESULT": "PASSED" if counts["validated_windows"] == TOTAL_WINDOWS and counts["collision_violations"] == 0 else "FAILED",
        "COLLISION_VIOLATIONS": counts["collision_violations"], "COLLISION_METHOD": "adaptive_discrete_interpolation",
        "CCD": "not_available", "CLEARANCE": None,
    })
    final_rmse = r5.POST_REPAIR_RMSE if counts["validated_windows"] == TOTAL_WINDOWS and all(counts[key] == 0 for key in ("position_violations", "velocity_violations", "acceleration_violations", "jerk_violations", "process_violations", "collision_violations")) else None
    metrics = {
        "CV_RMSE": r5.CV_RMSE, "RAW_NEURAL_RMSE": r5.RAW_RMSE, "POST_REPAIR_NEURAL_RMSE": r5.POST_REPAIR_RMSE,
        "FINAL_CERTIFIED_NEURAL_RMSE": final_rmse,
        "FINAL_GAIN_VS_CV": None if final_rmse is None else (r5.CV_RMSE - final_rmse) / r5.CV_RMSE,
    }
    write_json(output / "h12_r5a_neural_metric_certification.json", metrics)
    hashes = []
    for index, run in enumerate(formal_runs):
        ext_rows = all_extension_rows[index]
        decisions = []
        for shard in run["shards"]:
            decisions.extend(load_jsonl(Path(shard["probe"]) / "stage3_h12_r5_remediation_events.jsonl"))
        run_counts = r5.case_counts(run)
        certification = {key: value for key, value in run_counts.items() if key != "rows"}
        hashes.append({
            "trajectory_semantic_hash": run["semantic_sha256"],
            "ruckig_native_output_hash": native_output_hash(run),
            "extension_call_trace_hash": semantic_sha256(ext_rows),
            "helper_mutation_trace_hash": semantic_sha256([{"unit_id": row["unit_id"], "helper_mutation_invariant": row["helper_mutation_invariant"], "errors": row["helper_mutation_invariant_errors"]} for row in ext_rows]),
            "boundary_reconditioning_decision_hash": semantic_sha256(decisions),
            "post_validation_summary_hash": semantic_sha256(certification),
        })
    replay_ok = all(run["exact_result_coverage"] for run in formal_runs) and all(len({entry[key] for entry in hashes}) == 1 for key in hashes[0])
    write_json(output / "h12_r5a_replay_manifest.json", {"REPLAY": "3/3" if replay_ok else "0/3", "hashes": hashes, "runs": formal_runs})
    r5_terminal = load_json(R5_ROOT / "stage3_h12_r5_terminal_certificate.json")
    r5_replay = load_json(R5_ROOT / "h12_r5_replay_manifest.json")
    r5_semantic_hash = (r5_replay.get("repaired_trajectory_and_ruckig_output_semantic_hashes") or [None])[0]
    r5a_max_overshoot = max_observed_overshoot(formal_runs[0])
    comparison = {
        "R5_RUCKIG_SMOOTHING_SUCCESS": r5_terminal["RUCKIG_SMOOTHING_SUCCESS"], "R5A_RUCKIG_SMOOTHING_SUCCESS": counts["RUCKIG_SMOOTHING_SUCCESS"],
        "R5_DURATION_CEILING_HIT": r5_terminal["RUCKIG_DURATION_CEILING_HIT"], "R5A_DURATION_CEILING_HIT": counts["RUCKIG_DURATION_CEILING_HIT"],
        "R5_NATIVE_ERRORS": r5_terminal["RUCKIG_NATIVE_ERRORS"], "R5A_NATIVE_ERRORS": counts["RUCKIG_NATIVE_ERRORS"],
        "R5_POSITION_VIOLATIONS": r5_terminal["POSITION_VIOLATIONS"], "R5A_POSITION_VIOLATIONS": counts["position_violations"],
        "R5_VELOCITY_VIOLATIONS": r5_terminal["VELOCITY_VIOLATIONS"], "R5A_VELOCITY_VIOLATIONS": counts["velocity_violations"],
        "R5_ACCELERATION_VIOLATIONS": r5_terminal["ACCELERATION_VIOLATIONS"], "R5A_ACCELERATION_VIOLATIONS": counts["acceleration_violations"],
        "R5_JERK_VIOLATIONS": r5_terminal["JERK_VIOLATIONS"], "R5A_JERK_VIOLATIONS": counts["jerk_violations"],
        "R5_SPRAY_PROCESS_VIOLATIONS": r5_terminal["SPRAY_PROCESS_VIOLATIONS"], "R5A_SPRAY_PROCESS_VIOLATIONS": counts["process_violations"],
        "R5_COLLISION_VIOLATIONS": r5_terminal["COLLISION_VIOLATIONS"], "R5A_COLLISION_VIOLATIONS": counts["collision_violations"],
        "R5_MAX_OVERSHOOT": r5_terminal["MAX_OVERSHOOT_AFTER"], "R5A_MAX_OVERSHOOT": r5a_max_overshoot,
        "R5_FORMAL_SEMANTIC_HASH": r5_semantic_hash, "R5A_FORMAL_SEMANTIC_HASH": formal_runs[0]["semantic_sha256"],
        **summary,
    }
    stable_blockers = counts["acceleration_violations"] == r5_terminal["ACCELERATION_VIOLATIONS"] and counts["process_violations"] == r5_terminal["SPRAY_PROCESS_VIOLATIONS"]
    comparison["MOVEIT_HELPER_ARGUMENT_SEMANTICS_NOT_ROOT_CAUSE_OF_CURRENT_FIRST_BLOCKER"] = "SUPPORTED" if stable_blockers else "NOT_SUPPORTED"
    comparison["R5_ACCEL_SPRAY_CONTAMINATED_BY_EXTENSION_ARGUMENT_BUG"] = "NO_DIRECT_EXECUTION_PATH" if summary["FORMAL_INSTALLED_EXTENSION_CALLS"] == 0 else "EXECUTION_PATH_PRESENT_AND_AUDITED"
    write_json(output / "h12_r5a_r5_vs_r5a_comparison.json", comparison)
    after = snapshot()
    before = load_json(output / "h12_r5a_upstream_immutability_before.json")
    immutable = before == after and all(row["exists"] for row in before.values())
    h12_r5_immutable = immutable and all(row["exists"] for path, row in before.items() if row["h12_r5_evidence"])
    write_json(output / "h12_r5a_upstream_immutability.json", {
        "UPSTREAM_IMMUTABLE": "YES" if immutable else "NO", "H12_R5_IMMUTABLE": "YES" if h12_r5_immutable else "NO",
        "before": before, "after": after,
    })
    gate_input = {
        "H12_R5_IMMUTABLE": "YES" if h12_r5_immutable else "NO", "INSTALLED_BINARY_PROVENANCE": prov["INSTALLED_BINARY_PROVENANCE"],
        "INSTALLED_BINARY_OBSERVED_SEMANTICS": probe["BINARY_SECOND_ARGUMENT_SEMANTICS"],
        "INTERPOSER_SECOND_ARGUMENT_MATCHES_PROVEN_BINARY_SEMANTICS": "YES" if corrected else "NO",
        "HELPER_MUTATION_INVARIANT_FAILURES": len(invariant_failures), "FORMAL_PRIMITIVES": len(rows),
        "POST_RUCKIG_VALIDATED": counts["validated_windows"], "REPLAY": "3/3" if replay_ok else "0/3",
        "UPSTREAM_IMMUTABLE": "YES" if immutable else "NO", "FOCUSED_TESTS": focused["status"], "REGRESSION_TESTS": regression["status"],
        "PHYSICAL_ROBOT_CONNECTED": "NO", "FJT_GOALS_SENT": 0, "PHYSICAL_MOTION": 0,
    }
    passed, blocker = semantic_closure_gate(gate_input)
    h12_blocker = "acceleration_violation" if counts["acceleration_violations"] else ("spray_process_violation" if counts["process_violations"] else "none")
    terminal = {
        "schema_version": "stage3_h12_r5a_terminal_certificate_v1", "STAGE_3_H12_R5A": "PASSED" if passed else "BLOCKED",
        "FIRST_BLOCKER": blocker, "STAGE_3_H12": "BLOCKED" if h12_blocker != "none" else "PASSED", "H12_FIRST_BLOCKER": h12_blocker,
        "READY_FOR_STAGE_3_H13": "NO" if h12_blocker != "none" else "YES", **gate_input,
        "MOVEIT_VERSION": prov["moveit_version"], "MOVEIT_BINARY_SHA256": prov["library_sha256"], "BINARY_MATCHES_H12_R5": "YES" if prov["binary_matches_h12_r5"] else "NO",
        "INSTALLED_HEADER_DOCUMENTED_SEMANTICS": reconciliation["INSTALLED_HEADER_DOCUMENTED_SEMANTICS"],
        "OFFICIAL_2_12_4_CPP_IMPLEMENTATION_SEMANTICS": reconciliation["OFFICIAL_2_12_4_CPP_IMPLEMENTATION_SEMANTICS"],
        "R5_PRIOR_SEMANTICS_CLAIM_STATUS": reconciliation["R5_PRIOR_SEMANTICS_CLAIM_STATUS"],
        "INTERPOSER_ARGUMENT_BEFORE": "num_waypoints", "INTERPOSER_ARGUMENT_AFTER": "waypoint_idx", "INTERPOSER_CORRECTED": "YES",
        **summary, "PRIMITIVES": f"{len(rows)}/480", "TOTG_FINAL_PASS": f"{counts['TOTG_PASS']}/480",
        "RUCKIG_SMOOTHING_SUCCESS": f"{counts['RUCKIG_SMOOTHING_SUCCESS']}/480", "RUCKIG_DURATION_CEILING_HIT": counts["RUCKIG_DURATION_CEILING_HIT"],
        "RUCKIG_NATIVE_ERRORS": counts["RUCKIG_NATIVE_ERRORS"], "POST_RUCKIG_VALIDATED": f"{counts['validated_windows']}/189216",
        "POSITION_VIOLATIONS": counts["position_violations"], "VELOCITY_VIOLATIONS": counts["velocity_violations"],
        "ACCELERATION_VIOLATIONS": counts["acceleration_violations"], "JERK_VIOLATIONS": counts["jerk_violations"],
        "SPRAY_PROCESS_VIOLATIONS": counts["process_violations"], "COLLISION_VIOLATIONS": counts["collision_violations"],
        "R5_ACCELERATION_VIOLATIONS": r5_terminal["ACCELERATION_VIOLATIONS"], "R5A_ACCELERATION_VIOLATIONS": counts["acceleration_violations"],
        "R5_SPRAY_PROCESS_VIOLATIONS": r5_terminal["SPRAY_PROCESS_VIOLATIONS"], "R5A_SPRAY_PROCESS_VIOLATIONS": counts["process_violations"],
        **metrics, "FOCUSED_TESTS_DETAIL": focused["summary"], "REGRESSION_TESTS_DETAIL": regression["summary"],
        "SOURCE_H12_R5A_DIRECTORY": str(output), "NEXT_RECOMMENDED_STAGE": "H12-R6" if h12_blocker != "none" else "none",
    }
    write_json(output / "stage3_h12_r5a_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(report_text(terminal, output), encoding="utf-8", newline="\n")
    print(report_text(terminal, output))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("probe", "formal"), required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--native-timeout", type=int, default=1800)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "outputs" / f"stage3_h12_r5a_moveit_helper_semantics_closure_{now_utc()}"
    return run_probe(output, args.native_timeout) if args.phase == "probe" else run_formal(output, args.native_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
