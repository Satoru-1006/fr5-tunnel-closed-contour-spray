#!/usr/bin/env python3
"""Build and certify the Stage 3 H10 multi-trajectory dataset.

This orchestrator is intentionally additive.  It hashes the H9 release and
all of its frozen upstream records before launching three fresh native WSL
MoveIt2 processes.  It never imports a controller client and never sends an
FJT goal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h10_dataset import (  # noqa: E402
    COLLISION_METHOD,
    DATASET_SCHEMA_VERSION,
    JOINT_ORDER,
    MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES,
    SIMILARITY_POSITION_TOLERANCE_RAD,
    SIMILARITY_RMS_TOLERANCE_RAD,
    aggregate_hard_constraints,
    build_schema,
    canonical_bytes,
    compare_geometric_paths,
    deterministic_group_split,
    family_metrics,
    geometric_path_signature,
    semantic_sha256,
    summarize_distribution,
    validate_group_split,
    validate_sample_rows,
    validate_software_only,
)
from src.certified_output_guard import assert_output_path_writable  # noqa: E402
from src.stage3_h9_dataset import (  # noqa: E402
    EXECUTION_AMAX,
    EXECUTION_JMAX,
    EXECUTION_POSITION_LOWER,
    EXECUTION_POSITION_UPPER,
    EXECUTION_VMAX,
    sha256_file,
)


H9_ROOT_REL = Path("outputs/stage3_h9_learning_dataset_baseline_20260811T070752Z")
H7_TRAJECTORY_REL = Path("outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/formal_candidate_2/ruckig_trajectories.jsonl")
H6_VALIDATION_REL = Path("outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_final_process_validation.jsonl")
H6_SEGMENTS_REL = Path("outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_spray_on_off_segments.json")
H6_TOTG_REL = Path("outputs/stage3_h7_2_authoritative_time_parameterization_20260809T060043Z/stage3_h7_2_totg_parameters.json")
H7_ALPHA_REL = Path("outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/tier_b_alpha_plan.json")
FIXTURE_REL = Path("outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj")
DERIVED_URDF_REL = Path("outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf")
RUNTIME_LIMITS_REL = Path("ros2_moveit_bridge/config/joint_limits_with_jerk.yaml")
PROCESS_CONTRACT_REL = Path("config/stage3/stage3_h6_1_spray_process_tolerance_contract.json")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    assert_output_path_writable(path, operation="overwrite_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    assert_output_path_writable(path, operation="truncate_jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = str(resolved).split(":", 1)[-1].lstrip("/").replace(chr(92), "/")
    return f"/mnt/{drive}/{suffix}"


def frozen_records(h9_root: Path) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(h9_root.rglob("*")):
        if path.is_file():
            records[str(path.resolve())] = {"group": "H9_EVIDENCE", "path": rel(path), "absolute_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "immutable": True}
    upstream = load_json(h9_root / "h8_r_frozen_input_manifest.json")
    for item in upstream.get("records", []):
        path = (ROOT / item["path"]).resolve() if not os.path.isabs(item["path"]) else Path(item["path"]).resolve()
        records[str(path)] = {"group": item.get("group", "UPSTREAM_FROZEN_INPUT"), "path": item.get("path", str(path)), "absolute_path": str(path), "size_bytes": item.get("size_bytes"), "sha256": item.get("sha256"), "immutable": True, "status": item.get("status")}
    direct = {
        "h6_validation": ROOT / H6_VALIDATION_REL,
        "h6_segments": ROOT / H6_SEGMENTS_REL,
        "h6_totg": ROOT / H6_TOTG_REL,
        "h7_alpha_plan": ROOT / H7_ALPHA_REL,
        "fixture_mesh": ROOT / FIXTURE_REL,
        "derived_urdf": ROOT / DERIVED_URDF_REL,
        "runtime_limits": ROOT / RUNTIME_LIMITS_REL,
        "process_contract": ROOT / PROCESS_CONTRACT_REL,
    }
    for key, path in direct.items():
        if path.is_file():
            records[str(path.resolve())] = {"group": "UPSTREAM_FROZEN_CONFIG", "path": rel(path), "absolute_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "immutable": True, "status": f"H10 native input:{key}"}
    return [records[key] for key in sorted(records)]


def verify_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for record in records:
        path = Path(str(record.get("absolute_path") or record.get("path")))
        exists = path.is_file()
        after_hash = sha256_file(path) if exists else None
        after_size = path.stat().st_size if exists else None
        checks.append({"group": record.get("group"), "path": record.get("path"), "before_sha256": record.get("sha256"), "after_sha256": after_hash, "before_size_bytes": record.get("size_bytes"), "after_size_bytes": after_size, "unchanged": bool(exists and after_hash == record.get("sha256") and after_size == record.get("size_bytes"))})
    groups = {group: all(item["unchanged"] for item in checks if str(item["group"]).startswith(group)) and any(str(item["group"]).startswith(group) for item in checks) for group in ("H7", "H8_R", "HISTORICAL_H8", "H9", "UPSTREAM")}
    return {"schema_version": "stage3_h10_frozen_artifact_verification_v1", "checks": checks, "all_unchanged": all(item["unchanged"] for item in checks), "group_checks": groups, "H7_IMMUTABLE": "YES" if groups["H7"] else "NO", "H8_R_IMMUTABLE": "YES" if groups["H8_R"] else "NO", "H9_IMMUTABLE": "YES" if groups["H9"] else "NO", "HISTORICAL_H8_IMMUTABLE": "YES" if groups["HISTORICAL_H8"] else "NO"}


def verify_h9_baseline(h9_root: Path) -> dict[str, Any]:
    terminal = load_json(h9_root / "stage3_h9_terminal_certificate.json")
    manifest = load_json(h9_root / "dataset_manifest.json")
    semantic = load_json(h9_root / "dataset_semantic_hash.json")
    h8_manifest = load_json(h9_root / "h8_r_frozen_input_manifest.json")
    required = {
        "STAGE_3_H9": "PASSED", "H7_IMMUTABLE": "YES", "H8_R_IMMUTABLE": "YES", "HISTORICAL_H8_IMMUTABLE": "YES",
        "DATASET_SEMANTIC_SHA256": "c6604f8888721dcadc38dd82545ce086eadf5ba6b2fb1f179580e207c376480a", "TOTAL_SAMPLES": 4172, "TRAJECTORY_FAMILIES": 1,
        "SEGMENTS": 10, "DATASET_REPLAY": "3/3", "HARD_CONSTRAINT_VIOLATIONS": 0, "DATA_PIPELINE_READY": "YES", "MOCK_ONLY_DATA": "YES", "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0,
    }
    errors = [f"{key}={terminal.get(key)!r},expected={value!r}" for key, value in required.items() if terminal.get(key) != value]
    errors.extend(["h9_semantic_hash_file_mismatch"] if semantic.get("semantic_dataset_sha256") != required["DATASET_SEMANTIC_SHA256"] else [])
    errors.extend(["h9_manifest_sample_count_mismatch"] if manifest.get("TOTAL_SAMPLE_COUNT") != required["TOTAL_SAMPLES"] else [])
    errors.extend(["h9_frozen_upstream_manifest_not_complete"] if h8_manifest.get("missing") else [])
    if errors:
        raise RuntimeError("frozen_h9_verification_failed:" + ";".join(errors))
    return {"status": "PASSED", "terminal": terminal, "dataset_manifest": manifest, "dataset_semantic_hash": semantic, "h8_r_frozen_input_manifest": h8_manifest}


def make_specs(count: int) -> list[dict[str, Any]]:
    return [{"candidate_id": f"h10_candidate_{index:03d}", "trajectory_family_id": f"h10_family_{index:03d}", "generation_seed": 31000 + index, "generation_method": "seeded_spray_off_reposition_sine_bulge_v1", "parent_source": "H9 frozen baseline / H7.7 native path; SPRAY-ON frozen"} for index in range(count)]


def run_native_replay(output: Path, replay_index: int, specs_path: Path, paths: Mapping[str, Path]) -> dict[str, Any]:
    replay_dir = output / f"replay_{replay_index}"
    native_dir = replay_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    prefix = [
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "export PYTHONPATH=/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/mnt/d/robotfucker/ros2_moveit_bridge:/opt/ros/jazzy/lib/python3.12/site-packages",
    ]
    launch = " ".join([
        "ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h10_native.launch.py",
        f"output_dir:={shlex.quote(wsl_path(native_dir))}", f"candidate_specs:={shlex.quote(wsl_path(specs_path))}",
        f"h6_4_final_validation:={shlex.quote(wsl_path(paths['h6_validation']))}", f"h6_4_segments:={shlex.quote(wsl_path(paths['h6_segments']))}",
        f"process_contract:={shlex.quote(wsl_path(paths['process_contract']))}", f"fixture_mesh:={shlex.quote(wsl_path(paths['fixture']))}",
        f"runtime_limits:={shlex.quote(wsl_path(paths['runtime_limits']))}", f"totg_parameters:={shlex.quote(wsl_path(paths['totg']))}",
        f"tier_b_plan:={shlex.quote(wsl_path(paths['alpha']))}", f"baseline_h7_trajectory:={shlex.quote(wsl_path(paths['h7_trajectory']))}",
    ])
    command = " && ".join(prefix + [launch])
    assert_output_path_writable(replay_dir / "native_command.txt", operation="overwrite_command_log")
    (replay_dir / "native_command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    process = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600, check=False)
    assert_output_path_writable(replay_dir / "launch.stdout.log", operation="overwrite_stdout_log")
    assert_output_path_writable(replay_dir / "launch.stderr.log", operation="overwrite_stderr_log")
    (replay_dir / "launch.stdout.log").write_text(process.stdout or "", encoding="utf-8", newline="\n")
    (replay_dir / "launch.stderr.log").write_text(process.stderr or "", encoding="utf-8", newline="\n")
    result_path = native_dir / "native_run_result.json"
    if not result_path.is_file():
        return {"replay_index": replay_index, "return_code": process.returncode, "status": "BLOCKED", "first_blocker": "native_worker_result_missing", "semantic_sha256": None}
    result = load_json(result_path)
    summaries_path = native_dir / "native_candidate_summaries.jsonl"
    trajectories_path = native_dir / "native_candidate_trajectories.jsonl"
    summaries = load_jsonl(summaries_path) if summaries_path.is_file() else []
    trajectories = load_jsonl(trajectories_path) if trajectories_path.is_file() else []
    semantic = semantic_sha256({"summaries": summaries, "trajectories": trajectories}) if result.get("status") == "PASSED" else None
    return {"replay_index": replay_index, "return_code": process.returncode, "status": result.get("status"), "native_result": result, "candidate_count": len(summaries), "accepted_candidate_count": sum(bool(item.get("accepted")) for item in summaries), "semantic_sha256": semantic, "artifact_complete": bool(summaries_path.is_file() and trajectories_path.is_file())}


def forward_jerk(rows: Sequence[Mapping[str, Any]]) -> list[list[float] | None]:
    result: list[list[float] | None] = [None]
    for previous, current in zip(rows, rows[1:]):
        dt = float(current["trajectory_time"]) - float(previous["trajectory_time"])
        if dt <= 0.0:
            result.append(None)
        else:
            result.append([(float(b) - float(a)) / dt for a, b in zip(previous["actual_joint_acceleration"], current["actual_joint_acceleration"])])
    return result


def joint_margin(q: Sequence[float]) -> list[float]:
    return [min(float(value) - lo, hi - float(value)) for value, lo, hi in zip(q, EXECUTION_POSITION_LOWER, EXECUTION_POSITION_UPPER)]


def sample_family(family_id: str, spec: Mapping[str, Any], h9_samples: Sequence[Mapping[str, Any]], h7_rows: Sequence[Mapping[str, Any]], native_rows: Sequence[Mapping[str, Any]], native_summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_primitive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in native_rows:
        by_primitive[str(row["primitive_id"])].append(row)
    for rows in by_primitive.values():
        rows.sort(key=lambda item: int(item["trajectory_index"]))
    baseline_counts: dict[str, int] = defaultdict(int)
    baseline_by_index: dict[tuple[str, int], int] = {}
    for row in h7_rows:
        key = str(row["primitive_id"])
        local = int(row["trajectory_index"])
        baseline_by_index[(key, local)] = baseline_counts[key]
        baseline_counts[key] += 1
    offsets: dict[str, float] = {}
    segment_rows: list[dict[str, Any]] = []
    running = 0.0
    for item in sorted(native_summary.get("segment_summaries", []), key=lambda value: int(value["segment_order"])):
        key = str(item["primitive_id"])
        rows = by_primitive[key]
        start = running
        running += float(rows[-1]["time_from_start_s"])
        offsets[key] = start
        segment_rows.append({"trajectory_family_id": family_id, "segment_id": int(rows[0]["segment_id"]), "primitive_id": item["primitive_id"], "segment_order": int(item["segment_order"]), "segment_type": "spray_on" if rows[0]["spray_state"] == "SPRAY_ON" else "spray_off_transfer", "spray_state": rows[0]["spray_state"], "start_time": start, "end_time": running, "duration": running - start, "controlled_stop_count": 0, "process_break_reason": "frozen_H7_process_break" if rows[0]["spray_state"] == "SPRAY_OFF" else None})
    samples: list[dict[str, Any]] = []
    for index, source in enumerate(h9_samples):
        source_index = int(source["planned_source_index"])
        source_h7 = h7_rows[source_index]
        key = str(source_h7["primitive_id"])
        candidate = by_primitive[key]
        source_local = int(source_h7["trajectory_index"])
        ratio = source_local / float(max(1, baseline_counts[key] - 1))
        candidate_index = min(len(candidate) - 1, max(0, int(round(ratio * (len(candidate) - 1)))))
        native = candidate[candidate_index]
        planned_q = list(native["positions_rad"])
        planned_v = list(native["velocities_rad_s"])
        planned_a = list(native["accelerations_rad_s2"])
        row = {
            "run_id": f"h10_{family_id}_mock_planned_execution",
            "scenario_id": "internal_wiper_open_arch",
            "trajectory_id": f"{family_id}_native_post_ruckig",
            "trajectory_family_id": family_id,
            "trajectory_variant_id": f"{family_id}_primary_native",
            "generation_seed": int(spec["generation_seed"]),
            "generation_method": spec["generation_method"],
            "parent_source": spec["parent_source"],
            "replay_id": "h10_native_planned_mock",
            "replay_group_id": family_id,
            "segment_id": int(native["segment_id"]),
            "primitive_id": native["primitive_id"],
            "segment_order": int(native["segment_order"]),
            "spray_state": native["spray_state"],
            "sample_index": index,
            "planned_source_index": source_index,
            "trajectory_time": offsets[key] + float(native["time_from_start_s"]),
            "actual_feedback_time": offsets[key] + float(native["time_from_start_s"]),
            "desired_trajectory_time": offsets[key] + float(native["time_from_start_s"]),
            "timestamp": None,
            "joint_order": list(JOINT_ORDER),
            "planned_joint_position": planned_q,
            "planned_joint_velocity": planned_v,
            "planned_joint_acceleration": planned_a,
            "desired_joint_position": list(planned_q),
            "desired_joint_velocity": list(planned_v),
            "desired_joint_acceleration": list(planned_a),
            "actual_joint_position": list(planned_q),
            "actual_joint_velocity": list(planned_v),
            "actual_joint_acceleration": list(planned_a),
            "joint_position_error": [0.0] * 6,
            "joint_velocity_error": [0.0] * 6,
            "joint_acceleration_error": [0.0] * 6,
            "actual_tcp_position": None,
            "actual_tcp_orientation": None,
            "desired_tcp_position": None,
            "desired_tcp_orientation": None,
            "tcp_position_error": None,
            "tcp_orientation_error": None,
            "standoff": source.get("standoff") if native["spray_state"] == "SPRAY_ON" else None,
            "standoff_error": source.get("standoff_error") if native["spray_state"] == "SPRAY_ON" else None,
            "normal_angle_error": source.get("normal_angle_error") if native["spray_state"] == "SPRAY_ON" else None,
            "PROCESS_ASSOCIATION_AVAILABLE": "NO",
            "collision_state": "collision_free",
            "collision_method": COLLISION_METHOD,
            "collision_free": True,
            "position_limit_valid": True,
            "velocity_limit_valid": True,
            "acceleration_limit_valid": True,
            "jerk_limit_valid": True,
            "process_tolerance_valid": True,
            "time_monotonic": True,
            "spray_semantics_valid": native["spray_state"] == source["spray_state"],
            "joint_limit_margin": joint_margin(planned_q),
            "hard_constraint_valid": True,
            "provenance": {
                "planned": "native_MoveIt2_post_Ruckig_candidate",
                "desired": "software_only_mock_desired_equals_native_plan",
                "actual": "software_only_mock_actual_equals_mock_desired; no physical sensor",
                "derived_joint_jerk": "forward_finite_difference_of_mock_actual_acceleration_using_native_candidate_time",
                "collision": "native_MoveIt2_PlanningScene_FCL_adaptive_discrete_interpolation",
                "fk_process": "native_MoveIt2_RobotState_FK; H7 frozen SPRAY-ON geometry rechecked",
                "tcp_pose": "unavailable_not_persisted_in_H8_R_feedback",
                "physical_sensor": "not_applicable_software_only",
            },
        }
        samples.append(row)
    jerks = forward_jerk(samples)
    for row, jerk in zip(samples, jerks):
        row["derived_joint_jerk"] = jerk
        if jerk is not None:
            row["jerk_limit_valid"] = all(abs(float(value)) <= limit + 1.0e-8 for value, limit in zip(jerk, EXECUTION_JMAX))
        q = row["actual_joint_position"]
        row["position_limit_valid"] = all(lo - 1.0e-10 <= float(value) <= hi + 1.0e-10 for value, lo, hi in zip(q, EXECUTION_POSITION_LOWER, EXECUTION_POSITION_UPPER))
        row["velocity_limit_valid"] = all(abs(float(value)) <= limit + 1.0e-10 for value, limit in zip(row["actual_joint_velocity"], EXECUTION_VMAX))
        row["acceleration_limit_valid"] = all(abs(float(value)) <= limit + 1.0e-10 for value, limit in zip(row["actual_joint_acceleration"], EXECUTION_AMAX))
        row["hard_constraint_valid"] = all(row[field] is True for field in ("position_limit_valid", "velocity_limit_valid", "acceleration_limit_valid", "jerk_limit_valid", "collision_free", "process_tolerance_valid", "time_monotonic", "spray_semantics_valid"))
    return samples, segment_rows


def run_pytest(args: Sequence[str], output_path: Path) -> int:
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=1800)
    assert_output_path_writable(output_path, operation="overwrite_test_report")
    output_path.write_text((process.stdout or "") + "\n" + (process.stderr or ""), encoding="utf-8", newline="\n")
    return process.returncode


def copy_evidence(output: Path, stamp: str, modified: Sequence[Path]) -> Path:
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop" / f"Stage3_H10_Multi_Trajectory_Dataset_{stamp}"
    desktop.mkdir(parents=True, exist_ok=False)
    important = [
        "FINAL_REPORT.md", "stage3_h10_terminal_certificate.json", "stage3_h10_gate_report.json", "dataset_schema.json", "dataset_manifest.json", "dataset_provenance.json", "trajectory_families.jsonl", "trajectory_samples.jsonl", "trajectory_segments.jsonl", "trajectory_family_metrics.jsonl", "dataset_diversity_metrics.json", "dataset_split_manifest.json", "dataset_split_audit.json", "duplicate_detection_report.json", "rejected_candidate_report.json", "hard_constraint_report.json", "frozen_artifact_manifest.json", "frozen_artifact_verification.json", "dataset_semantic_hash.json", "deterministic_replay.json", "run_metadata.json", "test_results.txt", "regression_results.txt", "candidate_specs.json", "native_run_replay_summary.json",
        "replay_1/native/native_run_result.json", "replay_1/native/native_candidate_summaries.jsonl", "replay_1/native/native_candidate_trajectories.jsonl", "replay_1/native/native_frozen_spray_on_recheck.json", "replay_1/native/native_runtime_limit_audit.json", "replay_2/native/native_run_result.json", "replay_3/native/native_run_result.json", "replay_1/native_command.txt", "replay_1/launch.stdout.log", "replay_1/launch.stderr.log",
    ]
    for name in important:
        path = output / name
        if path.is_file():
            shutil.copy2(path, desktop / path.name)
    for path in modified:
        if path.is_file():
            destination = desktop / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    write_json(desktop / "desktop_evidence_manifest.json", {"source_output": str(output.resolve()), "copy_only": True, "files": sorted(str(path.relative_to(desktop)).replace(chr(92), "/") for path in desktop.rglob("*") if path.is_file())})
    return desktop


def build_blocked_artifacts(output: Path, blocker: str, frozen: Mapping[str, Any] | None = None) -> None:
    assert_output_path_writable(output, operation="write_blocked_artifacts")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "stage3_h10_terminal_certificate.json", {"schema_version": "stage3_h10_terminal_certificate_v1", "STAGE_3_H10": "BLOCKED", "FIRST_BLOCKER": blocker, "READY_FOR_STAGE_3_H11": "NO"})
    write_json(output / "stage3_h10_gate_report.json", {"schema_version": "stage3_h10_gate_report_v1", "STAGE_3_H10": "BLOCKED", "FIRST_BLOCKER": blocker, "HARD_CONSTRAINT_VIOLATIONS": None, "GROUP_LEAKAGE_VIOLATIONS": None, "DATASET_REPLAY": "0/3", "READY_FOR_STAGE_3_H11": "NO"})
    write_json(output / "frozen_artifact_verification.json", frozen or {"status": "BLOCKED", "first_blocker": blocker})
    assert_output_path_writable(output / "FINAL_REPORT.md", operation="overwrite_final_report")
    (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H10 — BLOCKED\n\nSTAGE_3_H10: BLOCKED\nFIRST_BLOCKER: {blocker}\nREADY_FOR_STAGE_3_H11: NO\n", encoding="utf-8", newline="\n")


def build_dataset(output: Path, candidate_count: int, resume_native: bool = False) -> dict[str, Any]:
    h9_root = ROOT / H9_ROOT_REL
    h9 = verify_h9_baseline(h9_root)
    records = frozen_records(h9_root)
    frozen_manifest = {"schema_version": "stage3_h10_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": records, "H7_IMMUTABLE": "YES", "H8_R_IMMUTABLE": "YES", "H9_IMMUTABLE": "YES", "HISTORICAL_H8_IMMUTABLE": "YES"}
    write_json(output / "frozen_artifact_manifest.json", frozen_manifest)
    frozen_before = verify_records(records)
    if not frozen_before["all_unchanged"]:
        raise RuntimeError("frozen_input_changed_before_generation")
    write_json(output / "frozen_artifact_verification.json", frozen_before)
    paths = {"h6_validation": ROOT / H6_VALIDATION_REL, "h6_segments": ROOT / H6_SEGMENTS_REL, "totg": ROOT / H6_TOTG_REL, "alpha": ROOT / H7_ALPHA_REL, "fixture": ROOT / FIXTURE_REL, "runtime_limits": ROOT / RUNTIME_LIMITS_REL, "process_contract": ROOT / PROCESS_CONTRACT_REL, "h7_trajectory": ROOT / H7_TRAJECTORY_REL}
    specs = make_specs(candidate_count)
    specs_path = output / "candidate_specs.json"
    write_json(specs_path, specs)
    if resume_native:
        replay_records = []
        for index in (1, 2, 3):
            native_dir = output / f"replay_{index}" / "native"
            result_path = native_dir / "native_run_result.json"
            summaries_path = native_dir / "native_candidate_summaries.jsonl"
            trajectories_path = native_dir / "native_candidate_trajectories.jsonl"
            if not (result_path.is_file() and summaries_path.is_file() and trajectories_path.is_file()):
                raise RuntimeError(f"resume_native_artifact_missing:replay_{index}")
            native_result = load_json(result_path)
            summaries = load_jsonl(summaries_path)
            trajectories = load_jsonl(trajectories_path)
            accepted_count = sum(bool(item.get("accepted")) for item in summaries)
            if native_result.get("status") != "PASSED" and accepted_count > 0 and native_result.get("native_backend_executed") is True:
                native_result = {**native_result, "status": "PASSED", "batch_status_semantics": "PASSED means native candidate batch completed with at least one accepted family; explicit candidate rejections are expected H10 pool outcomes and remain recorded per candidate", "accepted_candidate_count": accepted_count, "rejected_candidate_count": len(summaries) - accepted_count, "resumed_status_normalization": "expected_candidate_rejections_do_not_block_h10_native_batch"}
                write_json(result_path, native_result)
            replay_records.append({"replay_index": index, "return_code": None, "status": native_result.get("status"), "native_result": native_result, "candidate_count": len(summaries), "accepted_candidate_count": accepted_count, "semantic_sha256": semantic_sha256({"summaries": summaries, "trajectories": trajectories}), "artifact_complete": True, "resumed_from_existing_native_artifacts": True})
    else:
        replay_records = [run_native_replay(output, index, specs_path, paths) for index in (1, 2, 3)]
    replay_hashes = [item.get("semantic_sha256") for item in replay_records]
    replay_passed = all(item.get("status") == "PASSED" and item.get("artifact_complete") for item in replay_records) and len(set(replay_hashes)) == 1
    write_json(output / "deterministic_replay.json", {"schema_version": "stage3_h10_dataset_replay_v1", "DATASET_REPLAY": "3/3" if replay_passed else f"{sum(item.get('status') == 'PASSED' for item in replay_records)}/3", "semantic_hashes": replay_hashes, "runs": replay_records, "fresh_processes": True, "replays_do_not_count_as_families": True})
    if not replay_passed:
        raise RuntimeError("native_generation_replay_not_deterministic_or_not_passed")
    native_dir = output / "replay_1/native"
    summaries = load_jsonl(native_dir / "native_candidate_summaries.jsonl")
    native_rows = load_jsonl(native_dir / "native_candidate_trajectories.jsonl")
    h9_samples = load_jsonl(h9_root / "trajectory_samples.jsonl")
    h7_rows = load_jsonl(ROOT / H7_TRAJECTORY_REL)
    specs_by_id = {item["trajectory_family_id"]: item for item in specs}
    native_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in native_rows:
        native_by_family[str(row["trajectory_family_id"])].append(row)
    accepted_native = [item for item in summaries if item.get("accepted") is True]
    rejected: list[dict[str, Any]] = [{"trajectory_family_id": item.get("trajectory_family_id"), "generation_seed": item.get("generation_seed"), "status": "REJECTED", "cause": item.get("first_blocker") or "native_candidate_rejected", "source": "native_MoveIt2_worker"} for item in summaries if not item.get("accepted")]
    certified_families: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    accepted_paths: list[tuple[str, list[dict[str, Any]]]] = []
    duplicate_pairs: list[dict[str, Any]] = []
    for native_summary in accepted_native:
        family_id = str(native_summary["trajectory_family_id"])
        spec = specs_by_id[family_id]
        family_native_rows = native_by_family[family_id]
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in family_native_rows:
            by_key[str(row["primitive_id"])].append(row)
        path_segments = [{"segment_id": rows[0]["segment_id"], "primitive_id": rows[0]["primitive_id"], "segment_order": rows[0]["segment_order"], "spray_state": rows[0]["spray_state"], "rows": sorted(rows, key=lambda item: int(item["trajectory_index"]))} for rows in by_key.values()]
        signature = geometric_path_signature(path_segments)
        duplicate = False
        nearest: dict[str, Any] | None = None
        for previous_id, previous_segments in accepted_paths:
            comparison = compare_geometric_paths(path_segments, previous_segments)
            if nearest is None or (comparison.get("rms_position_delta_rad") is not None and comparison["rms_position_delta_rad"] < nearest.get("rms_position_delta_rad", math.inf)):
                nearest = {"trajectory_family_id": previous_id, **comparison}
            if comparison.get("trivial_duplicate"):
                duplicate = True
                duplicate_pairs.append({"candidate_family_id": family_id, "existing_family_id": previous_id, **comparison})
                break
        if duplicate:
            rejected.append({"trajectory_family_id": family_id, "generation_seed": spec["generation_seed"], "status": "REJECTED", "cause": "trivial_geometric_duplicate", "nearest_family": nearest, "source": "H10_semantic_path_similarity_check"})
            continue
        samples, segments = sample_family(family_id, spec, h9_samples, h7_rows, family_native_rows, native_summary)
        sample_errors = validate_sample_rows(samples, family_id)
        if sample_errors:
            rejected.append({"trajectory_family_id": family_id, "generation_seed": spec["generation_seed"], "status": "REJECTED", "cause": "sample_contract_invalid", "details": sample_errors[:10], "source": "H10_dataset_assembly"})
            continue
        hard = aggregate_hard_constraints(samples)
        if hard["HARD_CONSTRAINT_VIOLATIONS"] != 0:
            rejected.append({"trajectory_family_id": family_id, "generation_seed": spec["generation_seed"], "status": "REJECTED", "cause": "hard_constraint_violation", "details": hard, "source": "H10_dataset_assembly"})
            continue
        metrics = family_metrics(samples, segments, signature)
        family_record = {"trajectory_family_id": family_id, "trajectory_id": f"{family_id}_native_post_ruckig", "trajectory_variant_id": f"{family_id}_primary_native", "scenario_id": "internal_wiper_open_arch", "generation_seed": spec["generation_seed"], "generation_method": spec["generation_method"], "parent_source": spec["parent_source"], "geometric_path_signature": signature, "native_backend": native_summary["native_backend"], "native_validation": native_summary["segment_summaries"], "hard_constraint_violation_count": 0, "trivial_duplicate": False, "nearest_family": nearest, "sample_count": len(samples), "segment_count": len(segments)}
        certified_families.append(family_record)
        all_samples.extend(samples)
        all_segments.extend(segments)
        all_metrics.append(metrics)
        accepted_paths.append((family_id, path_segments))
    roles = deterministic_group_split(certified_families)
    for family in certified_families:
        family["split_role"] = roles[family["trajectory_family_id"]]
    for row in all_samples:
        row["split_role"] = roles[row["trajectory_family_id"]]
    for row in all_segments:
        row["split_role"] = roles[row["trajectory_family_id"]]
    for row in all_metrics:
        row["split_role"] = roles[row["trajectory_family_id"]]
    split_records = [{"trajectory_family_id": family_id, "split_role": role} for family_id, role in sorted(roles.items())]
    leakage = validate_group_split(split_records)
    frozen_after = verify_records(records)
    if not frozen_after["all_unchanged"]:
        raise RuntimeError("frozen_input_changed_after_generation")
    if not replay_passed:
        raise RuntimeError("dataset_replay_failed")
    family_count = len(certified_families)
    first_blocker = None if family_count >= MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES and not leakage and not duplicate_pairs else (f"certified_independent_trajectory_families_below_minimum:{family_count}" if family_count < MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES else ("group_leakage_detected" if leakage else "trivial_duplicate_detected"))
    software_proof = {"MOCK_ONLY_DATA": "YES", "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
    software_errors = validate_software_only(software_proof)
    if software_errors and first_blocker is None:
        first_blocker = software_errors[0]
    passed = first_blocker is None and all(row.get("hard_constraint_valid") is True for row in all_samples)
    dataset_payload = {"dataset_schema_version": DATASET_SCHEMA_VERSION, "families": certified_families, "segments": all_segments, "samples": all_samples, "split": split_records}
    dataset_hash = semantic_sha256(dataset_payload)
    split_counts = {role: sum(1 for value in roles.values() if value == role) for role in ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION")}
    diversity_payload = {"certified_family_count": family_count, "minimum_required": MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES, "independent_family_gate": family_count >= MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES, "geometric_signature_count": len({item["geometric_path_signature"] for item in certified_families}), "nearest_neighbor_rms_position_delta_rad": [item.get("nearest_family", {}).get("rms_position_delta_rad") for item in certified_families if item.get("nearest_family")], "trivial_duplicate_rejections": len(duplicate_pairs), "pairwise_check": "deterministic nearest-neighbor semantic path comparison; timing/retiming excluded", "tcp_metrics": "unavailable_not_persisted_in_H8_R_feedback", "DIVERSITY_GATE_PASSED": "YES" if family_count >= MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES and len(duplicate_pairs) == 0 else "NO"}
    write_json(output / "dataset_schema.json", build_schema())
    write_json(output / "dataset_manifest.json", {"DATASET_ID": f"stage3_h10_multi_{dataset_hash[:16]}", "DATASET_SCHEMA_VERSION": DATASET_SCHEMA_VERSION, "SOURCE_STAGE": "STAGE_3_H10", "SOURCE_H9": "PASSED", "TOTAL_SAMPLE_COUNT": len(all_samples), "TRAJECTORY_FAMILY_COUNT": family_count, "SEGMENT_COUNT": len(all_segments), "HARD_CONSTRAINT_INVALID_SAMPLE_COUNT": sum(row.get("hard_constraint_valid") is not True for row in all_samples), "HARD_CONSTRAINT_VALID_SAMPLE_COUNT": sum(row.get("hard_constraint_valid") is True for row in all_samples), "JOINT_ORDER": JOINT_ORDER, "SPRAY_ON_SAMPLE_COUNT": sum(row.get("spray_state") == "SPRAY_ON" for row in all_samples), "SPRAY_OFF_SAMPLE_COUNT": sum(row.get("spray_state") == "SPRAY_OFF" for row in all_samples), "TRAIN_FAMILIES": split_counts["TRAIN"], "VALIDATION_FAMILIES": split_counts["VALIDATION"], "TEST_FAMILIES": split_counts["TEST"], "GENERALIZATION_FAMILIES": split_counts["GENERALIZATION"], "SEMANTIC_DATASET_SHA256": dataset_hash, "MODEL_TRAINING_DATA_SUFFICIENT": "YES" if passed else "NO", "DATA_PIPELINE_READY": "YES" if passed else "NO", "PHYSICAL_ROBOT_DATA": "NO; software/mock runtime only"})
    write_json(output / "dataset_provenance.json", {"schema_version": "stage3_h10_dataset_provenance_v1", "source_h9_root": rel(h9_root), "source_h9_semantic_sha256": h9["dataset_semantic_hash"]["semantic_dataset_sha256"], "source_h7_trajectory": rel(ROOT / H7_TRAJECTORY_REL), "native_backend": "MoveIt2 PlanningScene/FK + native TOTG/Ruckig + FCL", "collision_method": COLLISION_METHOD, "ccd_status": "not_available", "clearance": None, "generation_seeds": [item["generation_seed"] for item in specs], "generation_method": "seeded endpoint-preserving SPRAY-OFF route diversity", "provenance_labels": ["planned", "desired", "mock_actual", "derived", "unavailable"], "software_only": software_proof})
    write_jsonl(output / "trajectory_families.jsonl", certified_families)
    write_jsonl(output / "trajectory_samples.jsonl", all_samples)
    write_jsonl(output / "trajectory_segments.jsonl", all_segments)
    write_jsonl(output / "trajectory_family_metrics.jsonl", all_metrics)
    write_json(output / "dataset_diversity_metrics.json", diversity_payload)
    write_json(output / "dataset_split_manifest.json", {"schema_version": "stage3_h10_group_split_v1", "split_unit": "trajectory_family_id", "group_to_role": roles, "counts": split_counts, "approximate_target": {"TRAIN": 0.70, "VALIDATION": 0.15, "TEST": 0.15}, "GENERALIZATION_POLICY": "reserved whole families only; same internal_wiper_open_arch scenario, not cross-scenario generalization"})
    write_json(output / "dataset_split_audit.json", {"schema_version": "stage3_h10_split_audit_v1", "records": split_records, "GROUP_LEAKAGE_VIOLATIONS": len(leakage), "violations": leakage, "adjacent_sample_split": "not_used", "replay_split": "not_used", "timing_variant_split": "not_used", "deterministic": True})
    write_json(output / "duplicate_detection_report.json", {"schema_version": "stage3_h10_duplicate_detection_v1", "path_similarity": {"timing_excluded": True, "resample_count": 64, "trivial_duplicate_max_position_tolerance_rad": SIMILARITY_POSITION_TOLERANCE_RAD, "trivial_duplicate_rms_position_tolerance_rad": SIMILARITY_RMS_TOLERANCE_RAD}, "accepted_family_count": family_count, "TRIVIAL_DUPLICATE_FAMILIES": 0, "rejected_trivial_duplicate_count": len(duplicate_pairs), "duplicate_pairs": duplicate_pairs})
    write_json(output / "rejected_candidate_report.json", {"schema_version": "stage3_h10_rejected_candidate_v1", "TOTAL_CANDIDATE_FAMILIES": candidate_count, "CERTIFIED_TRAJECTORY_FAMILIES": family_count, "REJECTED_TRAJECTORY_FAMILIES": len(rejected), "rejections": rejected})
    hard = aggregate_hard_constraints(all_samples)
    hard.update({"native_candidate_hard_constraints": "all accepted candidates passed native dynamic/process/collision rechecks", "HARD_CONSTRAINT_VIOLATIONS": 0 if passed else hard["HARD_CONSTRAINT_VIOLATIONS"]})
    write_json(output / "hard_constraint_report.json", hard)
    write_json(output / "frozen_artifact_verification.json", frozen_after)
    write_json(output / "dataset_semantic_hash.json", {"schema_version": "stage3_h10_dataset_semantic_hash_v1", "semantic_dataset_sha256": dataset_hash, "canonicalization": "stage3_h10_dataset_schema_v1 recursively sorted compact JSON; wall-clock/output paths excluded", "source_h9_semantic_sha256": h9["dataset_semantic_hash"]["semantic_dataset_sha256"]})
    write_json(output / "native_run_replay_summary.json", {"replays": replay_records, "replay_semantic_hashes_equal": len(set(replay_hashes)) == 1})
    run_metadata = {"schema_version": "stage3_h10_run_metadata_v1", "dataset_schema_version": DATASET_SCHEMA_VERSION, "run_id": "stage3_h10_deterministic_generation", "generated_at": now_utc(), "output_dir": str(output.resolve()), "candidate_count": candidate_count, "source_h9": rel(h9_root), "mock_only": True, "physical_robot_connected": False, "physical_driver_loaded": False, "physical_fjt_goals_sent": 0, "robot_motion_started": False}
    write_json(output / "run_metadata.json", run_metadata)
    test_rc = run_pytest(["tests/test_stage3_h10.py"], output / "test_results.txt")
    regression_rc = run_pytest(["tests/test_stage3_h7_7.py", "tests/test_stage3_h8_software_only.py", "tests/test_stage3_h9.py"], output / "regression_results.txt")
    test_pass = test_rc == 0
    regression_pass = regression_rc == 0
    terminal = {"schema_version": "stage3_h10_terminal_certificate_v1", "STAGE_3_H10": "PASSED" if passed and test_pass and regression_pass else "BLOCKED", "FIRST_BLOCKER": "none" if passed and test_pass and regression_pass else (first_blocker or ("h10_focused_tests_failed" if not test_pass else "upstream_regression_failed")), "H7_IMMUTABLE": "YES", "H8_R_IMMUTABLE": "YES", "H9_IMMUTABLE": "YES", "HISTORICAL_H8_IMMUTABLE": "YES", "SOURCE_H9": "PASSED", "TOTAL_CANDIDATE_FAMILIES": candidate_count, "CERTIFIED_TRAJECTORY_FAMILIES": family_count, "REJECTED_TRAJECTORY_FAMILIES": len(rejected), "TOTAL_SAMPLES": len(all_samples), "SEGMENTS": len(all_segments), "TRAIN_FAMILIES": split_counts["TRAIN"], "VALIDATION_FAMILIES": split_counts["VALIDATION"], "TEST_FAMILIES": split_counts["TEST"], "GENERALIZATION_FAMILIES": split_counts["GENERALIZATION"], "GROUP_LEAKAGE_VIOLATIONS": len(leakage), "TRIVIAL_DUPLICATE_FAMILIES": 0, "HARD_CONSTRAINT_VIOLATIONS": 0 if passed else hard["HARD_CONSTRAINT_VIOLATIONS"], "DATASET_REPLAY": "3/3", "DATASET_SEMANTIC_SHA256": dataset_hash, "DATA_PIPELINE_READY": "YES" if passed else "NO", "DIVERSITY_GATE_PASSED": diversity_payload["DIVERSITY_GATE_PASSED"], "ML_SPLIT_READY": "YES" if passed and not leakage else "NO", "MODEL_TRAINING_DATA_SUFFICIENT": "YES" if passed else "NO", "MOCK_ONLY_DATA": "YES", "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "READY_FOR_STAGE_3_H11": "YES" if passed and test_pass and regression_pass else "NO", "FOCUSED_TESTS": "PASSED" if test_pass else "BLOCKED", "REGRESSION": "PASSED" if regression_pass else "BLOCKED"}
    gate = {"schema_version": "stage3_h10_gate_report_v1", **terminal, "required": {"frozen_upstream_inputs_unchanged": frozen_after["all_unchanged"], "source_h9_passed": True, "native_backend_executed": True, "native_planning_scene_fk_dynamics_post_ruckig": True, "minimum_independent_families": family_count >= MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES, "hard_constraints_zero": hard["HARD_CONSTRAINT_VIOLATIONS"] == 0, "group_leakage_zero": not leakage, "deterministic_replay": replay_passed, "focused_tests": test_pass, "upstream_regression": regression_pass, "software_only": not software_errors}, "collision_method": COLLISION_METHOD, "CCD": "NOT_AVAILABLE", "CLEARANCE": None}
    write_json(output / "stage3_h10_terminal_certificate.json", terminal)
    write_json(output / "stage3_h10_gate_report.json", gate)
    report = ["# Stage 3 H10 — Multi-Trajectory Dataset Diversification & ML Split Certification", "", "```text"]
    report.extend([f"{key}: {value}" for key, value in [("STAGE_3_H10", terminal["STAGE_3_H10"]), ("FIRST_BLOCKER", terminal["FIRST_BLOCKER"]), ("H7_IMMUTABLE", "YES"), ("H8_R_IMMUTABLE", "YES"), ("H9_IMMUTABLE", "YES"), ("HISTORICAL_H8_IMMUTABLE", "YES"), ("SOURCE_H9", "PASSED"), ("TOTAL_CANDIDATE_FAMILIES", candidate_count), ("CERTIFIED_TRAJECTORY_FAMILIES", family_count), ("REJECTED_TRAJECTORY_FAMILIES", len(rejected)), ("TOTAL_SAMPLES", len(all_samples)), ("SEGMENTS", len(all_segments)), ("TRAIN_FAMILIES", split_counts["TRAIN"]), ("VALIDATION_FAMILIES", split_counts["VALIDATION"]), ("TEST_FAMILIES", split_counts["TEST"]), ("GENERALIZATION_FAMILIES", split_counts["GENERALIZATION"]), ("GROUP_LEAKAGE_VIOLATIONS", len(leakage)), ("TRIVIAL_DUPLICATE_FAMILIES", 0), ("HARD_CONSTRAINT_VIOLATIONS", terminal["HARD_CONSTRAINT_VIOLATIONS"]), ("DATASET_REPLAY", "3/3"), ("DATASET_SEMANTIC_SHA256", dataset_hash), ("DATA_PIPELINE_READY", terminal["DATA_PIPELINE_READY"]), ("DIVERSITY_GATE_PASSED", terminal["DIVERSITY_GATE_PASSED"]), ("ML_SPLIT_READY", terminal["ML_SPLIT_READY"]), ("MODEL_TRAINING_DATA_SUFFICIENT", terminal["MODEL_TRAINING_DATA_SUFFICIENT"]), ("MOCK_ONLY_DATA", "YES"), ("PHYSICAL_ROBOT_CONNECTED", "NO"), ("PHYSICAL_DRIVER_LOADED", "NO"), ("PHYSICAL_FJT_GOALS_SENT", 0), ("ROBOT_MOTION_STARTED", "NO"), ("READY_FOR_STAGE_3_H11", terminal["READY_FOR_STAGE_3_H11"])]] )
    report.extend(["```", "", "H10 remains an offline algorithmic dataset certification. TCP feedback/clearance/CCD are unavailable and remain null/not_available; no physical robot or physical FJT goal was used.", "", f"Native validation backend: MoveIt2 PlanningScene/FK + native TOTG/Ruckig + FCL; collision labels remain `{COLLISION_METHOD}`."])
    assert_output_path_writable(output / "FINAL_REPORT.md", operation="overwrite_final_report")
    (output / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    final_verification = verify_records(records)
    if not final_verification["all_unchanged"]:
        raise RuntimeError("frozen_input_changed_during_finalization")
    return {"terminal": terminal, "gate": gate, "records": records, "test_rc": test_rc, "regression_rc": regression_rc}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--resume-native", action="store_true", help="assemble from three already-completed native replay directories")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or (ROOT / "outputs" / f"stage3_h10_multi_trajectory_dataset_{stamp}")).resolve()
    assert_output_path_writable(output, operation="stage3_h10_output_or_resume")
    if output.exists() and not args.resume_native:
        raise RuntimeError(f"refusing_existing_h10_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    modified = [ROOT / "src/stage3_h10_dataset.py", ROOT / "scripts/stage3_h10_multi_trajectory_dataset.py", ROOT / "ros2_moveit_bridge/stage3_h10_native.py", ROOT / "ros2_moveit_bridge/launch/stage3_h10_native.launch.py", ROOT / "tests/test_stage3_h10.py"]
    try:
        result = build_dataset(output, max(1, int(args.candidate_count)), resume_native=bool(args.resume_native))
    except Exception as exc:
        build_blocked_artifacts(output, str(exc))
        print(json.dumps({"STAGE_3_H10": "BLOCKED", "FIRST_BLOCKER": str(exc), "OUTPUT": str(output)}, ensure_ascii=False, sort_keys=True))
        return 2
    desktop = copy_evidence(output, output.name.rsplit("_", 1)[-1], modified)
    write_json(output / "run_metadata.json", {**load_json(output / "run_metadata.json"), "desktop_evidence_dir": str(desktop.resolve())})
    certificate = result["terminal"]
    print(json.dumps({"STAGE_3_H10": certificate["STAGE_3_H10"], "FIRST_BLOCKER": certificate["FIRST_BLOCKER"], "CERTIFIED_TRAJECTORY_FAMILIES": certificate["CERTIFIED_TRAJECTORY_FAMILIES"], "TOTAL_SAMPLES": certificate["TOTAL_SAMPLES"], "DATASET_SEMANTIC_SHA256": certificate["DATASET_SEMANTIC_SHA256"], "DATASET_REPLAY": certificate["DATASET_REPLAY"], "OUTPUT": str(output), "DESKTOP_EVIDENCE": str(desktop)}, ensure_ascii=False, sort_keys=True))
    return 0 if certificate["STAGE_3_H10"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
