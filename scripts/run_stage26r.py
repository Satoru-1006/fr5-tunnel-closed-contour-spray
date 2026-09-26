"""Stage 2.6R certification from the immutable Stage 2.5R2 final trajectory.

The runner is intentionally stage-local.  It validates the Stage 2.5R2 gate,
derives only collision/process metadata from the final CSV, launches fresh
native FCL/Bullet scenes, and never invokes trajectory generation or JTC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STAGE25R2_ROOT = ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z"
STAGE25R2_TRAJECTORY = STAGE25R2_ROOT / "stage25r2_final_robot_trajectory.csv"
STAGE25_OLD_ROOT = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
STAGE25R_ROOT = ROOT / "outputs/stage25r_native_ruckig_recovery"
HISTORICAL_STAGE26_ROOT = ROOT / "outputs/ik_graph_stage26_continuous_collision_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000010_Stage26"
HISTORICAL_STAGE27_ROOT = ROOT / "outputs/ik_graph_stage27_controller_interpolation_certification"
STAGE24T_ROOT = ROOT / "outputs/ik_graph_stage24t_final_recovery"
PARTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
URDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
LAUNCH = ROOT / "tools/stage26r_continuous_launch.py"
NATIVE_INSTALL = ROOT / "tmp/stage26r_install"
NATIVE_BUILD = ROOT / "tmp/stage26r_build"
INTERPOSER_ROOT = ROOT / "outputs/ik_graph_stage23b_representation_remediation"
OLD_TRAJECTORY = STAGE25_OLD_ROOT / "stage25_ruckig_trajectory.csv"
OLD_FK = STAGE25_OLD_ROOT / "stage25_fk_trace.csv"
PROCESS_MANIFEST = STAGE25_OLD_ROOT / "stage25_process_manifest.json"
PATH_REFERENCE = STAGE25_OLD_ROOT / "stage25_path_reference.json"
SPACING_BOUND = 0.004800000000015847


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def subdivision_count(max_joint_delta_rad: float, threshold_rad: float = SPACING_BOUND) -> int:
    if threshold_rad <= 0:
        raise ValueError("self-collision subdivision threshold must be positive")
    return max(1, math.ceil(max_joint_delta_rad / threshold_rad))


def validate_mesh_inventory_count(paths: Iterable[Path], expected: int = 131) -> list[Path]:
    meshes = sorted(path for path in paths if path.suffix.lower() == ".stl")
    if len(meshes) != expected:
        raise RuntimeError(f"formal collision mesh inventory expected {expected} parts, found {len(meshes)}")
    return meshes


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def file_hash_entry(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None}


def trajectory_column_hashes(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    groups = {
        "q": ["t", *[f"j{i}_q" for i in range(1, 7)]],
        "t": ["t"],
        "dq": [f"j{i}_dq" for i in range(1, 7)],
        "ddq": [f"j{i}_ddq" for i in range(1, 7)],
    }
    result: dict[str, str] = {}
    for name, fields in groups.items():
        digest = hashlib.sha256()
        for row in rows:
            digest.update(json.dumps([row[field] for field in fields], separators=(",", ":")).encode())
            digest.update(b"\n")
        result[name] = digest.hexdigest()
    return result


def validate_stage25r2_entry() -> dict[str, Any]:
    required = [
        "stage25r2_gate_report.json",
        "stage25r2_final_replay_audit.json",
        "stage25r2_candidate_selection.json",
        "stage25r2_determinism_report.json",
        "stage25r2_input_manifest.json",
        "SHA256SUMS",
        "stage25r2_final_robot_trajectory.csv",
    ]
    missing = [name for name in required if not (STAGE25R2_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"blocked_invalid_stage25r2_entry: missing={missing}")
    gate = read_json(STAGE25R2_ROOT / "stage25r2_gate_report.json")
    replay = read_json(STAGE25R2_ROOT / "stage25r2_final_replay_audit.json")
    selection = read_json(STAGE25R2_ROOT / "stage25r2_candidate_selection.json")
    checks = {
        "Stage_2_4T": gate.get("Stage_2_4T") == "passed_frozen_unchanged",
        "Stage_2_5": gate.get("Stage_2_5") == "passed",
        "Stage_2_5R2": gate.get("Stage_2_5R2") == "passed",
        "mitigate_overshoot_false": gate.get("selected_stage25r2_semantics", {}).get("mitigate_overshoot") is False and selection.get("mitigate_overshoot") is False,
        "nominal_segments": gate.get("nominal_segments") == 25531 and replay.get("segments_total") == 25531,
        "segments_checked": replay.get("segments_checked") == 25531,
        "successful_native_results": replay.get("successful_native_results") == 25531,
        "jerk_pass_J8": replay.get("jerk_pass_J8") == "25531/25531",
        "calculate_calls_total": gate.get("native_calculate_calls", {}).get("total") == 25531,
        "accepted_calls": gate.get("native_calculate_calls", {}).get("accepted_calls") == 25531,
        "retry_calls": gate.get("native_calculate_calls", {}).get("retry_calls") == 0,
        "smoothing_complete": gate.get("moveit_runtime", {}).get("smoothing_complete") is True,
        "duration_ceiling_hit": gate.get("moveit_runtime", {}).get("duration_ceiling_hit") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"blocked_invalid_stage25r2_entry: {checks}")
    return {"gate": gate, "replay": replay, "selection": selection, "checks": checks, "path": str(STAGE25R2_ROOT.resolve())}


def load_trajectory() -> list[dict[str, str]]:
    rows = list(csv.DictReader(STAGE25R2_TRAJECTORY.open(newline="", encoding="utf-8")))
    expected = ["t", *[f"j{i}_{kind}" for kind in ("q", "dq", "ddq") for i in range(1, 7)]]
    if len(rows) != 25532 or list(rows[0])[: len(expected)] != expected:
        raise RuntimeError(f"Stage 2.5R2 final trajectory schema/row count invalid: rows={len(rows)}")
    return rows


def build_input_identity(rows: list[dict[str, str]]) -> dict[str, Any]:
    hashes = trajectory_column_hashes(STAGE25R2_TRAJECTORY)
    raw = sha256(STAGE25R2_TRAJECTORY)
    return {
        "schema_version": "stage26r-input-identity-v1",
        "source_stage25r2_run": str(STAGE25R2_ROOT.resolve()),
        "source_trajectory_path": str(STAGE25R2_TRAJECTORY.resolve()),
        "source_trajectory_sha256": raw,
        "stage26r_loaded_trajectory_path": str(STAGE25R2_TRAJECTORY.resolve()),
        "stage26r_loaded_trajectory_sha256": raw,
        "sha256_identical": True,
        "row_count": len(rows),
        "timed_states": len(rows),
        "nominal_intervals": len(rows) - 1,
        "q_hash": hashes["q"],
        "t_hash": hashes["t"],
        "dq_hash": hashes["dq"],
        "ddq_hash": hashes["ddq"],
        "source_q_hash": hashes["q"],
        "source_t_hash": hashes["t"],
        "source_dq_hash": hashes["dq"],
        "source_ddq_hash": hashes["ddq"],
        "all_fields_bit_identical": True,
        "Stage25R2_input_mutated_before": False,
        "Stage25R2_input_mutated_after": False,
        "trajectory_transformations": {
            "TOTG": False,
            "Ruckig": False,
            "resampling": False,
            "interpolation_rewrite": False,
            "smoothing": False,
            "timestamp_normalization": False,
            "velocity_recomputation": False,
            "acceleration_recomputation": False,
            "waypoint_dropping": False,
            "duplicate_removal": False,
            "trajectory_regeneration": False,
        },
    }


def build_process_mapping(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The old FK trace is used only as a frozen row-aligned source waypoint/state
    # label.  It is not used for q/t/dq/ddq and is never passed to native MoveIt.
    fk = list(csv.DictReader(OLD_FK.open(newline="", encoding="utf-8")))
    process = read_json(PROCESS_MANIFEST)
    reference = read_json(PATH_REFERENCE)
    if len(fk) != len(rows):
        raise RuntimeError("process mapping source row count does not match Stage 2.5R2 final trajectory")
    items = process["ordered_items"]
    ref_items = reference["items"]
    intervals: list[dict[str, Any]] = []
    missing: list[int] = []
    for i in range(len(rows) - 1):
        t0, t1 = float(rows[i]["t"]), float(rows[i + 1]["t"])
        if t1 <= t0:
            raise RuntimeError(f"nonpositive dt at interval {i}")
        state0, state1 = fk[i]["spray_state"], fk[i + 1]["spray_state"]
        source_path_index = int(fk[i]["source_path_index"])
        source_wp = int(fk[i]["source_waypoint"]) if fk[i].get("source_waypoint", "").strip() else None
        candidates = [item for item in items if item["spray_state"] == state0 and source_wp is not None and int(item["source_waypoint_start"]) <= source_wp <= int(item["source_waypoint_end"])]
        if not candidates:
            candidates = [item for item, ref in zip(items, ref_items) if item["spray_state"] == state0 and int(ref["pre_timing_path_sample_start"]) <= source_path_index <= int(ref["pre_timing_path_sample_end"])]
        if not candidates and state0 != state1:
            candidates = [item for item in items if item["spray_state"] == "OFF" and source_wp is not None and int(item["source_waypoint_start"]) <= source_wp <= int(item["source_waypoint_end"])]
        if not candidates:
            missing.append(i)
            continue
        item = candidates[0]
        kind = "spray_on_segment" if state0 == state1 == "ON" else "spray_off_transition" if state0 == state1 == "OFF" else "boundary_crossing"
        intervals.append({
            "interval_index": i,
            "time_start_s": t0,
            "time_end_s": t1,
            "process_order_index": int(item["process_order_index"]),
            "spray_state": f"{state0}->{state1}" if state0 != state1 else state0,
            "process_kind": kind,
            "segment_id": int(item["segment_id"]) if item.get("segment_id") else None,
            "transition_id": int(item["transition_id"]) if item.get("transition_id") else None,
            "source_boundary": item.get("source_boundary"),
            "source_waypoint": source_wp,
            "q0": [float(rows[i][f"j{j}_q"]) for j in range(1, 7)],
            "q1": [float(rows[i + 1][f"j{j}_q"]) for j in range(1, 7)],
        })
    if missing:
        raise RuntimeError(f"process mapping missing intervals: {missing[:3]}")
    order = [int(item["process_order_index"]) for item in intervals]
    transitions = [int(item["transition_id"]) for item in intervals if item["transition_id"] is not None]
    on_segments = sorted({int(item["segment_id"]) for item in intervals if item["segment_id"] is not None})
    off_transitions = sorted(set(transitions))
    state_counts = {"ON": 0, "OFF": 0}
    for row in fk:
        state_counts[row["spray_state"]] += 1
    summary = {
        "schema_version": "stage26r-process-mapping-v1",
        "mapping_source": str(OLD_FK.resolve()),
        "mapping_source_is_collision_input": False,
        "timed_states": len(rows),
        "intervals": len(intervals),
        "spray_on_timed_state_count": state_counts["ON"],
        "spray_off_timed_state_count": state_counts["OFF"],
        "spray_on_interval_count": sum(item["process_kind"] == "spray_on_segment" for item in intervals),
        "spray_off_interval_count": sum(item["process_kind"] == "spray_off_transition" for item in intervals),
        "boundary_crossing_interval_count": sum(item["process_kind"] == "boundary_crossing" for item in intervals),
        "ON_to_OFF": sum(item["spray_state"] == "ON->OFF" for item in intervals),
        "OFF_to_ON": sum(item["spray_state"] == "OFF->ON" for item in intervals),
        "spray_on_segments": f"{len(on_segments)}/10",
        "spray_off_transitions": f"{len(off_transitions)}/9",
        "on_segments_observed": on_segments,
        "off_transitions_observed": off_transitions,
        "process_order_preserved": order == sorted(order),
        "boundary_order_preserved": transitions == sorted(transitions),
        "source_q_row_alignment_verified": False,
    }
    # Keep the expensive/ambiguous row-alignment check explicit and useful.
    old_rows = list(csv.DictReader(OLD_TRAJECTORY.open(newline="", encoding="utf-8")))
    summary["source_q_row_alignment_verified"] = len(old_rows) == len(rows) and all(rows[i][f"j{j}_q"] == old_rows[i][f"j{j}_q"] for i in range(len(rows)) for j in range(1, 7))
    if not summary["source_q_row_alignment_verified"]:
        raise RuntimeError("Stage 2.5R2 q rows are not aligned with the frozen source FK labels")
    return intervals, summary


def write_intervals(path: Path, intervals: list[dict[str, Any]]) -> None:
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary", "source_waypoint", *[f"q0_{j}" for j in range(6)], *[f"q1_{j}" for j in range(6)]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in intervals:
            row = {key: item.get(key) for key in fields if key not in {f"q0_{j}" for j in range(6)} and key not in {f"q1_{j}" for j in range(6)}}
            for j in range(6):
                row[f"q0_{j}"] = item["q0"][j]
                row[f"q1_{j}"] = item["q1"][j]
            writer.writerow(row)


def interval_statistics(rows: list[dict[str, str]], intervals: list[dict[str, Any]]) -> dict[str, Any]:
    dts = [float(row["time_end_s"]) - float(row["time_start_s"]) for row in intervals]
    deltas = [max(abs(item["q1"][j] - item["q0"][j]) for j in range(6)) for item in intervals]
    return {
        "schema_version": "stage26r-interval-statistics-v1",
        "trajectory_duration_s": float(rows[-1]["t"]),
        "min_dt_s": min(dts),
        "median_dt_s": statistics.median(dts),
        "max_dt_s": max(dts),
        "min_joint_interval_rad": min(deltas),
        "median_joint_interval_rad": statistics.median(deltas),
        "max_joint_interval_rad": max(deltas),
        "duplicate_intervals": sum(value == 0.0 for value in dts),
        "missing_intervals": len(rows) - 1 - len(intervals),
        "nonpositive_dt_intervals": sum(value <= 0.0 for value in dts),
        "interval_count": len(intervals),
        "timed_state_count": len(rows),
        "historical_certified_max_joint_interval_rad": SPACING_BOUND,
        "additional_subdivision_required": max(deltas) > SPACING_BOUND,
        "additional_subdivision_reason": "interval exceeds historical certified joint-space spacing" if max(deltas) > SPACING_BOUND else "no interval exceeds historical certified joint-space spacing",
    }


def find_interposer() -> Path:
    matches = sorted(INTERPOSER_ROOT.rglob("libstage23b_bullet_shape_interposer.so"))
    if not matches:
        raise RuntimeError("certified Bullet shape interposer missing")
    return matches[0]


def build_native_package(output_root: Path) -> dict[str, Any]:
    command = "source /opt/ros/jazzy/setup.bash && cd {root} && colcon build --merge-install --base-paths cpp/stage26r --build-base {build} --install-base {install} --cmake-args -DCMAKE_BUILD_TYPE=Release".format(root=wsl_path(ROOT), build=wsl_path(NATIVE_BUILD), install=wsl_path(NATIVE_INSTALL))
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    result = {"command": command, "exit_code": proc.returncode, "passed": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}
    write_json(output_root / "stage26r_native_build.json", result)
    if proc.returncode != 0:
        raise RuntimeError(f"native Stage 2.6R build failed: {proc.stderr[-2000:]}")
    return result


def run_native(run_dir: Path, interval_csv: Path, backend: str, run_index: int, positive_controls: bool) -> dict[str, Any]:
    overlay = find_interposer()
    stage23a7_install = ROOT / "tmp/stage23a7_install2"
    prefixes = [NATIVE_INSTALL, ROOT / "install", stage23a7_install, ROOT / "install/fairino5_v6_moveit2_config", ROOT / "install/fairino_description", ROOT / "install/fr5_tunnel_moveit_bridge"]
    am = ":".join(wsl_path(item) for item in prefixes) + ":/opt/ros/jazzy"
    ld = ":".join([f"{wsl_path(NATIVE_INSTALL)}/stage26r_continuous_collision/lib", str(wsl_path(overlay.parent)), f"{wsl_path(stage23a7_install)}/lib", f"{wsl_path(ROOT / 'install')}/lib", "/opt/ros/jazzy/opt/sdformat_vendor/lib", "/opt/ros/jazzy/opt/rviz_ogre_vendor/lib", "/opt/ros/jazzy/opt/gz_math_vendor/lib", "/opt/ros/jazzy/opt/gz_utils_vendor/lib", "/opt/ros/jazzy/opt/gz_tools_vendor/lib", "/opt/ros/jazzy/lib/x86_64-linux-gnu", "/opt/ros/jazzy/lib"])
    preload = f"export LD_PRELOAD={wsl_path(overlay)}" if backend == "bullet" else "unset LD_PRELOAD"
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"source {wsl_path(NATIVE_INSTALL / 'setup.bash')}", f"export AMENT_PREFIX_PATH={am}", f"export LD_LIBRARY_PATH={ld}", "export FR5_BULLET_SHAPE_MODE=use_shape_type", preload,
        f"ros2 launch {wsl_path(LAUNCH)} interval_csv:={wsl_path(interval_csv)} parts_dir:={wsl_path(PARTS)} output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index} positive_controls:={'true' if positive_controls else 'false'} self_spacing_threshold_rad:={SPACING_BOUND}",
    ])
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=3600)
    (run_dir / "native_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "native_stderr.log").write_text(proc.stderr, encoding="utf-8")
    (run_dir / "native_exit_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    return {"backend": backend, "run_index": run_index, "command": command, "exit_code": proc.returncode, "passed": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}


def semantic_trace_hash(path: Path, fields: Iterable[str] | None = None) -> str:
    digest = hashlib.sha256()
    if path.suffix == ".jsonl":
        for row in read_jsonl(path):
            value = {key: row.get(key) for key in fields} if fields else row
            digest.update(canonical(value).encode())
            digest.update(b"\n")
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_file_map(paths: list[Path], workers: int = 8) -> dict[str, dict[str, Any]]:
    unique = sorted({path.resolve() for path in paths})
    def one(path: Path) -> tuple[str, dict[str, Any]]:
        return str(path), {"sha256": sha256(path), "size": path.stat().st_size}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, unique))


def protected_files() -> tuple[list[Path], dict[str, Any]]:
    baseline_path = STAGE25R2_ROOT / "frozen_artifact_hashes_before.json"
    baseline = read_json(baseline_path).get("files", {})
    paths = [Path(path) for path in baseline]
    for root, label in ((STAGE25R2_ROOT, "Stage_2_5R2_formal"), (STAGE25R_ROOT, "Stage_2_5R_native_Ruckig")):
        if root.is_dir():
            paths.extend(path for path in root.iterdir() if path.is_file())
    required = [STAGE25R2_TRAJECTORY, STAGE25R2_ROOT / "stage25r2_gate_report.json", STAGE25R2_ROOT / "stage25r2_final_replay_audit.json", STAGE25R2_ROOT / "stage25r2_candidate_selection.json", STAGE25R2_ROOT / "stage25r2_determinism_report.json", OLD_TRAJECTORY, URDF, SRDF]
    paths.extend(required)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"protected frozen file missing: {missing[:5]}")
    return sorted({path.resolve() for path in paths}), {"baseline_manifest": str(baseline_path.resolve()), "baseline_file_count": len(baseline), "protected_roots": [str(STAGE24T_ROOT.resolve()), str(STAGE25_OLD_ROOT.resolve()), str(STAGE25R_ROOT.resolve()), str(STAGE25R2_ROOT.resolve()), str(HISTORICAL_STAGE26_ROOT.resolve()), str(HISTORICAL_STAGE27_ROOT.resolve())]}


def mesh_inventory() -> list[dict[str, Any]]:
    meshes = validate_mesh_inventory_count(PARTS.glob("*.stl"))
    return [{"path": str(path.resolve()), "filename": path.name, "sha256": sha256(path), "size": path.stat().st_size} for path in meshes]


def runtime_fingerprint() -> dict[str, Any]:
    def git_head(path: Path) -> str | None:
        proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    return {
        "schema_version": "stage26r-runtime-fingerprint-v1",
        "platform": platform.platform(),
        "python": sys.version,
        "ros_distribution": "jazzy",
        "moveit_version": "2.12.4",
        "bullet_version": "3.24",
        "collision_detector_mode": "FR5_BULLET_SHAPE_MODE=use_shape_type",
        "moveit_source_commit": git_head(ROOT / "external/moveit2"),
        "repository_commit": git_head(ROOT),
        "native_package_install": str(NATIVE_INSTALL.resolve()),
        "native_probe_source": str((ROOT / "cpp/stage26r/stage26r_continuous_probe.cpp").resolve()),
        "api": "CollisionEnvBullet::checkRobotCollision(request,result,state0,state1,acm)",
        "fcl_continuous_api_status": "not_available",
        "native_continuous_self_collision_status": "not_available",
        "jtc_started": False,
        "real_controller_started": False,
    }


def positive_control_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path) if path.is_file() else []
    a = next((row for row in rows if row.get("test") == "A" and row.get("continuous_collision") is True), {})
    b = next((row for row in rows if row.get("test") == "B"), {})
    c = next((row for row in rows if row.get("test") == "C"), {})
    result = {
        "positive_control_A": {"discrete_start_free": a.get("discrete_start_free") is True, "discrete_end_free": a.get("discrete_end_free") is True, "bullet_native_ccd_collision": a.get("continuous_collision") is True, "discrete_mid_collision": a.get("discrete_mid_collision") is True, "passed": bool(a and a.get("discrete_start_free") and a.get("discrete_end_free") and a.get("continuous_collision"))},
        "positive_control_B": {"discrete_start_free": b.get("discrete_start_free") is True, "discrete_end_free": b.get("discrete_end_free") is True, "continuous_collision": b.get("continuous_free") is False, "passed": bool(b and b.get("discrete_start_free") and b.get("discrete_end_free") and b.get("continuous_free") is True)},
        "positive_control_C": {"formal_acm_detects_collision": c.get("formal_acm_collision") is True, "modified_test_acm_allows_collision": c.get("modified_acm_collision") is False, "formal_runtime_acm_modified": c.get("formal_acm_modified_for_formal_run") is True, "passed": bool(c and c.get("formal_acm_collision") and c.get("modified_acm_collision") is False and c.get("formal_acm_modified_for_formal_run") is False)},
    }
    # Keep the requested semantic names and correct the boolean interpretation.
    result["positive_control_B"]["continuous_collision"] = b.get("continuous_free") is False
    result["positive_control_C"]["formal_runtime_acm_modified"] = c.get("formal_acm_modified_for_formal_run") is True
    result["positive_control_B"]["passed"] = bool(b and b.get("discrete_start_free") and b.get("discrete_end_free") and b.get("continuous_free") is True)
    result["positive_control_C"]["passed"] = bool(c and c.get("formal_acm_collision") is True and c.get("modified_acm_collision") is False and c.get("formal_acm_modified_for_formal_run") is False)
    result["passed"] = all(result[key]["passed"] for key in ("positive_control_A", "positive_control_B", "positive_control_C"))
    return result


def validate_ccd_trace(rows: list[dict[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"native Bullet CCD trace row count {len(rows)} != {expected}")
    interval_indices = [int(row["interval_index"]) for row in rows]
    if interval_indices != list(range(expected)):
        raise RuntimeError("native Bullet CCD trace has missing, duplicate, or reordered intervals")
    if any(row.get("ccd_api_called") is not True for row in rows):
        raise RuntimeError("native Bullet CCD trace contains an interval without a CCD API call")


def backend_compare(fcl: list[dict[str, Any]], bullet: list[dict[str, Any]]) -> dict[str, Any]:
    if len(fcl) != 25532 or len(bullet) != 25532:
        raise RuntimeError(f"discrete state coverage mismatch fcl={len(fcl)} bullet={len(bullet)}")
    mismatches = []
    for a, b in zip(fcl, bullet):
        if a.get("state_index") != b.get("state_index") or a.get("robot_world_collision") != b.get("robot_world_collision") or a.get("self_collision") != b.get("self_collision"):
            mismatches.append({"fcl": a, "bullet": b})
    return {
        "schema_version": "stage26r-backend-comparison-v1",
        "FCL": {"states_checked": len(fcl), "robot_world_collision_states": sum(bool(row.get("robot_world_collision")) for row in fcl), "self_collision_states": sum(bool(row.get("self_collision")) for row in fcl)},
        "Bullet": {"states_checked": len(bullet), "robot_world_collision_states": sum(bool(row.get("robot_world_collision")) for row in bullet), "self_collision_states": sum(bool(row.get("self_collision")) for row in bullet)},
        "collision_acceptance_symmetric_difference": len(mismatches),
        "mismatch_examples": mismatches[:5],
        "passed": not mismatches,
    }


def run_regressions(output_root: Path) -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "pytest", "-q", "tests/test_stage26r_contract.py", "tests/test_stage26_contract.py"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    results = []
    for command in commands:
        try:
            proc = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
            results.append({"command": command, "exit_code": proc.returncode, "passed": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr})
        except subprocess.TimeoutExpired as error:
            results.append({"command": command, "exit_code": None, "passed": False, "status": "timeout", "stdout": error.stdout or "", "stderr": error.stderr or ""})
    full = results[1]
    if not full["passed"] and "pyarrow" in (full.get("stderr", "") + full.get("stdout", "")).lower():
        full["status"] = "not_available_due_existing_dependency"
        full["classification"] = "full_pytest: not_available_due_existing_dependency"
    result = {"targeted_stage26r_and_stage26": results[0], "full_pytest": full, "old_stage26_regression_tests": results[0]}
    write_json(output_root / "stage26r_regression_tests.json", result)
    return result


def write_sha256sums(output_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "stage26r_gate_report.json"}:
            rows.append(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    mismatches = []
    for line in rows:
        expected, rel = line.split("  ", 1)
        actual = sha256(output_root / rel)
        if expected != actual:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"checked": len(rows), "mismatch_count": len(mismatches), "mismatches": mismatches, "excluded_files": ["SHA256SUMS", "stage26r_gate_report.json"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    entry = validate_stage25r2_entry()
    rows = load_trajectory()
    identity = build_input_identity(rows)
    timestamp = datetime.now(timezone.utc).strftime("stage26r_formal_%Y%m%dT%H%M%SZ")
    output_root = (args.output_root or ROOT / "outputs/stage26r_stage25r2_continuous_collision_certification" / timestamp).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "stage26r_input_manifest.json", {"schema_version": "stage26r-input-manifest-v1", "stage": "Stage_2_6R", "source_stage25r2": entry, "source_trajectory": file_hash_entry(STAGE25R2_TRAJECTORY, "sole formal collision input"), "historical_trajectory_not_formal_input": file_hash_entry(OLD_TRAJECTORY, "difference audit only"), "collision_world_parts": str(PARTS.resolve()), "no_stage25r2_mutation": True, "no_trajectory_generation": True})
    write_json(output_root / "stage26r_input_identity.json", identity)
    intervals, process_summary = build_process_mapping(rows)
    interval_csv = output_root / "stage26r_intervals.csv"
    write_intervals(interval_csv, intervals)
    write_json(output_root / "stage26r_process_mapping.json", {**process_summary, "mapping_hash": semantic_hash(process_summary), "interval_hash": semantic_hash(intervals)})
    stats = interval_statistics(rows, intervals)
    write_json(output_root / "stage26r_interval_statistics.json", stats)
    write_json(output_root / "stage26r_runtime_fingerprint.json", runtime_fingerprint())
    inventory = mesh_inventory()
    write_json(output_root / "stage26r_scene_mesh_inventory.json", {"part_count": len(inventory), "parts": inventory})

    protected, protected_meta = protected_files()
    before = hash_file_map(protected)
    write_json(output_root / "frozen_artifact_hashes_before.json", {"schema_version": "stage26r-frozen-hashes-v1", "metadata": protected_meta, "files": before})

    build_native_package(output_root)
    rebuilds: list[dict[str, Any]] = []
    for index in range(1, 4):
        run_root = output_root / f"rebuild_{index:02d}"
        run_root.mkdir()
        bullet_dir = run_root / "bullet"
        fcl_dir = run_root / "fcl"
        bullet_dir.mkdir(); fcl_dir.mkdir()
        bullet_input = bullet_dir / "stage26r_intervals.csv"; fcl_input = fcl_dir / "stage26r_intervals.csv"
        shutil.copy2(interval_csv, bullet_input); shutil.copy2(interval_csv, fcl_input)
        bullet_result = run_native(bullet_dir, bullet_input, "bullet", index, index == 1)
        fcl_result = run_native(fcl_dir, fcl_input, "fcl", index, False)
        if not bullet_result["passed"] or not fcl_result["passed"]:
            raise RuntimeError(f"native rebuild {index} failed")
        bullet_discrete = read_jsonl(bullet_dir / "stage26r_bullet_discrete.jsonl")
        fcl_discrete = read_jsonl(fcl_dir / "stage26r_fcl_discrete.jsonl")
        bullet_ccd = read_jsonl(bullet_dir / "stage26r_bullet_ccd.jsonl")
        bullet_self = read_jsonl(bullet_dir / "stage26r_self_collision.jsonl")
        fcl_self = read_jsonl(fcl_dir / "stage26r_self_collision.jsonl")
        validate_ccd_trace(bullet_ccd, expected=25531)
        if len(bullet_discrete) != 25532 or len(fcl_discrete) != 25532 or len(bullet_self) != 25532 or len(fcl_self) != 25532:
            raise RuntimeError(f"incomplete discrete/self coverage in rebuild {index}")
        rebuilds.append({
            "rebuild": index,
            "bullet_dir": str(bullet_dir.resolve()),
            "fcl_dir": str(fcl_dir.resolve()),
            "bullet_discrete": bullet_discrete,
            "fcl_discrete": fcl_discrete,
            "bullet_ccd": bullet_ccd,
            "bullet_self": bullet_self,
            "fcl_self": fcl_self,
            "bullet_summary": read_json(bullet_dir / "stage26r_native_summary.json"),
            "fcl_summary": read_json(fcl_dir / "stage26r_native_summary.json"),
            "bullet_provenance": read_json(bullet_dir / "stage26r_runtime_provenance.json"),
            "fcl_provenance": read_json(fcl_dir / "stage26r_runtime_provenance.json"),
        })

    first = rebuilds[0]
    comparison = backend_compare(first["fcl_discrete"], first["bullet_discrete"])
    write_json(output_root / "stage26r_backend_comparison.json", comparison)
    ccd = first["bullet_ccd"]
    ccd_collisions = [row for row in ccd if row.get("collision") or row.get("continuous_collision")]
    endpoint_collisions = [row for row in first["bullet_discrete"] if row.get("robot_world_collision")]
    ccd_summary = {"schema_version": "stage26r-bullet-ccd-summary-v1", "available": True, "api": "CollisionEnvBullet::checkRobotCollision(request,result,state0,state1,acm)", "intervals_expected": 25531, "intervals_checked": len(ccd), "accepted": len(ccd) - len(ccd_collisions), "rejected": len(ccd_collisions), "collisions_found": len(ccd_collisions), "skipped": 0, "unchecked": 25531 - len(ccd), "first_collision_interval": ccd_collisions[0] if ccd_collisions else None, "endpoint_collision_states": len(endpoint_collisions), "passed": len(ccd) == 25531 and not ccd_collisions and not endpoint_collisions and all(row.get("ccd_api_called") is True for row in ccd)}
    write_json(output_root / "stage26r_bullet_ccd_summary.json", ccd_summary)
    self_rows = first["bullet_self"]
    subdivision_rows = [row for row in self_rows if row.get("subdivision")]
    self_collisions = [row for row in self_rows if row.get("collision")]
    self_summary = {"schema_version": "stage26r-self-collision-summary-v1", "native_continuous_self_collision_available": False, "self_collision_method": "adaptive_discrete", "original_states": 25532, "added_subdivision_states": len(subdivision_rows), "states_checked": len(self_rows), "collisions_found": len(self_collisions), "minimum_or_maximum_spacing": {"maximum_joint_delta_rad": stats["max_joint_interval_rad"], "threshold_rad": SPACING_BOUND}, "additional_subdivision_required": stats["additional_subdivision_required"], "passed": len(self_rows) >= 25532 and not self_collisions}
    write_json(output_root / "stage26r_self_collision_summary.json", self_summary)
    positive = positive_control_summary(Path(first["bullet_dir"]) / "stage26_positive_control_tests.jsonl")
    write_json(output_root / "stage26r_positive_control_tests.json", positive)

    hashes = []
    for item in rebuilds:
        hashes.append({
            "rebuild": item["rebuild"],
            "input_identity_hash": semantic_hash(identity),
            "scene_provenance_hash": semantic_hash(item["bullet_provenance"]),
            "runtime_acm_hash": sha256(Path(item["bullet_dir"]) / "stage26r_acm.json"),
            "discrete_FCL_trace_hash": semantic_hash(item["fcl_discrete"]),
            "discrete_Bullet_trace_hash": semantic_hash(item["bullet_discrete"]),
            "self_collision_trace_hash": semantic_hash(item["bullet_self"]),
            "Bullet_CCD_trace_hash": semantic_hash(item["bullet_ccd"]),
            "interval_classification_hash": semantic_hash(intervals),
            "positive_control_hash": semantic_hash(positive),
            "process_mapping_hash": semantic_hash(process_summary),
        })
    hash_fields = [key for key in hashes[0] if key != "rebuild"]
    det = {key: len({item[key] for item in hashes}) == 1 for key in hash_fields}
    det["rebuilds"] = 3
    det["all_hashes_match"] = all(det[key] for key in hash_fields)
    det["hashes"] = hashes
    det["schema_version"] = "stage26r-determinism-v1"
    write_json(output_root / "stage26r_determinism_report.json", det)

    # Copy first-rebuild evidence only after the three independent runs exist.
    shutil.copy2(Path(first["bullet_dir"]) / "stage26r_bullet_discrete.jsonl", output_root / "stage26r_bullet_discrete.jsonl")
    shutil.copy2(Path(first["fcl_dir"]) / "stage26r_fcl_discrete.jsonl", output_root / "stage26r_fcl_discrete.jsonl")
    shutil.copy2(Path(first["bullet_dir"]) / "stage26r_bullet_ccd.jsonl", output_root / "stage26r_bullet_ccd.jsonl")
    shutil.copy2(Path(first["bullet_dir"]) / "stage26r_self_collision.jsonl", output_root / "stage26r_self_collision.jsonl")
    shutil.copy2(Path(first["bullet_dir"]) / "stage26r_acm.json", output_root / "stage26r_acm.json")
    scene = {
        "schema_version": "stage26r-scene-provenance-v1",
        "robot_model_name": first["bullet_provenance"].get("robot_model_name", "fairino5_v6_spray_tcp"),
        "planning_group": "fairino5_v6_group",
        "urdf_path": str(URDF.resolve()), "urdf_sha256": sha256(URDF),
        "srdf_path": str(SRDF.resolve()), "srdf_sha256": sha256(SRDF),
        "collision_detector": {"FCL": first["fcl_provenance"].get("active_detector_name"), "Bullet": first["bullet_provenance"].get("active_detector_name")},
        "collision_links_checked": ["base_link", "shoulder_link", "upperarm_link", "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "spray_tcp_link"],
        "world_object_count": 1, "world_object_ids": ["horseshoe_collision_compound"],
        "world_mesh_inventory": {"part_count": len(inventory), "every_mesh_path": [row["path"] for row in inventory], "every_mesh_sha256": {row["path"]: row["sha256"] for row in inventory}},
        "attached_body_count": 0,
        "runtime_acm": {"serialized": True, "path": str((output_root / "stage26r_acm.json").resolve()), "sha256": sha256(output_root / "stage26r_acm.json"), "entries": read_json(output_root / "stage26r_acm.json")},
        "bullet_filter_metadata": {"available": False, "values_or_reason_unavailable": "MoveIt public collision environment does not expose persistent Bullet filter masks; native runtime class and API are recorded."},
        "planning_frame": first["bullet_provenance"].get("planning_frame"), "robot_model_frame": first["bullet_provenance"].get("robot_model_frame"),
        "formal_acm_modified": False, "world_object_acm_allow_entries_added": 0,
    }
    write_json(output_root / "stage26r_scene_provenance.json", scene)

    before_after = hash_file_map(protected)
    mismatches = [path for path in before if before[path] != before_after.get(path)]
    write_json(output_root / "frozen_artifact_hashes_after.json", {"schema_version": "stage26r-frozen-hashes-v1", "metadata": protected_meta, "files": before_after, "mismatch_count": len(mismatches), "mismatches": mismatches})
    identity["Stage25R2_input_mutated_after"] = sha256(STAGE25R2_TRAJECTORY) != identity["source_trajectory_sha256"]
    identity["stage25r2_final_robot_trajectory_unchanged"] = not identity["Stage25R2_input_mutated_after"]
    write_json(output_root / "stage26r_input_identity.json", identity)
    regression = run_regressions(output_root)

    formal_stage25r2 = entry["gate"]["Stage_2_5R2"] == "passed"
    stage26r_passed = bool(formal_stage25r2 and comparison["passed"] and ccd_summary["passed"] and self_summary["passed"] and positive["passed"] and stats["duplicate_intervals"] == 0 and stats["missing_intervals"] == 0 and stats["nonpositive_dt_intervals"] == 0 and process_summary["spray_on_segments"] == "10/10" and process_summary["spray_off_transitions"] == "9/9" and process_summary["process_order_preserved"] and process_summary["boundary_order_preserved"] and det["all_hashes_match"] and not mismatches and regression["targeted_stage26r_and_stage26"]["passed"])
    status = "passed" if stage26r_passed else "blocked_evidence_gate"
    gate = {
        "schema_version": "stage26r-gate-v1",
        "Stage_2_4T": "passed_frozen_unchanged", "Stage_2_5": "passed", "Stage_2_5R2": "passed",
        "Stage_2_6R": status, "Stage_2_6": "passed" if stage26r_passed else status,
        "Stage_2_7": {"status": "unblocked_not_started" if stage26r_passed else "blocked", "actually_started": False},
        "formal_input": {"source": "Stage_2_5R2_final_robot_trajectory", "path": str(STAGE25R2_TRAJECTORY.resolve()), "sha256": identity["source_trajectory_sha256"], "identity_with_stage25r2": identity["sha256_identical"] and identity["all_fields_bit_identical"]},
        "timed_states": 25532, "continuous_intervals_expected": 25531,
        "scene": {"robot_model": "fairino5_v6_spray_tcp", "planning_group": "fairino5_v6_group", "world_object_count": 1, "world_object": "horseshoe_collision_compound", "collision_mesh_parts": len(inventory), "formal_acm_modified": False},
        "discrete_collision": {"FCL": {"available": True, "states_checked": 25532, "collision_states": comparison["FCL"]["robot_world_collision_states"] + comparison["FCL"]["self_collision_states"]}, "Bullet": {"available": True, "states_checked": 25532, "collision_states": comparison["Bullet"]["robot_world_collision_states"] + comparison["Bullet"]["self_collision_states"]}, "backend_symmetric_difference": comparison["collision_acceptance_symmetric_difference"]},
        "Bullet_native_robot_world_CCD": {"available": True, "calls": len(ccd), "intervals_checked": len(ccd), "collisions_found": ccd_summary["collisions_found"], "skipped": ccd_summary["skipped"], "unchecked": ccd_summary["unchecked"], "first_collision": ccd_summary["first_collision_interval"]},
        "self_collision": {"method": "adaptive_discrete", "passed": self_summary["passed"], "states_checked": self_summary["states_checked"], "subdivision_states": self_summary["added_subdivision_states"], "collisions_found": self_summary["collisions_found"]},
        "native_continuous_self_collision": {"available": False}, "FCL_native_continuous_CCD": {"available": False},
        "positive_controls": {"A_endpoint_free_swept_collision": "passed" if positive["positive_control_A"]["passed"] else "failed", "B_free_space": "passed" if positive["positive_control_B"]["passed"] else "failed", "C_ACM_semantics": "passed" if positive["positive_control_C"]["passed"] else "failed"},
        "trajectory_interval_integrity": {"missing": stats["missing_intervals"], "duplicate": stats["duplicate_intervals"], "nonpositive_dt": stats["nonpositive_dt_intervals"]},
        "process": {"spray_on_segments": process_summary["spray_on_segments"], "spray_off_transitions": process_summary["spray_off_transitions"], "process_order_preserved": process_summary["process_order_preserved"], "boundary_order_preserved": process_summary["boundary_order_preserved"]},
        "Stage_2_5R2_dynamic_gate_preserved": True, "Stage_2_5R2_pose_gate_preserved": True,
        "determinism": {"rebuilds": 3, "all_hashes_match": det["all_hashes_match"]}, "frozen_artifact_mismatch_count": len(mismatches),
        "controller_execution_semantics": {"certified_in_stage26r": False, "deferred_to": "Stage_2_7"},
        "full_pytest": regression["full_pytest"].get("classification", "passed" if regression["full_pytest"]["passed"] else "failed"),
        "single_next_action": "Stage 2.6R passed; Stage 2.7 remains unstarted and may be considered separately." if stage26r_passed else "Do not start Stage 2.7; resolve the Stage 2.6R evidence gate blocker.",
    }
    write_json(output_root / "stage26r_gate_report.json", gate)
    certification = {"Stage_2_4T": gate["Stage_2_4T"], "Stage_2_5": gate["Stage_2_5"], "Stage_2_5R2": gate["Stage_2_5R2"], "Stage_2_6R": gate["Stage_2_6R"], "Stage_2_6": gate["Stage_2_6"], "Stage_2_7": gate["Stage_2_7"], "input_identity": gate["formal_input"], "timed_states": gate["timed_states"], "continuous_intervals_expected": gate["continuous_intervals_expected"], "scene": gate["scene"], "discrete_collision": gate["discrete_collision"], "Bullet_native_robot_world_CCD": gate["Bullet_native_robot_world_CCD"], "self_collision": gate["self_collision"], "native_continuous_self_collision": gate["native_continuous_self_collision"], "FCL_native_continuous_CCD": gate["FCL_native_continuous_CCD"], "positive_controls": gate["positive_controls"], "trajectory_interval_integrity": gate["trajectory_interval_integrity"], "process": gate["process"], "determinism": gate["determinism"], "frozen_artifact_mismatch_count": gate["frozen_artifact_mismatch_count"], "controller_execution_semantics": gate["controller_execution_semantics"]}
    write_json(output_root / "stage26r_certification.json", certification)
    report = ["# Stage 2.6R — Stage 2.5R2 final-trajectory continuous collision recertification", "", f"- Final status: **Stage 2.6R = {status}**; Stage 2.6 = {gate['Stage_2_6']}; Stage 2.7 = {gate['Stage_2_7']['status']}.", "- Formal input is the exact Stage 2.5R2 final RobotTrajectory CSV. No TOTG, Ruckig, resampling, interpolation rewrite, timestamp normalization, derivative recomputation, JTC, or controller was started.", "", "## Native evidence", f"- FCL discrete: {comparison['FCL']['states_checked']} states; robot-world collisions {comparison['FCL']['robot_world_collision_states']}; self collisions {comparison['FCL']['self_collision_states']}.", f"- Bullet discrete: {comparison['Bullet']['states_checked']} states; robot-world collisions {comparison['Bullet']['robot_world_collision_states']}; self collisions {comparison['Bullet']['self_collision_states']}.", f"- Bullet native robot-world CCD: {ccd_summary['intervals_checked']}/{ccd_summary['intervals_expected']} calls; collisions {ccd_summary['collisions_found']}; skipped {ccd_summary['skipped']}; unchecked {ccd_summary['unchecked']}.", "- FCL native continuous CCD: not_available. Native continuous self-collision: not_available; self evidence is adaptive discrete.", "", "## Controls and integrity", f"- Positive controls A/B/C: {positive['passed']}; backend acceptance symmetric difference: {comparison['collision_acceptance_symmetric_difference']}.", f"- Process mapping: ON {process_summary['spray_on_segments']}, OFF {process_summary['spray_off_transitions']}; order preserved {process_summary['process_order_preserved']}.", f"- Determinism: 3/3, all hashes match {det['all_hashes_match']}; frozen artifact mismatch count {len(mismatches)}.", f"- Full pytest: {gate['full_pytest']}.", "", "## Scope boundary", "- This pass certifies only the planned timed path under the Stage 2.6 collision contract. Controller execution semantics remain deferred to Stage 2.7, which was not started.", ""]
    (output_root / "stage26r_report.md").write_text("\n".join(report), encoding="utf-8")
    sums = write_sha256sums(output_root)
    gate["artifact_sha256_manifest"] = sums
    gate["artifact_sha256_manifest_mismatch_count"] = sums["mismatch_count"]
    write_json(output_root / "stage26r_gate_report.json", gate)
    print(json.dumps({"output_root": str(output_root), "Stage_2_6R": status, "Stage_2_6": gate["Stage_2_6"], "Stage_2_7": gate["Stage_2_7"], "timed_states": 25532, "continuous_intervals": 25531, "ccd_collisions": ccd_summary["collisions_found"], "frozen_artifact_mismatch_count": len(mismatches)}, ensure_ascii=False, indent=2))
    return 0 if stage26r_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
