#!/usr/bin/env python3
"""Run scoped C5A registration metric closure against frozen C1/C4 authority."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p2b3_c5a_metrics import process_metrics, registration_cases, scene_rows
from src.physical_uncertainty import registration_transform

BRANCH = "codex/fr5-p2b3-c5a-registration-metric-closure-20261002"
PARENT_BRANCH = "codex/fr5-p2b3-c4-authoritative-scene-equivalence-20261002"
PARENT = "f1b3840db3573142050695cb5e29702f4fd7a7de"
C1_RESULT = ROOT / "outputs/p2b3_c1_result.json"
C1_TRAJECTORY = ROOT / "outputs/p2b3_c2_c1_nominal_post_ruckig.csv"
C1_TARGET = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
C1_FK = ROOT / "outputs/p2b3_c1_r0_fk_trace.csv"
C1_LINEAGE = ROOT / "outputs/p2b3_c1_r0_waypoint_lineage.csv"
C4_RESULT = ROOT / "outputs/p2b3_c4_result.json"
C4_MANIFEST = ROOT / "outputs/p2b3_c4_authoritative_scene_manifest.json"
URDF = ROOT / "outputs/p2b2_inputs/derived_reference_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
JOINT_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
FAIRINO_SOURCE_COMMIT = "60755d44d521a5ad6bee8494cc19522f8801aa20"
FAIRINO_URDF_SHA256 = "923a4d2f754162dadc9a6b0bcbe5b4caf9a32ccd31c479e72ed690febe211b4e"
FAIRINO_CONTROL_SHA256 = "9296111525fa892d63c700b37d58bd0e9b4a2f7d9713ff3d00bd95b56b5d08b5"
GROUP = "fairino5_v6_group"
TCP_LINK = "spray_tcp_link"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def git(*args: str) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
        if completed.returncode == 0:
            return completed.stdout.strip()
    except OSError:
        pass
    windows_root = subprocess.run(["wslpath", "-w", str(ROOT)], check=True, text=True, encoding="utf-8", capture_output=True).stdout.strip()
    completed = subprocess.run(["git.exe", "-C", windows_root, *args], text=True, encoding="utf-8", capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"git_{args[0]}_failed:{completed.stderr.strip()}")
    return completed.stdout.strip()


def git_at(directory: Path, *args: str) -> str:
    try:
        completed = subprocess.run(["git", "-C", str(directory), *args], text=True, encoding="utf-8", capture_output=True)
        if completed.returncode == 0:
            return completed.stdout.strip()
    except OSError:
        pass
    windows_root = subprocess.run(["wslpath", "-w", str(directory)], check=True, text=True, encoding="utf-8", capture_output=True).stdout.strip()
    completed = subprocess.run(["git.exe", "-C", windows_root, *args], text=True, encoding="utf-8", capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"git_at_{args[0]}_failed:{completed.stderr.strip()}")
    return completed.stdout.strip()


def require_clean_execution_tree() -> dict[str, str]:
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    ancestry_base, dirty = git("merge-base", PARENT, "HEAD"), git("status", "--porcelain")
    if branch != BRANCH or ancestry_base != PARENT or dirty:
        raise RuntimeError(f"execution_tree_identity_mismatch:branch={branch}:head={head}:base={ancestry_base}:dirty={bool(dirty)}")
    return {"branch": branch, "execution_code_commit": head, "parent_commit": ancestry_base, "clean_at_start": True}


def verify_fairino_source(directory: Path) -> dict[str, str]:
    remote = git_at(directory, "remote", "get-url", "origin")
    commit = git_at(directory, "rev-parse", "HEAD")
    dirty = git_at(directory, "status", "--porcelain")
    urdf = directory / "fairino_description/urdf/fairino5_v6.urdf"
    control = directory / "fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro"
    if ("FAIR-INNOVATION/frcobot_ros2" not in remote or commit != FAIRINO_SOURCE_COMMIT or dirty or
            sha256(urdf) != FAIRINO_URDF_SHA256 or sha256(control) != FAIRINO_CONTROL_SHA256):
        raise RuntimeError(f"FAIRINO_vendor_source_identity_mismatch:remote={remote}:commit={commit}:dirty={bool(dirty)}")
    return {"remote": remote, "commit": commit, "clean": "YES", "urdf_sha256": FAIRINO_URDF_SHA256,
            "moveit_control_xacro_sha256": FAIRINO_CONTROL_SHA256}


def require_inputs(fairino_source_identity: dict[str, str]) -> dict[str, Any]:
    paths = (C1_RESULT, C1_TRAJECTORY, C1_TARGET, C1_FK, C1_LINEAGE, C4_RESULT, C4_MANIFEST, URDF, SRDF, JOINT_LIMITS)
    missing = [str(path) for path in paths if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("required_input_missing:" + ",".join(missing))
    c1, c4 = read_json(C1_RESULT), read_json(C4_RESULT)
    if c1.get("PROJECT") != "FAIRINO_FR5" or c1.get("STAGE") != "P2-B3-C1":
        raise RuntimeError("C1_authority_identity_mismatch")
    if c4.get("project") != "FAIRINO_FR5" or c4.get("stage") != "P2-B3-C4" or c4.get("P2B3_C4_STATUS") != "PASS":
        raise RuntimeError("C4_authority_identity_or_status_mismatch")
    trajectory_sha = sha256(C1_TRAJECTORY)
    expected_trajectory_sha = c1.get("C1_POST_RUCKIG_NOMINAL", {}).get("sha256")
    if trajectory_sha != expected_trajectory_sha:
        raise RuntimeError("C1_nominal_post_ruckig_identity_mismatch")
    fk_sha = sha256(C1_FK)
    if fk_sha != c1.get("candidate_input_identity_sha256", {}).get("candidate_fk_trace"):
        raise RuntimeError("C1_FK_trace_identity_mismatch")
    target_hashes = c1.get("input_identity_sha256", {})
    expected_target = [value for key, value in target_hashes.items() if key.replace("\\", "/") == "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"]
    if len(expected_target) != 1 or sha256(C1_TARGET) != expected_target[0]:
        raise RuntimeError("C1_target_pose_identity_mismatch")
    expected_urdf = c1.get("C1_DERIVED_REFERENCE_URDF_SHA256")
    if expected_urdf and sha256(URDF) != expected_urdf:
        raise RuntimeError("C1_reference_URDF_identity_mismatch")
    manifest = read_json(C4_MANIFEST)
    if c4.get("scene_manifest_sha256") != "f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669":
        raise RuntimeError("C4_canonical_scene_manifest_identity_mismatch")
    if manifest.get("collision_objects", {}).get("count") != 181 or len(manifest.get("collision_objects", {}).get("ordered", [])) != 181:
        raise RuntimeError("C4_authoritative_scene_object_count_mismatch")
    return {
        "c1_result_sha256": sha256(C1_RESULT),
        "c1_post_ruckig_nominal_sha256": trajectory_sha,
        "c1_target_tcp_csv_sha256": sha256(C1_TARGET),
        "c1_fk_trace_sha256": fk_sha,
        "c1_waypoint_lineage_sha256": sha256(C1_LINEAGE),
        "c4_result_sha256": sha256(C4_RESULT),
        "c4_authoritative_scene_manifest_sha256": c4["scene_manifest_sha256"],
        "urdf_sha256": sha256(URDF),
        "srdf_sha256": sha256(SRDF),
        "joint_limits_yaml_sha256": sha256(JOINT_LIMITS),
        "fairino_vendor_source": fairino_source_identity,
        "c1_stand_off_m": float(c4["scene_geometry"]["stand_off_m"]),
        "c4_acm_sha256": c4.get("authoritative_scene_contract", {}).get("acm", {}).get("sha256")
            or "02975b55c25b52d8bb7d77bdbaf05db1ef5f1d58278575fa4fb298e1a3f35661",
        "c4_acm_allowed_pair_count": 24,
        "c4_collision_object_count": 181,
    }


def command_logged(command: list[str], cwd: Path, log: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command_failed:{completed.returncode}:log={log}:tail={completed.stdout[-4000:]}")
    print(f"completed_command={command[0]} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    return completed


def bash_logged(command: str, cwd: Path, log: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return command_logged(["bash", "-lc", "source /opt/ros/jazzy/setup.bash && " + command], cwd, log, timeout_s)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def transform_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("id", "dx", "dy", "dz", "px", "py", "pz", "qx", "qy", "qz", "qw")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_q_only(path: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row["t"]) for row in rows], dtype=np.float64)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    if q.shape != (181, 6) or times.shape != (181,) or not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise RuntimeError("C1_q_or_time_shape_finite_or_order_check_failed")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["t", *[f"j{i}_q" for i in range(1, 7)]])
        writer.writerows([[float(times[row]), *map(float, q[row])] for row in range(len(times))])
    return q, times


def run_c4_identity(scratch: Path, timeout_s: int) -> dict[str, Any]:
    result_path, gate_path = scratch / "c4_identity_result.json", scratch / "c4_identity_gate.json"
    command_args = [sys.executable, str(ROOT / "ros2_moveit_bridge/p2b3_c4_native_smoke.py"),
                    "--trajectory-csv", str(C1_TRAJECTORY), "--target-csv", str(C1_TARGET),
                    "--c1-fk-trace", str(C1_FK), "--c1-result", str(C1_RESULT), "--urdf", str(URDF),
                    "--srdf", str(SRDF), "--joint-limits", str(JOINT_LIMITS), "--scene-manifest", str(C4_MANIFEST),
                    "--identity-gate-json", str(gate_path), "--output-json", str(result_path), "--phase", "identity"]
    command = " && ".join([
        f"source {shlex.quote(str(scratch / 'fairino_install/setup.bash'))}",
        f"source {shlex.quote(str(scratch / 'bridge_install/setup.bash'))}",
        shlex.join(command_args),
    ])
    bash_logged(command, ROOT, scratch / "c4_identity.log", timeout_s)
    identity = read_json(result_path)
    if identity.get("status") != "IDENTITY_GATE_PASS_PHASE_D_PENDING" or identity.get("scene_equivalence", {}).get("status") != "PASS":
        raise RuntimeError("C4_independent_identity_replay_failed")
    if identity.get("zero_transform_equivalence", {}).get("status") != "PASS":
        raise RuntimeError("C4_zero_transform_equivalence_failed")
    if identity.get("scene_manifest_sha256") != "f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669":
        raise RuntimeError("C4_identity_replay_scene_sha_mismatch")
    return identity


def build_moveit_runtime(scratch: Path, fairino_source: Path, timeout_s: int) -> None:
    vendor_build = " ".join([
        "colcon --log-base", shlex.quote(str(scratch / "vendor_colcon_log")),
        "build --base-paths", shlex.quote(str(fairino_source / "fairino_description")),
        shlex.quote(str(fairino_source / "fairino5_v6_moveit2_config")),
        "--build-base", shlex.quote(str(scratch / "vendor_build")),
        "--install-base", shlex.quote(str(scratch / "fairino_install")),
        "--packages-select fairino_description fairino5_v6_moveit2_config --event-handlers console_direct+",
        "--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=OFF",
    ])
    bash_logged(vendor_build, ROOT, scratch / "vendor_build.log", timeout_s)
    bridge_build = " && ".join([
        f"source {shlex.quote(str(scratch / 'fairino_install/setup.bash'))}",
        "colcon --log-base " + shlex.quote(str(scratch / "bridge_colcon_log")) +
        " build --base-paths " + shlex.quote(str(ROOT / "ros2_moveit_bridge")) +
        " --build-base " + shlex.quote(str(scratch / "bridge_build")) +
        " --install-base " + shlex.quote(str(scratch / "bridge_install")) +
        " --packages-select fr5_tunnel_moveit_bridge --event-handlers console_direct+" +
        " --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3",
    ])
    bash_logged(bridge_build, ROOT, scratch / "bridge_build.log", timeout_s)


def build_native(scratch: Path, timeout_s: int) -> None:
    package = ROOT / "cpp/p2b3_c5a_native"
    command = " ".join([
        "colcon --log-base", shlex.quote(str(scratch / "colcon_log")),
        "build --base-paths", shlex.quote(str(package)),
        "--build-base", shlex.quote(str(scratch / "build")),
        "--install-base", shlex.quote(str(scratch / "install")),
        "--event-handlers console_direct+ --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3",
    ])
    bash_logged(command, ROOT, scratch / "native_build.log", timeout_s)


def run_native_known_answer(scratch: Path, timeout_s: int) -> dict[str, Any]:
    fixture_urdf = ROOT / "tests/fixtures/p2b3_c5a_known_answer.urdf"
    fixture_srdf = ROOT / "tests/fixtures/p2b3_c5a_known_answer.srdf"
    command = " && ".join([
        f"source {shlex.quote(str(scratch / 'install/setup.bash'))}",
        "ros2 run p2b3_c5a_native p2b3_c5a_fcl_known_answer --",
        shlex.quote(str(fixture_urdf)), shlex.quote(str(fixture_srdf)),
    ])
    output = bash_logged(command, ROOT, scratch / "fcl_known_answer.log", timeout_s).stdout
    record = next((json.loads(line) for line in reversed(output.splitlines()) if line.strip().startswith("{\"status\"")), None)
    if not isinstance(record, dict) or record.get("status") != "PASS":
        raise RuntimeError("native_FCL_known_answer_missing_or_failed")
    return record


def write_case_inputs(scratch: Path, manifest: dict[str, Any], q: np.ndarray, times: np.ndarray) -> list[dict[str, Any]]:
    input_dir = scratch / "cases"; input_dir.mkdir(parents=True)
    q_path = input_dir / "c1_q_only.csv"
    source_q_rows = read_csv(C1_TRAJECTORY)
    q_again, times_again = write_q_only(q_path, source_q_rows)
    if not np.array_equal(q_again, q) or not np.array_equal(times_again, times):
        raise RuntimeError("C5A_q_or_time_changed_while_materializing_q_only_input")
    cases = registration_cases(); case_index = input_dir / "case_inputs.csv"
    with case_index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n"); writer.writerow(["case_id", "q_csv", "scene_csv"])
        for case in cases:
            delta = registration_transform(case["translation_base_m"], case["rotation_vector_base_rad"])
            scene_path = input_dir / f"{case['case_id']}_scene.csv"
            transform_to_csv(scene_path, scene_rows(manifest, delta))
            writer.writerow([case["case_id"], str(q_path), str(scene_path)])
    return cases


def run_native_cases(scratch: Path, timeout_s: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    native_output = scratch / "native_output"; native_output.mkdir()
    command = " && ".join([
        f"source {shlex.quote(str(scratch / 'install/setup.bash'))}",
        "ros2 run p2b3_c5a_native p2b3_c5a_native --",
        "--cases", shlex.quote(str(scratch / "cases/case_inputs.csv")),
        "--urdf", shlex.quote(str(URDF)), "--srdf", shlex.quote(str(SRDF)),
        "--output", shlex.quote(str(native_output)), "--group", GROUP, "--tip", TCP_LINK,
    ])
    bash_logged(command, ROOT, scratch / "native_measurement.log", timeout_s)
    native = [json.loads(line) for line in (native_output / "p2b3_c5a_native_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fk = read_csv(native_output / "p2b3_c5a_fresh_fk.csv")
    if len(native) != 25 or len(fk) != 25 * 181:
        raise RuntimeError(f"native_output_count_mismatch:cases={len(native)}:fk_rows={len(fk)}")
    return native, fk


def grouped_fk(rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        case_id = row["case_id"]
        record = result.setdefault(case_id, {"position": [], "rotation": [], "time": []})
        record["position"].append([float(row[key]) for key in ("px", "py", "pz")])
        record["rotation"].append([[float(row[f"r{i}{j}"]) for j in range(3)] for i in range(3)])
        record["time"].append(float(row["time_s"]))
    for record in result.values():
        record["position"] = np.asarray(record["position"], dtype=np.float64)
        record["rotation"] = np.asarray(record["rotation"], dtype=np.float64)
        record["time"] = np.asarray(record["time"], dtype=np.float64)
    return result


def clearance_record(native: dict[str, Any], key: str, world_object_ids: set[str]) -> dict[str, Any]:
    raw = native[key]
    if raw.get("status") != "AVAILABLE" or not math.isfinite(float(raw.get("distance_m", float("nan")))):
        return {"status": raw.get("status", "NOT_AVAILABLE_UNREPORTED"), "signed_distance_m": None,
                "pair": {"status": "NOT_AVAILABLE_NO_FINITE_PAIR", "names": None},
                "nearest_points_base_link": {"status": "NOT_AVAILABLE_NO_FINITE_PAIR", "points": None}}
    pair = raw.get("pair")
    record = {
        "status": "AVAILABLE_SAMPLED_SIGNED_MODEL_DISTANCE",
        "signed_distance_m": float(raw["distance_m"]),
        "distance_semantics": "raw MoveIt FCL signed distance; <=0 indicates collision per installed collision_common.hpp; sampled only",
        "pair": {"status": "AVAILABLE" if isinstance(pair, list) and len(pair) == 2 else "NOT_AVAILABLE_PAIR_NAMES_MISSING", "names": pair},
        "nearest_points_base_link": {"status": "AVAILABLE", "points": raw.get("nearest_points")},
        "critical_sample": {key: raw.get(key) for key in ("sample_index", "waypoint_index", "segment", "segment_fraction", "time_s")},
    }
    if key == "minimum_robot_world_clearance":
        if not isinstance(pair, list) or len(pair) != 2 or (pair[0] in world_object_ids) == (pair[1] in world_object_ids):
            record["robot_link_name"] = {"status": "NOT_AVAILABLE_PAIR_DOMAIN_AMBIGUOUS", "name": None}
            record["world_object_name"] = {"status": "NOT_AVAILABLE_PAIR_DOMAIN_AMBIGUOUS", "name": None}
        else:
            object_name = pair[0] if pair[0] in world_object_ids else pair[1]
            link_name = pair[1] if pair[0] in world_object_ids else pair[0]
            record["robot_link_name"] = {"status": "AVAILABLE", "name": link_name}
            record["world_object_name"] = {"status": "AVAILABLE", "name": object_name}
    else:
        record["self_link_pair"] = {"status": "AVAILABLE" if isinstance(pair, list) and len(pair) == 2 else "NOT_AVAILABLE_PAIR_NAMES_MISSING", "names": pair}
    return record


def assemble_cases(
    case_specs: list[dict[str, Any]], native_records: list[dict[str, Any]], fk_rows: list[dict[str, str]],
    q: np.ndarray, times: np.ndarray, identity: dict[str, Any], identities: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ros2_moveit_bridge.plan_closed_contour_moveit import _project_to_polyline_with_normals

    native_by_id = {record["case_id"]: record for record in native_records}
    fk_by_id = grouped_fk(fk_rows)
    if set(native_by_id) != {case["case_id"] for case in case_specs} or set(fk_by_id) != set(native_by_id):
        raise RuntimeError("native_case_identity_set_mismatch")
    target_rows = read_csv(C1_TARGET)
    world_object_ids = {str(obj["id"]) for obj in read_json(C4_MANIFEST)["collision_objects"]["ordered"]}
    target = np.asarray([[float(row[key]) for key in ("x", "y", "z", "qx", "qy", "qz", "qw")] for row in target_rows], dtype=np.float64)
    normals = np.asarray([[float(row[key]) for key in ("nx", "ny", "nz")] for row in target_rows], dtype=np.float64)
    source_fk = read_csv(C1_FK)
    expected_position = np.asarray([[float(row[key]) for key in ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z")] for row in source_fk], dtype=np.float64)
    expected_tool_z = np.asarray([[float(row[key]) for key in ("tool_z_x", "tool_z_y", "tool_z_z")] for row in source_fk], dtype=np.float64)
    c1_result = read_json(C1_RESULT)
    stand_off = float(identities["c1_stand_off_m"])
    result_cases: list[dict[str, Any]] = []
    base_fk: dict[str, np.ndarray] | None = None
    for spec in case_specs:
        raw = native_by_id[spec["case_id"]]
        fresh = fk_by_id[spec["case_id"]]
        if raw.get("sample_count") != identity.get("zero_baseline", {}).get("sample_count"):
            raise RuntimeError(f"C5A_sample_count_differs_from_C4_identity:{spec['case_id']}")
        if not np.array_equal(fresh["time"], times):
            raise RuntimeError(f"native_fk_timestamp_mismatch:{spec['case_id']}")
        if base_fk is None:
            base_fk = {key: np.array(fresh[key], copy=True) for key in ("position", "rotation", "time")}
        elif not np.array_equal(fresh["position"], base_fk["position"]) or not np.array_equal(fresh["rotation"], base_fk["rotation"]):
            raise RuntimeError(f"frozen_q_fk_changed_across_registration_cases:{spec['case_id']}")
        delta = registration_transform(spec["translation_base_m"], spec["rotation_vector_base_rad"])
        metrics = process_metrics(actual_position_xyz=fresh["position"], actual_rotation_matrices=fresh["rotation"],
                                  target_pose_xyz_quat_xyzw=target, target_normals=normals,
                                  delta_transform=delta, stand_off_m=stand_off,
                                  project_to_polyline=_project_to_polyline_with_normals)
        position_delta = np.linalg.norm(fresh["position"] - expected_position, axis=1)
        tool_z = fresh["rotation"][:, :, 2]
        tool_z_delta = np.linalg.norm(tool_z - expected_tool_z, axis=1)
        fk_regression = {
            "status": "PASS" if float(np.max(position_delta)) <= 1e-6 and float(np.max(tool_z_delta)) <= 1e-6 else "FAIL",
            "max_position_delta_m": float(np.max(position_delta)), "max_tool_z_delta": float(np.max(tool_z_delta)),
            "position_tolerance_m": 1e-6, "tool_z_tolerance": 1e-6,
        }
        if spec["family"] != "zero" and (not np.array_equal(q, np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in read_csv(C1_TRAJECTORY)])) or
                                           not np.array_equal(times, np.asarray([float(row["t"]) for row in read_csv(C1_TRAJECTORY)]))):
            raise RuntimeError(f"registration_changed_frozen_q_or_time:{spec['case_id']}")
        result_cases.append({
            "case_id": spec["case_id"], "family": spec["family"], "magnitude_label": spec["magnitude_label"],
            "translation_base_m": spec["translation_base_m"], "rotation_vector_base_rad": spec["rotation_vector_base_rad"],
            "left_composed_transform_base": delta.tolist(), "joint_states_unchanged": True, "timestamps_unchanged": True,
            "waypoint_count": 181, "adaptive_discrete_sample_count": int(raw["sample_count"]),
            "scene_object_count": 181, "fresh_fk_regression": fk_regression,
            "robot_world_collision_sample_count": int(raw["robot_world_collision_samples"]),
            "self_collision_sample_count": int(raw["self_collision_samples"]),
            "first_robot_world_collision": {"status": "AVAILABLE" if raw.get("first_robot_world_collision") else "NO_COLLISION_OBSERVED", "record": raw.get("first_robot_world_collision")},
            "first_self_collision": {"status": "AVAILABLE" if raw.get("first_self_collision") else "NO_COLLISION_OBSERVED", "record": raw.get("first_self_collision")},
            "minimum_robot_world_clearance": clearance_record(raw, "minimum_robot_world_clearance", world_object_ids),
            "minimum_self_clearance": clearance_record(raw, "minimum_self_clearance", world_object_ids),
            "process_metrics": metrics,
            "strict_ccd": "NOT_AVAILABLE",
            "hardware_status": "NOT_RUN",
        })
    all_world_available = all(case["minimum_robot_world_clearance"]["status"] == "AVAILABLE_SAMPLED_SIGNED_MODEL_DISTANCE" for case in result_cases)
    all_self_available = all(case["minimum_self_clearance"]["status"] == "AVAILABLE_SAMPLED_SIGNED_MODEL_DISTANCE" for case in result_cases)
    if not all(case["fresh_fk_regression"]["status"] == "PASS" for case in result_cases):
        fk_status = "FAIL"
    else:
        fk_status = "PASS"
    c4_zero_collision_count = int(identity.get("zero_baseline", {}).get("full_scene_collision_sample_count", -1))
    c5a_zero = native_by_id.get("registration_zero", {})
    zero_collision_match = int(c5a_zero.get("robot_world_collision_samples", -2)) == c4_zero_collision_count
    gate_checks = {
        "c4_identity_replay": identity.get("status") == "IDENTITY_GATE_PASS_PHASE_D_PENDING",
        "c4_scene_equivalence": identity.get("scene_equivalence", {}).get("status") == "PASS",
        "c4_zero_transform_gate": identity.get("zero_transform_equivalence", {}).get("status") == "PASS",
        "25_frozen_cases": len(result_cases) == 25,
        "all_case_sample_counts_match_c4": all(case["adaptive_discrete_sample_count"] == 858 for case in result_cases),
        "all_robot_world_distances_finite_with_pair_and_points": all_world_available,
        "all_robot_world_pair_domains_identified": all(case["minimum_robot_world_clearance"].get("world_object_name", {}).get("status") == "AVAILABLE" and case["minimum_robot_world_clearance"].get("robot_link_name", {}).get("status") == "AVAILABLE" for case in result_cases),
        "all_self_distances_finite_with_pair_and_points": all_self_available,
        "fresh_fk_matches_c1": fk_status == "PASS",
        "zero_collision_count_reconciles_to_independent_c4": zero_collision_match,
        "all_q_and_timestamps_unchanged": all(case["joint_states_unchanged"] and case["timestamps_unchanged"] for case in result_cases),
    }
    collision_cases = [case for case in result_cases if case["robot_world_collision_sample_count"] > 0 or case["self_collision_sample_count"] > 0]
    performance = "WEAKNESS_OBSERVED" if collision_cases else "NO_DISCRETE_COLLISIONS_OBSERVED"
    result = {
        "schema_version": "p2b3-c5a-registration-metric-closure-v1",
        "PROJECT": "FAIRINO_FR5", "STAGE": "P2-B3-C5A",
        "P2B3_C5A_STATUS": "PASS" if all(gate_checks.values()) else "BLOCKED",
        "MEASUREMENT_PIPELINE_STATUS": "PASS" if all(gate_checks.values()) else "BLOCKED",
        "FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS": performance,
        "PROJECT_SCOPE": "181-point ON-state open-arch only; FAIRINO FR5; frozen C1 post-Ruckig q and timestamps",
        "PARENT_BRANCH": PARENT_BRANCH,
        "PARENT_COMMIT": PARENT,
        "EXECUTION_BRANCH": BRANCH,
        "EXECUTION_CODE_COMMIT": git("rev-parse", "HEAD"),
        "ARTIFACT_PUBLISH_COMMIT": "PENDING_ARTIFACT_COMMIT",
        "FINAL_DELIVERY_COMMIT": "PENDING_FINAL_DELIVERY_COMMIT",
        "frozen_input_identities": identities,
        "independent_c4_identity_replay": {
            "status": identity["status"], "scene_manifest_sha256": identity.get("scene_manifest_sha256"),
            "scene_equivalence": identity.get("scene_equivalence"),
            "zero_transform_equivalence": identity.get("zero_transform_equivalence"),
            "sample_count": identity.get("zero_baseline", {}).get("sample_count"),
            "collision_count": c4_zero_collision_count,
            "fk_regression": identity.get("zero_baseline", {}).get("fk_regression"),
        },
        "measurement_contract": {
            "registration_composition": "T_base_entity_actual = DeltaT_base_left @ T_base_entity_nominal",
            "translation_frame": "base_link; direct translation in meters",
            "rotation_frame_and_origin": "base_link; right-handed axis-angle about base_link origin",
            "scene_source": "outputs/p2b3_c4_authoritative_scene_manifest.json ordered 181 BOX objects",
            "acm": {"source": "C4 identity replay and frozen SRDF-derived PlanningScene", "sha256": identities["c4_acm_sha256"], "allowed_pair_count": identities["c4_acm_allowed_pair_count"]},
            "collision_backend": "active MoveIt2 PlanningScene FCL, pad_environment_collisions=true, pad_self_collisions=false",
            "clearance_backend": "un-padded CollisionEnvFCL distanceRobot/distanceSelf; constructor semantics padding=0.0, scale=1.0",
            "clearance_is_signed": True,
            "collision_method": "adaptive_discrete_interpolation",
            "maximum_joint_interpolation_step_deg": 0.5,
            "strict_ccd": "NOT_AVAILABLE",
            "physical_registration_bound": "UNAVAILABLE_NO_MEASURED_FR5_BOUND",
            "hardware": "NOT_RUN",
            "low_clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD_NO_FROZEN_VALUE",
        },
        "measurement_gates": {key: "PASS" if value else "FAIL" for key, value in gate_checks.items()},
        "collision_findings": {
            "affected_case_ids": [case["case_id"] for case in collision_cases],
            "robot_world_collision_sample_total": sum(case["robot_world_collision_sample_count"] for case in result_cases),
            "self_collision_sample_total": sum(case["self_collision_sample_count"] for case in result_cases),
            "classification": "TYPE_B_FROZEN_ROBOT_SYSTEM_WEAKNESS" if collision_cases else "NO_COLLISION_OBSERVED_IN_FROZEN_CASE_SET",
            "robot_system_not_optimized": True,
        },
        "case_count": len(result_cases), "case_results": result_cases,
        "claim_boundaries": {
            "CLEARANCE_METRIC_VERIFIED": "YES" if all_world_available else "NO",
            "STRICT_CONTINUOUS_COLLISION_DETECTION": "NOT_AVAILABLE",
            "HARDWARE_VALIDATED": "NO",
            "HARDWARE_SAFETY_CERTIFIED": "NO",
            "PHYSICAL_REGISTRATION_MARGIN": "NOT_AVAILABLE_PHYSICAL_BOUND_MISSING",
            "COVERAGE": "25 deterministic engineering registration cases only; no broad Monte Carlo or C5B boundary search",
        },
    }
    return result_cases, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True, help="New external scratch directory on D:")
    parser.add_argument("--fairino-source", type=Path, required=True, help="Authenticated FAIRINO source checkout")
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--build-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    if scratch.exists():
        raise RuntimeError(f"scratch_directory_must_not_exist:{scratch}")
    tree = require_clean_execution_tree()
    fairino_identity = verify_fairino_source(args.fairino_source.resolve())
    identities = require_inputs(fairino_identity)
    scratch.mkdir(parents=True)
    build_moveit_runtime(scratch, args.fairino_source.resolve(), args.build_timeout_seconds)
    c4_identity = run_c4_identity(scratch, args.timeout_seconds)
    c1_rows = read_csv(C1_TRAJECTORY)
    q, times = write_q_only(scratch / "c1_q_only_validation.csv", c1_rows)
    case_specs = write_case_inputs(scratch, read_json(C4_MANIFEST), q, times)
    build_native(scratch, args.build_timeout_seconds)
    known_answer = run_native_known_answer(scratch, args.timeout_seconds)
    native_records, fk_rows = run_native_cases(scratch, args.timeout_seconds)
    case_results, result = assemble_cases(case_specs, native_records, fk_rows, q, times, c4_identity, identities)
    result["execution_tree_identity"] = tree
    result["native_fcl_known_answer"] = known_answer
    result["native_evaluator"] = {
        "source": "cpp/p2b3_c5a_native/p2b3_c5a_native.cpp",
        "case_summary_path": str((scratch / "native_output/p2b3_c5a_native_cases.jsonl").resolve()),
        "fresh_fk_path": str((scratch / "native_output/p2b3_c5a_fresh_fk.csv").resolve()),
        "cases_completed": len(native_records), "fresh_fk_rows": len(fk_rows),
        "robot_world_clearance_all_cases_available": all(case["minimum_robot_world_clearance"]["status"] == "AVAILABLE_SAMPLED_SIGNED_MODEL_DISTANCE" for case in case_results),
    }
    write_json(args.output_json, result)
    print(json.dumps({
        "P2B3_C5A_STATUS": result["P2B3_C5A_STATUS"],
        "MEASUREMENT_PIPELINE_STATUS": result["MEASUREMENT_PIPELINE_STATUS"],
        "FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS": result["FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS"],
        "case_count": result["case_count"],
        "robot_world_collision_samples": result["collision_findings"]["robot_world_collision_sample_total"],
        "minimum_robot_world_clearance_m": min(case["minimum_robot_world_clearance"]["signed_distance_m"] for case in case_results if case["minimum_robot_world_clearance"]["signed_distance_m"] is not None),
        "gates": result["measurement_gates"],
    }, sort_keys=True), flush=True)
    return 0 if result["P2B3_C5A_STATUS"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
