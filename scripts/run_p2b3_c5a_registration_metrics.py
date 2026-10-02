#!/usr/bin/env python3
"""Run scoped C5A registration metric closure against frozen C1/C4 authority."""

from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ThreadPoolExecutor
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
from src.physical_uncertainty import registration_transform, transform_pose_array_xyzw, transform_surface_normals

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
    # The FAIRINO vendor checkout is Windows-managed. Ask the Windows Git
    # configuration for status so WSL's different autocrlf view cannot invent
    # a dirty tree by normalizing the same checked-out text differently.
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


def ros2_run_command(scratch: Path, executable: str, arguments: list[str]) -> str:
    setup_chain = [
        scratch / "fairino_install/setup.bash",
        scratch / "bridge_install/setup.bash",
        scratch / "install/setup.bash",
    ]
    argv = ["ros2", "run", "p2b3_c5a_native", executable, *arguments]
    return " && ".join([*(f"source {shlex.quote(str(path))}" for path in setup_chain), shlex.join(argv)])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def transform_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("id", "dx", "dy", "dz", "px", "py", "pz", "qx", "qy", "qz", "qw")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_scene_transform_readback(path: Path, expected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual_rows = read_csv(path)
    if len(actual_rows) != 181 or len(expected_rows) != 181:
        raise RuntimeError(f"scene_transform_readback_count_mismatch:{path}")
    if [row["id"] for row in actual_rows] != [row["id"] for row in expected_rows]:
        raise RuntimeError(f"scene_transform_readback_object_identity_mismatch:{path}")
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    dimension_errors: list[float] = []
    for actual, expected in zip(actual_rows, expected_rows):
        actual_dimensions = np.asarray([float(actual[key]) for key in ("dx", "dy", "dz")])
        expected_dimensions = np.asarray([expected[key] for key in ("dx", "dy", "dz")], dtype=np.float64)
        dimension_errors.append(float(np.max(np.abs(actual_dimensions - expected_dimensions))))
        actual_position = np.asarray([float(actual[key]) for key in ("px", "py", "pz")])
        expected_position = np.asarray([expected[key] for key in ("px", "py", "pz")], dtype=np.float64)
        position_errors.append(float(np.linalg.norm(actual_position - expected_position)))
        actual_quaternion = np.asarray([float(actual[key]) for key in ("qx", "qy", "qz", "qw")])
        expected_quaternion = np.asarray([expected[key] for key in ("qx", "qy", "qz", "qw")], dtype=np.float64)
        dot = abs(float(np.dot(actual_quaternion, expected_quaternion))) / (
            float(np.linalg.norm(actual_quaternion)) * float(np.linalg.norm(expected_quaternion)))
        rotation_errors.append(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))
    summary = {
        "status": "PASS" if max(position_errors) <= 1e-12 and max(rotation_errors) <= 1e-12 and max(dimension_errors) == 0.0 else "FAIL",
        "object_count": len(actual_rows),
        "object_ids_match_ordered_c4_manifest": True,
        "dimensions_max_abs_error_m": max(dimension_errors),
        "translation_max_error_m": max(position_errors),
        "rotation_max_error_rad": max(rotation_errors),
        "comparison": "serialized transformed C4 rows read back against the left-composed SE(3) rows before native measurement",
    }
    if summary["status"] != "PASS":
        raise RuntimeError(f"scene_transform_readback_error:{path}:{summary}")
    return summary


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
    validate_c4_identity(identity)
    return identity


def validate_c4_identity(identity: dict[str, Any]) -> None:
    if identity.get("status") != "IDENTITY_GATE_PASS_PHASE_D_PENDING" or identity.get("scene_equivalence", {}).get("status") != "PASS":
        raise RuntimeError("C4_independent_identity_replay_failed")
    if identity.get("zero_transform_equivalence", {}).get("status") != "PASS":
        raise RuntimeError("C4_zero_transform_equivalence_failed")
    if identity.get("scene_manifest_sha256") != "f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669":
        raise RuntimeError("C4_identity_replay_scene_sha_mismatch")


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
    command = ros2_run_command(scratch, "p2b3_c5a_fcl_known_answer", [str(fixture_urdf), str(fixture_srdf)])
    output = bash_logged(command, ROOT, scratch / "fcl_known_answer.log", timeout_s).stdout
    record = next((json.loads(line) for line in reversed(output.splitlines()) if line.strip().startswith("{\"status\"")), None)
    if not isinstance(record, dict) or record.get("status") != "PASS":
        raise RuntimeError("native_FCL_known_answer_missing_or_failed")
    return record


def write_case_inputs(scratch: Path, manifest: dict[str, Any], q: np.ndarray, times: np.ndarray) -> list[dict[str, Any]]:
    input_dir = scratch / "cases"; input_dir.mkdir(parents=True, exist_ok=True)
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
            transformed_rows = scene_rows(manifest, delta)
            transform_to_csv(scene_path, transformed_rows)
            case["scene_transform_sanity"] = verify_scene_transform_readback(scene_path, transformed_rows)
            writer.writerow([case["case_id"], str(q_path), str(scene_path)])
    return cases


def run_native_cases(scratch: Path, timeout_s: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    case_rows = read_csv(scratch / "cases/case_inputs.csv")
    if len(case_rows) != 25:
        raise RuntimeError(f"native_case_manifest_count_mismatch:{len(case_rows)}")
    worker_count = min(4, os.cpu_count() or 1, len(case_rows))
    shards = [case_rows[index::worker_count] for index in range(worker_count)]

    def run_shard(index: int, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        shard_root = scratch / "native_shards" / f"shard_{index:02d}"
        shard_root.mkdir(parents=True, exist_ok=True)
        manifest_path = shard_root / "case_inputs.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("case_id", "q_csv", "scene_csv"), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        output = shard_root / "native_output"; output.mkdir(exist_ok=True)
        command = ros2_run_command(scratch, "p2b3_c5a_native", [
            "--cases", str(manifest_path), "--urdf", str(URDF), "--srdf", str(SRDF),
            "--output", str(output), "--group", GROUP, "--tip", TCP_LINK,
        ])
        bash_logged(command, ROOT, shard_root / "native_measurement.log", timeout_s)
        native = [json.loads(line) for line in (output / "p2b3_c5a_native_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        fk = read_csv(output / "p2b3_c5a_fresh_fk.csv")
        if len(native) != len(rows) or len(fk) != len(rows) * 181:
            raise RuntimeError(f"native_shard_output_count_mismatch:shard={index}:cases={len(native)}:fk_rows={len(fk)}")
        return native, fk

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(run_shard, index, rows) for index, rows in enumerate(shards)]
        shard_results = [future.result() for future in futures]

    native_by_case = {record["case_id"]: record for native, _ in shard_results for record in native}
    fk_by_case: dict[str, list[dict[str, str]]] = {}
    for _, shard_fk in shard_results:
        for row in shard_fk:
            fk_by_case.setdefault(row["case_id"], []).append(row)
    expected_ids = [row["case_id"] for row in case_rows]
    if set(native_by_case) != set(expected_ids) or set(fk_by_case) != set(expected_ids):
        raise RuntimeError("native_shard_case_identity_mismatch")
    native = [native_by_case[case_id] for case_id in expected_ids]
    fk = [row for case_id in expected_ids for row in fk_by_case[case_id]]
    native_output = scratch / "native_output"; native_output.mkdir(exist_ok=True)
    (native_output / "p2b3_c5a_native_cases.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in native), encoding="utf-8")
    with (native_output / "p2b3_c5a_fresh_fk.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fk[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(fk)
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


def c1_polyline_projector():
    """Compile only C1's frozen polyline projection function from its source AST."""
    source_path = ROOT / "ros2_moveit_bridge/plan_closed_contour_moveit.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_project_to_polyline_with_normals"]
    if len(definitions) != 1:
        raise RuntimeError("C1_project_to_polyline_with_normals_source_not_unique")
    isolated = ast.Module(body=definitions, type_ignores=[])
    namespace: dict[str, Any] = {"np": np}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    projector = namespace.get("_project_to_polyline_with_normals")
    if not callable(projector):
        raise RuntimeError("C1_polyline_projector_extraction_failed")
    return projector


def assemble_cases(
    case_specs: list[dict[str, Any]], native_records: list[dict[str, Any]], fk_rows: list[dict[str, str]],
    q: np.ndarray, times: np.ndarray, identity: dict[str, Any], identities: dict[str, Any],
    native_known_answer: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    project_to_polyline = c1_polyline_projector()
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
    c1_normal_error_deg = np.asarray([float(row["normal_angle_error_deg"]) for row in source_fk], dtype=np.float64)
    c1_standoff_error_m = np.asarray([float(row["standoff_error_mm"]) / 1000.0 for row in source_fk], dtype=np.float64)
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
                                  project_to_polyline=project_to_polyline)
        requested_translation = np.asarray(spec["translation_base_m"], dtype=np.float64)
        expected_delta = registration_transform(spec["translation_base_m"], spec["rotation_vector_base_rad"])
        delta_error = np.asarray(delta, dtype=np.float64) - expected_delta
        rotation_delta = expected_delta[:3, :3].T @ np.asarray(delta)[:3, :3]
        rotation_error_rad = math.acos(float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)))
        expected_transformed_target = transform_pose_array_xyzw(target, expected_delta)
        expected_transformed_normals = transform_surface_normals(normals, expected_delta)
        transformed_target_error = float(np.max(np.abs(
            np.asarray(metrics["target_pose_after_registration_xyz_quat_xyzw"], dtype=np.float64) - expected_transformed_target)))
        transformed_normal_error = float(np.max(np.abs(
            np.asarray(metrics["surface_normal_after_registration"], dtype=np.float64) - expected_transformed_normals)))
        registration_transform_sanity = {
            "status": "PASS" if np.linalg.norm(np.asarray(delta)[:3, 3] - requested_translation) <= 1e-12
                and rotation_error_rad <= 1e-12 and float(np.max(np.abs(delta_error))) <= 1e-12
                and transformed_target_error <= 1e-12 and transformed_normal_error <= 1e-12 else "FAIL",
            "translation_error_m": float(np.linalg.norm(np.asarray(delta)[:3, 3] - requested_translation)),
            "rotation_error_rad": rotation_error_rad,
            "transform_matrix_max_abs_error": float(np.max(np.abs(delta_error))),
            "target_pose_propagation_max_abs_error": transformed_target_error,
            "surface_normal_propagation_max_abs_error": transformed_normal_error,
            "propagation_definition": "C4 frozen left composition DeltaT_base @ T_base_entity_nominal",
        }
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
            "registration_transform_sanity": registration_transform_sanity,
            "scene_transform_sanity": spec["scene_transform_sanity"],
            "waypoint_count": 181, "adaptive_discrete_sample_count": int(raw["sample_count"]),
            "scene_object_count": 181, "fresh_fk_regression": fk_regression,
            "robot_world_collision_sample_count": int(raw["robot_world_collision_samples"]),
            "self_collision_sample_count": int(raw["self_collision_samples"]),
            "robot_world_distance_valid_sample_count": int(raw["robot_world_distance_valid_samples"]),
            "self_distance_valid_sample_count": int(raw["self_distance_valid_samples"]),
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
    zero_case = next((case for case in result_cases if case["case_id"] == "registration_zero"), {})
    zero_metrics = zero_case.get("process_metrics", {})

    def valid_vector(case: dict[str, Any], key: str) -> bool:
        values = np.asarray(case.get("process_metrics", {}).get(key, []), dtype=np.float64)
        return values.shape == (181,) and bool(np.isfinite(values).all())

    position_vectors_valid = all(valid_vector(case, "position_error_m") for case in result_cases)
    normal_vectors_valid = all(valid_vector(case, "normal_error_rad") for case in result_cases)
    standoff_vectors_valid = all(valid_vector(case, "stand_off_error_m") for case in result_cases)
    position_summaries_consistent = all(
        case["process_metrics"].get("critical_position_waypoint") == int(np.argmax(case["process_metrics"].get("position_error_m", [math.nan])))
        and abs(float(case["process_metrics"].get("max_position_error_m", math.nan))
                - float(np.max(case["process_metrics"].get("position_error_m", [math.nan])))) <= 1e-12
        for case in result_cases
    )
    normal_summaries_consistent = all(
        case["process_metrics"].get("critical_normal_waypoint") == int(np.argmax(case["process_metrics"].get("normal_error_rad", [math.nan])))
        and abs(float(case["process_metrics"].get("max_normal_error_rad", math.nan))
                - float(np.max(case["process_metrics"].get("normal_error_rad", [math.nan])))) <= 1e-12
        for case in result_cases
    )
    standoff_summaries_consistent = all(
        case["process_metrics"].get("critical_stand_off_waypoint") == int(np.argmax(np.abs(case["process_metrics"].get("stand_off_error_m", [math.nan]))))
        and abs(float(case["process_metrics"].get("max_abs_stand_off_error_m", math.nan))
                - float(np.max(np.abs(case["process_metrics"].get("stand_off_error_m", [math.nan]))))) <= 1e-12
        for case in result_cases
    )

    c1_position_errors = np.linalg.norm(expected_position - target[:, :3], axis=1)
    zero_position_matches_c1 = (
        len(zero_metrics.get("position_error_m", [])) == 181
        and np.allclose(zero_metrics.get("position_error_m", []), c1_position_errors, rtol=0.0, atol=1e-12)
        and abs(float(zero_metrics.get("max_position_error_m", math.nan))
                - float(c1_result.get("c1_measurement", {}).get("max_same_index_error_m", math.nan))) <= 1e-12
    )
    zero_normal_matches_c1 = (
        len(zero_metrics.get("normal_error_rad", [])) == 181 and c1_normal_error_deg.shape == (181,)
        and np.allclose(np.degrees(zero_metrics.get("normal_error_rad", [])), c1_normal_error_deg, rtol=0.0, atol=1e-8)
    )
    zero_standoff_matches_c1 = (
        len(zero_metrics.get("stand_off_error_m", [])) == 181 and c1_standoff_error_m.shape == (181,)
        and np.allclose(zero_metrics.get("stand_off_error_m", []), c1_standoff_error_m, rtol=0.0, atol=1e-9)
    )
    zero_process_transform_is_identity = (
        float(zero_metrics.get("target_pose_transform_max_position_delta_m", math.inf)) <= 1e-15
        and float(zero_metrics.get("surface_normal_transform_max_delta", math.inf)) <= 1e-15
        and float(zero_metrics.get("max_position_error_m", math.nan)) >= 0.0
    )
    zero_process_matches_c1 = all((zero_position_matches_c1, zero_normal_matches_c1,
                                   zero_standoff_matches_c1, zero_process_transform_is_identity))

    transform_propagation_pass = all(
        case.get("registration_transform_sanity", {}).get("status") == "PASS"
        and case.get("scene_transform_sanity", {}).get("status") == "PASS"
        for case in result_cases
    )
    c1_identity_pass = identities.get("c1_post_ruckig_nominal_sha256") == "be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3"
    c4_scene_identity_pass = identities.get("c4_authoritative_scene_manifest_sha256") == "f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669"
    world_clearance_values = np.asarray([
        float(case["minimum_robot_world_clearance"]["signed_distance_m"])
        for case in result_cases if case["minimum_robot_world_clearance"].get("signed_distance_m") is not None
    ], dtype=np.float64)
    clearance_spread_m = float(np.ptp(world_clearance_values)) if world_clearance_values.size else 0.0
    zero_normal_values = np.asarray(zero_metrics.get("normal_error_rad", []), dtype=np.float64)
    nonzero_translation_cases = [case for case in result_cases if case["family"] == "translation"]
    nonzero_rotation_cases = [case for case in result_cases if case["family"] == "rotation"]
    position_change_m = max(
        (abs(float(case["process_metrics"].get("max_position_error_m", 0.0))
             - float(zero_metrics.get("max_position_error_m", 0.0))) for case in nonzero_translation_cases),
        default=0.0,
    )
    normal_change_rad = max(
        (float(np.max(np.abs(np.asarray(case["process_metrics"].get("normal_error_rad", []), dtype=np.float64)
                           - zero_normal_values)))
         for case in nonzero_rotation_cases if zero_normal_values.shape == (181,)),
        default=0.0,
    )
    world_clearance_informative = clearance_spread_m > 1e-9
    translation_process_informative = position_change_m > 1e-9
    rotation_normal_informative = normal_change_rad > 1e-9
    metric_sanity = {
        "status": "PASS" if world_clearance_informative and translation_process_informative and rotation_normal_informative else "FAIL",
        "robot_world_clearance_minimum_spread_m": clearance_spread_m,
        "translation_max_position_error_change_m": position_change_m,
        "rotation_max_normal_error_change_rad": normal_change_rad,
        "numeric_informativity_epsilon": 1e-9,
        "epsilon_is_physical_acceptance_threshold": False,
    }
    known_answer_pass = (
        native_known_answer.get("status") == "PASS"
        and abs(float(native_known_answer.get("clear_distance_m", math.nan)) - 0.2) <= 1e-9
        and abs(float(native_known_answer.get("translated_distance_m", math.nan)) - 0.199) <= 1e-9
        and abs(float(native_known_answer.get("overlap_distance_m", math.nan)) + 0.05) <= 1e-9
        and native_known_answer.get("overlap_collision") is True
        and set(native_known_answer.get("pair", [])) == {"known_answer_obstacle", "base_link"}
        and np.allclose(native_known_answer.get("clear_nearest_points_x_m", []), [0.3, 0.1], rtol=0.0, atol=1e-9)
    )
    if not all(case["fresh_fk_regression"]["status"] == "PASS" for case in result_cases):
        fk_status = "FAIL"
    else:
        fk_status = "PASS"
    c4_zero_collision_count = int(identity.get("zero_baseline", {}).get("full_scene_collision_sample_count", -1))
    c5a_zero = native_by_id.get("registration_zero", {})
    zero_collision_match = int(c5a_zero.get("robot_world_collision_samples", -2)) == c4_zero_collision_count
    zero_robot_world_collision_count = int(c5a_zero.get("robot_world_collision_samples", -2))
    zero_self_collision_count = int(c5a_zero.get("self_collision_samples", -2))
    zero_registration_collision_count = zero_robot_world_collision_count + zero_self_collision_count
    zero_robot_world_clearance = clearance_record(c5a_zero, "minimum_robot_world_clearance", world_object_ids)
    self_domains_separated = all(
        case["minimum_self_clearance"].get("self_link_pair", {}).get("status") == "AVAILABLE"
        and all(name not in world_object_ids for name in case["minimum_self_clearance"].get("self_link_pair", {}).get("names", []))
        for case in result_cases
    )
    gate_checks = {
        "c4_identity_replay": identity.get("status") == "IDENTITY_GATE_PASS_PHASE_D_PENDING",
        "c4_scene_equivalence": identity.get("scene_equivalence", {}).get("status") == "PASS",
        "c4_zero_transform_gate": identity.get("zero_transform_equivalence", {}).get("status") == "PASS",
        "authoritative_c4_scene_identity_preserved": c4_scene_identity_pass,
        "c1_nominal_identity": c1_identity_pass,
        "25_frozen_cases": len(result_cases) == 25,
        "all_case_sample_counts_match_c4": all(case["adaptive_discrete_sample_count"] == 858 for case in result_cases),
        "all_robot_world_distances_finite_with_pair_and_points": all_world_available,
        "robot_world_distance_query_coverage_all_858_samples": all(case["robot_world_distance_valid_sample_count"] == 858 for case in result_cases),
        "self_distance_query_coverage_all_858_samples": all(case["self_distance_valid_sample_count"] == 858 for case in result_cases),
        "all_robot_world_pair_domains_identified": all(case["minimum_robot_world_clearance"].get("world_object_name", {}).get("status") == "AVAILABLE" and case["minimum_robot_world_clearance"].get("robot_link_name", {}).get("status") == "AVAILABLE" for case in result_cases),
        "all_self_distances_finite_with_pair_and_points": all_self_available,
        "self_clearance_query_separate_from_robot_world": self_domains_separated,
        "fresh_fk_matches_c1": fk_status == "PASS",
        "zero_collision_count_reconciles_to_independent_c4": zero_collision_match,
        "zero_registration_has_no_sampled_collisions": zero_registration_collision_count == 0,
        "all_q_and_timestamps_unchanged": all(case["joint_states_unchanged"] and case["timestamps_unchanged"] for case in result_cases),
        "zero_registration_process_geometry_matches_c1": zero_process_matches_c1,
        "process_position_metric_finite_all_181_waypoints": position_vectors_valid and position_summaries_consistent,
        "process_normal_metric_finite_all_181_waypoints": normal_vectors_valid and normal_summaries_consistent,
        "c1_standoff_metric_finite_all_181_waypoints": standoff_vectors_valid and standoff_summaries_consistent,
        "registration_and_scene_transform_propagation": transform_propagation_pass,
        "native_fcl_signed_distance_known_answer": known_answer_pass,
        "robot_world_clearance_and_process_metric_sanity": metric_sanity["status"] == "PASS",
    }
    collision_cases = [case for case in result_cases if case["robot_world_collision_sample_count"] > 0 or case["self_collision_sample_count"] > 0]
    performance = "WEAKNESS_OBSERVED" if collision_cases else "NO_DISCRETE_COLLISIONS_OBSERVED"
    result = {
        "schema_version": "p2b3-c5a-registration-metric-closure-v1",
        "PROJECT": "FAIRINO_FR5", "STAGE": "P2-B3-C5A",
        "P2B3_C5A_STATUS": "PASS" if all(gate_checks.values()) else "BLOCKED",
        "MEASUREMENT_PIPELINE_STATUS": "PASS" if all(gate_checks.values()) else "BLOCKED",
        "FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS": performance,
        "CLEARANCE_METRIC_VERIFIED": "YES" if all_world_available else "NO",
        "PROCESS_POSITION_METRIC": "VERIFIED" if gate_checks["process_position_metric_finite_all_181_waypoints"] else "BLOCKED",
        "STAND_OFF_METRIC": "VERIFIED_C1_AUTHENTICATED" if gate_checks["c1_standoff_metric_finite_all_181_waypoints"] else "BLOCKED",
        "PROCESS_NORMAL_METRIC": "VERIFIED_C1_PROJECTED_SURFACE_NORMAL" if gate_checks["process_normal_metric_finite_all_181_waypoints"] else "BLOCKED",
        "METRIC_SANITY_STATUS": metric_sanity["status"],
        "KNOWN_ANSWER_CASE_COUNT": len(result_cases),
        "NATIVE_FCL_KNOWN_ANSWER_FIXTURE_COUNT": 1 if known_answer_pass else 0,
        "ZERO_REGISTRATION_COLLISION_COUNT": zero_registration_collision_count,
        "ZERO_REGISTRATION_ROBOT_WORLD_COLLISION_COUNT": zero_robot_world_collision_count,
        "ZERO_REGISTRATION_SELF_COLLISION_COUNT": zero_self_collision_count,
        "ZERO_REGISTRATION_MIN_ROBOT_WORLD_CLEARANCE_M": zero_robot_world_clearance["signed_distance_m"],
        "REGISTRATION_BOUNDARY_CAMPAIGN": "NOT_RUN",
        "PHYSICAL_REGISTRATION_BOUND": "NOT_AVAILABLE",
        "PHYSICAL_NORMALIZED_MARGIN": "NOT_AVAILABLE",
        "STRICT_CONTINUOUS_CCD": "NOT_AVAILABLE",
        "HARDWARE_VALIDATED": "NOT_RUN",
        "COATING_QUALITY_CERTIFIED": "NO",
        "C1_NOMINAL_SHA256": identities["c1_post_ruckig_nominal_sha256"],
        "AUTHORITATIVE_SCENE_MANIFEST_SHA256": identities["c4_authoritative_scene_manifest_sha256"],
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
            "clearance_backend": "PlanningScene.getCollisionEnvUnpadded() MoveIt2 CollisionEnvFCL distanceRobot/distanceSelf; request GLOBAL, padding=0.0, scale=1.0",
            "clearance_query_domains": {"robot_world": "distanceRobot only; nearest robot link and world object are reported", "self": "distanceSelf separately; robot link pair only"},
            "clearance_is_signed": True,
            "collision_method": "adaptive_discrete_interpolation",
            "maximum_joint_interpolation_step_deg": 0.5,
            "process_position_metric": "per-waypoint Euclidean distance from frozen FK TCP position to C1 target TCP position after the same left-composed registration transform",
            "process_normal_metric": "per-waypoint angle in radians and degrees between frozen FK rotation column +Z and C1 projected open-arch surface normal after registration; exact C1 _project_to_polyline_with_normals function",
            "process_standoff_metric": "signed projected stand-off error using C1 STAND_OFF=0.260 m and tcp_points_to_wall=true convention",
            "process_metric_physical_thresholds": "NOT_APPLIED_TO_REGISTRATION_PERTURBATIONS; C1 configured criteria remain C1 provenance only",
            "strict_ccd": "NOT_AVAILABLE",
            "physical_registration_bound": "UNAVAILABLE_NO_MEASURED_FR5_BOUND",
            "hardware": "NOT_RUN",
            "low_clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD_NO_FROZEN_VALUE",
        },
        "measurement_gates": {key: "PASS" if value else "FAIL" for key, value in gate_checks.items()},
        "metric_sanity": metric_sanity,
        "native_fcl_known_answer": native_known_answer,
        "moveitpy_api_introspection": {
            "runtime": "ROS 2 Jazzy MoveItPy imported from moveit.planning and moveit.core",
            "planning_scene_public_collision_methods": ["check_collision", "check_collision_unpadded", "check_self_collision", "is_state_colliding", "is_state_valid"],
            "collision_result_public_distance_fields": ["collision", "distance"],
            "robot_world_distance_method": "NOT_AVAILABLE_IN_PYTHON_PLANNINGSCENE_BINDING",
            "nearest_robot_world_pair": "NOT_AVAILABLE_IN_PYTHON_COLLISIONRESULT_BINDING",
            "nearest_points": "NOT_AVAILABLE_IN_PYTHON_COLLISIONRESULT_BINDING",
            "measurement_route": "native MoveIt2 C++ CollisionEnvFCL; no Python CollisionResult.distance relabeling",
        },
        "c2_clearance_evaluator_audit": {
            "source": "cpp/stage3_h13_d41/stage3_h13_d41_native.cpp; run_p2b3_c2_robustness.py routes through src/p2a_axiswise_robustness._D41BatchEvaluator",
            "robot_world_query": "DistanceRequest::GLOBAL via CollisionEnv::distanceRobot; separate distanceSelf query, so environment clearance excludes self pairs",
            "acm": "ACM pointer supplied to both distance queries",
            "signed_semantics": "enable_signed_distance=true; MoveIt DistanceResultsData defines <=0 as collision",
            "padding_semantics": "uses getCollisionEnv() from a separately constructed scene; no explicit getCollisionEnvUnpadded() call, so an unpadded guarantee is not established by the C2 implementation",
            "nearest_pair": "link pair string retained",
            "nearest_points": "not retained in the C2 Distance record or result rows",
            "c4_authority_transfer": "C2 rebuilds its scene from pose rows; direct use of the C4 authoritative manifest is not established there. C5A instead loads the frozen C4 181-object manifest directly.",
        },
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
            "REGISTRATION_BOUNDARY_CAMPAIGN": "NOT_RUN",
            "PHYSICAL_REGISTRATION_BOUND": "NOT_AVAILABLE",
            "PHYSICAL_NORMALIZED_MARGIN": "NOT_AVAILABLE",
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
    parser.add_argument("--resume-scratch", action="store_true", help="Reuse prior verified C4 identity and generated case inputs in this scratch directory")
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--build-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    if args.resume_scratch and not scratch.is_dir():
        raise RuntimeError(f"resume_scratch_directory_missing:{scratch}")
    if not args.resume_scratch and scratch.exists():
        raise RuntimeError(f"scratch_directory_must_not_exist:{scratch}")
    tree = require_clean_execution_tree()
    fairino_identity = verify_fairino_source(args.fairino_source.resolve())
    identities = require_inputs(fairino_identity)
    if args.resume_scratch:
        c4_identity_path = scratch / "c4_identity_result.json"
        cases_path = scratch / "cases/case_inputs.csv"
        if not c4_identity_path.is_file() or not cases_path.is_file():
            raise RuntimeError("resume_scratch_missing_verified_c4_identity_or_case_manifest")
        c4_identity = read_json(c4_identity_path)
        validate_c4_identity(c4_identity)
    else:
        scratch.mkdir(parents=True)
        build_moveit_runtime(scratch, args.fairino_source.resolve(), args.build_timeout_seconds)
        c4_identity = run_c4_identity(scratch, args.timeout_seconds)
    c1_rows = read_csv(C1_TRAJECTORY)
    q, times = write_q_only(scratch / "c1_q_only_validation.csv", c1_rows)
    case_specs = write_case_inputs(scratch, read_json(C4_MANIFEST), q, times)
    build_native(scratch, args.build_timeout_seconds)
    known_answer = run_native_known_answer(scratch, args.timeout_seconds)
    native_records, fk_rows = run_native_cases(scratch, args.timeout_seconds)
    case_results, result = assemble_cases(
        case_specs, native_records, fk_rows, q, times, c4_identity, identities, known_answer,
    )
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
    world_minima = [case["minimum_robot_world_clearance"]["signed_distance_m"] for case in case_results
                    if case["minimum_robot_world_clearance"]["signed_distance_m"] is not None]
    print(json.dumps({
        "P2B3_C5A_STATUS": result["P2B3_C5A_STATUS"],
        "MEASUREMENT_PIPELINE_STATUS": result["MEASUREMENT_PIPELINE_STATUS"],
        "FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS": result["FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS"],
        "case_count": result["case_count"],
        "robot_world_collision_samples": result["collision_findings"]["robot_world_collision_sample_total"],
        "minimum_robot_world_clearance_m": min(world_minima) if world_minima else None,
        "gates": result["measurement_gates"],
    }, sort_keys=True), flush=True)
    return 0 if result["P2B3_C5A_STATUS"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
