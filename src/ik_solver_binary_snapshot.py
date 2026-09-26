"""Stage 1.9.2 immutable input snapshot and hash helpers."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


TARGET_CALL_ID = "call:00000182"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def f64_le(values: list[float]) -> bytes:
    return struct.pack("<" + "d" * len(values), *(float(v) for v in values))


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> list[float]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = (qx / norm, qy / norm, qz / norm, qw / norm)
    return [
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), 0.0,
        2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw), 0.0,
        2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _joint_limits(urdf_path: Path) -> list[float]:
    root = ET.parse(urdf_path).getroot()
    values: list[float] = []
    for name in JOINT_NAMES:
        joint = next((item for item in root.findall("joint") if item.attrib.get("name") == name), None)
        if joint is None or joint.find("limit") is None:
            raise ValueError(f"missing joint limits for {name} in {urdf_path}")
        limit = joint.find("limit")
        values.extend([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
    return values


def _corpus_match(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    corpus_path = root / "outputs/ik_graph_stage191/frozen_ik_call_corpus.parquet"
    result: dict[str, Any] = {"path": corpus_path.as_posix(), "call_id": TARGET_CALL_ID, "status": "not_checked"}
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        result.update({"status": "unavailable", "reason": f"pyarrow_import:{type(exc).__name__}"})
        return result
    rows = pq.read_table(corpus_path).to_pylist()
    matches = [row for row in rows if row.get("call_id") == TARGET_CALL_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one {TARGET_CALL_ID} corpus row, found {len(matches)}")
    row = matches[0]
    expected = {
        "waypoint_id": case.get("waypoint_id"),
        "task_pose_stable_id": case.get("task_pose_stable_id"),
        "target_x": case["target_pose"]["x"], "target_y": case["target_pose"]["y"], "target_z": case["target_pose"]["z"],
        "target_qx": case["target_pose"]["qx"], "target_qy": case["target_pose"]["qy"], "target_qz": case["target_pose"]["qz"], "target_qw": case["target_pose"]["qw"],
    }
    mismatches = {}
    for key, value in expected.items():
        if value is None or key not in row:
            continue
        if isinstance(value, float):
            if float(row[key]) != value:
                mismatches[key] = {"case": value, "corpus": row[key]}
        elif row[key] != value:
            mismatches[key] = {"case": value, "corpus": row[key]}
    result.update({"status": "match" if not mismatches else "mismatch", "row_count": len(rows), "mismatches": mismatches, "row": row})
    if mismatches:
        raise ValueError(f"Stage 1.9.1 corpus mismatch for {TARGET_CALL_ID}: {mismatches}")
    return result


def build_snapshot(root: Path, output_dir: Path) -> dict[str, Any]:
    source = root / "outputs/ik_graph_stage191/minimal_reproduction_case.json"
    case = json.loads(source.read_text(encoding="utf-8"))
    if case.get("first_differing_record_id") != TARGET_CALL_ID:
        raise ValueError("Stage 1.9.1 minimal case does not identify call:00000182")
    # The three stored Stage 1.9.1 records are evidence, not a replacement input.
    pose = {str(k): float(v) for k, v in case["target_pose"].items()}
    seed = [float(v) for v in case["seed"]]
    init = case["robot_state_initialization"]
    all_variables = list(init["joint_positions"]) + list(init["joint_velocities"]) + list(init["joint_accelerations"])
    urdf_source = root / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"
    urdf = output_dir / "fairino5_v6_spray_tcp.expanded.urdf"
    if not urdf.exists():
        urdf = urdf_source
    srdf = root / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
    kinematics = root / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/kinematics.yaml"
    joint_limits_config = root / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
    transform = _quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    transform[3], transform[7], transform[11] = pose["x"], pose["y"], pose["z"]
    options = {"timeout_s": 0.0, "attempt_count": 1, "random_restart": False, "search_discretization": 0.005, "consistency_limits": None}
    files = {
        "target_transform_f64.bin": (f64_le(transform), {"dtype": "float64", "endianness": "little", "shape": [4, 4], "order": "row_major_homogeneous_transform"}),
        "seed_joint_vector_f64.bin": (f64_le(seed), {"dtype": "float64", "endianness": "little", "shape": [6], "order": JOINT_NAMES}),
        "all_robot_variables_f64.bin": (f64_le(all_variables), {"dtype": "float64", "endianness": "little", "shape": [18], "order": ["joint_positions[j1..j6]", "joint_velocities[j1..j6]", "joint_accelerations[j1..j6]"]}),
        "joint_limits_f64.bin": (f64_le(_joint_limits(urdf)), {"dtype": "float64", "endianness": "little", "shape": [12], "order": [f"{j}.{b}" for j in JOINT_NAMES for b in ("lower", "upper")]}),
        "solver_options.bin": (canonical_json(options), {"encoding": "utf-8", "format": "canonical_json", "fields": sorted(options)}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    field_specs = {}
    for name, (data, spec) in files.items():
        (output_dir / name).write_bytes(data)
        hashes[name.replace(".bin", "_sha256")] = sha256_bytes(data)
        field_specs[name] = {**spec, "byte_length": len(data), "sha256": sha256_bytes(data)}
    corpus = _corpus_match(root, case)
    corpus_row = corpus.get("row", {})
    waypoint_id = corpus_row.get("waypoint_id")
    task_pose_stable_id = corpus_row.get("task_pose_stable_id")
    manifest = {
        "schema_version": "1.0", "call_id": TARGET_CALL_ID,
        "source_minimal_case": source.relative_to(root).as_posix(),
        "waypoint_id": waypoint_id, "task_pose_stable_id": task_pose_stable_id,
        "target_pose": pose, "seed": seed, "binary_fields": field_specs,
        "target_transform_raw_sha256": hashes["target_transform_f64_sha256"],
        "seed_vector_raw_sha256": hashes["seed_joint_vector_f64_sha256"],
        "all_robot_variables_raw_sha256": hashes["all_robot_variables_f64_sha256"],
        "joint_limits_raw_sha256": hashes["joint_limits_f64_sha256"],
        "solver_options_raw_sha256": hashes["solver_options_sha256"],
        "urdf_sha256": sha256_file(urdf), "urdf_source_sha256": sha256_file(urdf_source), "srdf_sha256": sha256_file(srdf),
        "kinematics_sha256": sha256_file(kinematics), "joint_limits_config_sha256": sha256_file(joint_limits_config),
        "solver_plugin_name": "kdl_kinematics_plugin/KDLKinematicsPlugin",
        "solver_api_entry": "RobotState.set_from_ik / KinematicsBase.getPositionIK",
        "group_name": "fairino5_v6_group", "base_frame": "base_link", "tip_link": "spray_tcp_link",
        "mimic_joint_values": [], "redundant_joint_indices": [], "consistency_limits": None,
        "search_discretization": 0.005, "requested_timeout": 0.0,
        "raw_input_requires_bytewise_match": True, "corpus_match": corpus,
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = dict(case)
    result.update({"stage": "stage_1_9_2", "target_call_id": TARGET_CALL_ID, "source_minimal_case": source.relative_to(root).as_posix(), "binary_snapshot": "outputs/ik_graph_stage192/minimal_case", "input_manifest": manifest})
    (output_dir / "minimal_reproduction_case.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
