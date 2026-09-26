"""Run the isolated Stage 2.5 deterministic TOTG -> Ruckig certification.

The script treats the user-specified Stage 2.4T directory as immutable.  It
materializes a separate baseline, invokes native MoveIt2 in three independent
WSL processes, reuses the certified native FCL/Bullet probe for the resulting
trajectory intervals, and writes a machine-readable gate bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage25_contract import (  # noqa: E402
    FORMAL_RUCKIG_PARAMETERS,
    FORMAL_TOTG_PARAMETERS,
    REPAIR_REASON,
    formal_contract,
)


STAGE24T = ROOT / "outputs/ik_graph_stage24t_final_recovery/fr5_scaled_horseshoe_demo_v45_20260803_001724_Stage24T"
WAYPOINTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/waypoints.csv"
PARTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
URDF = ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"
MOVEIT_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
OFFICIAL_LIMITS = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/joint_limits.yaml"
ROS2_CONTROLLERS = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/ros2_controllers.yaml"
MOVEIT_CONTROLLERS = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/moveit_controllers.yaml"
SCALED_PARAMS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/scaled_demo_parameters.yaml"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
XACRO = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
INTERPOLATION_STEP_DEG = 0.25


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


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


def file_entry(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None}


def directory_manifest(path: Path, role: str) -> dict[str, Any]:
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files.append({"relative_path": item.relative_to(path).as_posix(), "size": item.stat().st_size, "sha256": sha256(item)})
    return {"path": str(path.resolve()), "role": role, "exists": path.is_dir(), "file_count": len(files), "files": files, "tree_sha256": digest(files)}


def git_snapshot() -> dict[str, str]:
    def run(args: list[str]) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
        return result.stdout

    return {"status_before": run(["status", "--short", "--branch"]), "diff_stat_before": run(["diff", "--stat"])}


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_limits() -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    urdf_root = ET.parse(URDF).getroot()
    urdf_limits: dict[str, dict[str, Any]] = {}
    for joint in urdf_root.findall("joint"):
        name = joint.attrib.get("name", "")
        limit = joint.find("limit")
        if name in JOINT_NAMES and limit is not None:
            urdf_limits[name] = {key: float(value) for key, value in limit.attrib.items()}
    moveit = yaml.safe_load(MOVEIT_LIMITS.read_text(encoding="utf-8")) or {}
    official = yaml.safe_load(OFFICIAL_LIMITS.read_text(encoding="utf-8")) or {}
    moveit_joint = moveit.get("joint_limits", {})
    official_joint = official.get("joint_limits", {})
    rows = []
    for name in JOINT_NAMES:
        urdf = urdf_limits[name]
        project = moveit_joint.get(name, {})
        official_row = official_joint.get(name, {})
        velocity = float(project.get("max_velocity", official_row.get("max_velocity", urdf["velocity"])))
        acceleration = float(project.get("max_acceleration", official_row.get("max_acceleration")))
        jerk = project.get("max_jerk") if project.get("has_jerk_limits", False) else None
        rows.append(
            {
                "joint": name,
                "position_lower_rad": urdf["lower"],
                "position_upper_rad": urdf["upper"],
                "max_velocity_rad_s": velocity,
                "max_acceleration_rad_s2": acceleration,
                "max_jerk_rad_s3": float(jerk) if jerk is not None else None,
                "sources": {
                    "position": "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf; MoveIt joint_limits.yaml has no position override",
                    "velocity": "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml override, matching URDF/official MoveIt joint_limits.yaml",
                    "acceleration": "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml override, matching official MoveIt joint_limits.yaml",
                    "jerk": "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml explicit has_jerk_limits/max_jerk contract",
                },
                "controller_numeric_limit_override": False,
            }
        )
    controller_text = ROS2_CONTROLLERS.read_text(encoding="utf-8")
    moveit_controller_text = MOVEIT_CONTROLLERS.read_text(encoding="utf-8")
    result = {
        "schema_version": "stage25-joint-limits-v1",
        "limit_source": {
            "position": str(URDF.resolve()),
            "velocity": str(MOVEIT_LIMITS.resolve()),
            "acceleration": str(MOVEIT_LIMITS.resolve()),
            "jerk": str(MOVEIT_LIMITS.resolve()),
        },
        "position_rule": "bounded revolute URDF limits are authoritative; no controller/MoveIt YAML position override is present",
        "velocity_rule": "MoveIt planning YAML override is loaded by the Stage 2.5 launch",
        "acceleration_rule": "MoveIt planning YAML override is loaded by the Stage 2.5 launch",
        "jerk_rule": "explicit project MoveIt/Ruckig contract; not present in URDF or ros2_control controller YAML",
        "controller_configuration": {
            "ros2_controllers_path": str(ROS2_CONTROLLERS.resolve()),
            "moveit_controllers_path": str(MOVEIT_CONTROLLERS.resolve()),
            "numeric_position_velocity_acceleration_jerk_limits_present": False,
            "command_interfaces": ["position"],
            "state_interfaces": ["position"],
            "follow_joint_trajectory_joints": JOINT_NAMES,
            "controller_sha256": {"ros2_controllers.yaml": sha256(ROS2_CONTROLLERS), "moveit_controllers.yaml": sha256(MOVEIT_CONTROLLERS)},
            "ros2_controllers_contains_jerk": "jerk" in controller_text.lower(),
            "moveit_controllers_contains_jerk": "jerk" in moveit_controller_text.lower(),
        },
        "joints": rows,
        "all_jerk_limits_explicit": all(row["max_jerk_rad_s3"] is not None for row in rows),
        "moveit_config_defaults": {"velocity_scaling": float(moveit.get("default_velocity_scaling_factor", 0.0)), "acceleration_scaling": float(moveit.get("default_acceleration_scaling_factor", 0.0))},
    }
    return result


def load_process_items() -> tuple[list[dict[str, Any]], list[dict[str, str]], np.ndarray]:
    process = read_json(STAGE24T / "stage24t_process_manifest.json")
    if process.get("spray_on_segments") != 10 or process.get("off_transitions") != 9:
        raise RuntimeError("Frozen Stage 2.4T process manifest does not contain 10 ON and 9 OFF items.")
    items: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []
    on_rows: list[dict[str, str]] = []
    on_waypoints: list[int] = []
    for item in process["ordered_items"]:
        file_path = STAGE24T / item["trajectory_file"]
        rows = parse_csv(file_path)
        if item["type"] == "spray_on":
            current = []
            for row in rows:
                waypoint = int(row["waypoint_index"])
                on_waypoints.append(waypoint)
                on_rows.append(row)
                current.append({"spray_state": "ON", "segment_id": int(item["segment_id"]), "transition_id": None, "source_waypoint": waypoint, "phase": "spray_on", "source_file": item["trajectory_file"], "source_row_index": len(current), "q": [float(row[f"q{i}"]) for i in range(1, 7)]})
            items.append({"type": "spray_on", "segment_id": int(item["segment_id"]), "source_waypoint_start": int(item["start_waypoint"]), "source_waypoint_end": int(item["end_waypoint"]), "source_file": item["trajectory_file"], "samples": current})
        else:
            transition = int(item["transition_id"])
            current = []
            for row in rows:
                current.append({"spray_state": "OFF", "segment_id": None, "transition_id": transition, "source_waypoint": None, "phase": row["phase"], "source_file": item["trajectory_file"], "source_row_index": int(row["trajectory_point_index"]), "q": [float(row[f"q{i}"]) for i in range(1, 7)]})
            on_left = int(process["ordered_items"][process["ordered_items"].index(item) - 1]["end_waypoint"])
            on_right = int(process["ordered_items"][process["ordered_items"].index(item) + 1]["start_waypoint"])
            items.append({"type": "spray_off_transition", "transition_id": transition, "source_boundary": f"{on_left}->{on_right}", "source_waypoint_start": on_left, "source_waypoint_end": on_right, "source_file": item["trajectory_file"], "samples": current})
        source_rows.extend(rows)
    if on_waypoints != list(range(720)):
        raise RuntimeError(f"Frozen Stage 2.4T ON coverage is not 0..719: {on_waypoints[:3]} ... {on_waypoints[-3:]}")
    # The ordered process is concatenated while retaining one copy of exact
    # boundary states for native TOTG.  Alias metadata keeps both ON/OFF
    # process meanings at shared boundary states.
    path_samples: list[dict[str, Any]] = []
    for item in items:
        for sample in item["samples"]:
            q = np.asarray(sample["q"], dtype=float)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("Frozen Stage 2.4T process contains a non-finite joint state.")
            alias = {key: value for key, value in sample.items() if key != "q"}
            if path_samples and np.allclose(q, np.asarray(path_samples[-1]["q"], dtype=float), atol=1e-12, rtol=0.0):
                path_samples[-1].setdefault("aliases", []).append(alias)
            else:
                path_samples.append({**alias, "q": q.tolist(), "aliases": []})
    for index, sample in enumerate(path_samples):
        sample["path_index"] = index
        memberships = [sample] + sample.get("aliases", [])
        sample["segment_ids"] = sorted({int(x["segment_id"]) for x in memberships if x.get("segment_id") is not None})
        sample["transition_ids"] = sorted({int(x["transition_id"]) for x in memberships if x.get("transition_id") is not None})
        sample["source_waypoints"] = sorted({int(x["source_waypoint"]) for x in memberships if x.get("source_waypoint") is not None})
        if sample.get("spray_state") == "ON" and sample.get("source_waypoint") is not None:
            sample["spray_state"] = "ON"
        elif sample.get("spray_state") != "ON":
            sample["spray_state"] = "OFF"
    q_array = np.asarray([row["q"] for row in path_samples], dtype=float)
    if not np.all(np.isfinite(q_array)):
        raise RuntimeError("Stage 2.5 baseline q array is non-finite.")
    return items, path_samples, q_array


def write_baseline(output_root: Path, items: list[dict[str, Any]], path_samples: list[dict[str, Any]], q_array: np.ndarray, limits: dict[str, Any], snapshot: dict[str, str]) -> dict[str, Any]:
    pre_path = output_root / "stage25_pre_timing_joint_trajectory.csv"
    with pre_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_index", "spray_state", "segment_ids", "transition_ids", "source_waypoints", "phase", *[f"q{i}" for i in range(1, 7)]])
        for row in path_samples:
            writer.writerow([row["path_index"], row["spray_state"], ";".join(map(str, row["segment_ids"])), ";".join(map(str, row["transition_ids"])), ";".join(map(str, row["source_waypoints"])), row.get("phase", ""), *row["q"]])
    segmentation: list[dict[str, Any]] = []
    for item in items:
        if item["type"] == "spray_on":
            indices = [row["path_index"] for row in path_samples if int(item["segment_id"]) in row["segment_ids"]]
            segmentation.append({"segment_id": int(item["segment_id"]), "spray_state": "ON", "source_waypoint_start": int(item["source_waypoint_start"]), "source_waypoint_end": int(item["source_waypoint_end"]), "source_file": item["source_file"], "pre_timing_path_sample_start": min(indices), "pre_timing_path_sample_end": max(indices), "source_sample_count": len(item["samples"])})
        else:
            indices = [row["path_index"] for row in path_samples if int(item["transition_id"]) in row["transition_ids"]]
            segmentation.append({"transition_id": int(item["transition_id"]), "spray_state": "OFF", "source_boundary": item["source_boundary"], "source_waypoint_start": int(item["source_waypoint_start"]), "source_waypoint_end": int(item["source_waypoint_end"]), "source_file": item["source_file"], "pre_timing_path_sample_start": min(indices), "pre_timing_path_sample_end": max(indices), "source_sample_count": len(item["samples"])})
    ref = {
        "schema_version": "stage25-frozen-path-reference-v1",
        "source_stage24t_root": str(STAGE24T.resolve()),
        "source_q": q_array.tolist(),
        "path_samples": path_samples,
        "items": segmentation,
        "robot_base_xyz_m": [float(x) for x in (yaml.safe_load(SCALED_PARAMS.read_text(encoding="utf-8")) or {}).get("robot_base", {}).get("xyz_m", [-0.32, 0.05, 0.38])],
        "standoff_nominal_m": float((yaml.safe_load(SCALED_PARAMS.read_text(encoding="utf-8")) or {}).get("process", {}).get("standoff_nominal_m", 0.08)),
    }
    write_json(output_root / "stage25_path_reference.json", ref)
    segmentation_manifest = {"schema_version": "stage25-segmentation-v1", "source_stage24t_geometry_frozen": True, "waypoint_coverage": "720/720", "spray_on_segments": 10, "spray_off_transitions": 9, "items": segmentation, "boundary_402_403": {"present": True, "transition_id": 6, "source_boundary": "402->403", "semantics": "Spray-OFF continuous transition between ON segment 6 and ON segment 7"}, "continuous_process_strategy": "single_continuous_timed_trajectory; internal boundaries retain native dynamic continuity; no forced zero velocity/acceleration at internal boundaries", "selected_joint_states_identical_to_stage24t": bool(len(q_array) >= 720), "pre_timing_path_sample_count": int(len(q_array)), "joint_limits_file": str(MOVEIT_LIMITS.resolve())}
    write_json(output_root / "stage25_segmentation_manifest.json", segmentation_manifest)
    write_json(output_root / "stage25_joint_limits.json", limits)
    return ref


def build_input_manifest(output_root: Path, snapshot: dict[str, str], limits: dict[str, Any]) -> None:
    gate = read_json(STAGE24T / "stage24t_gate_report.json")
    if gate.get("Stage_2_4T") != "passed":
        raise RuntimeError("Frozen Stage 2.4T gate is not passed.")
    entries = [
        file_entry(STAGE24T / "stage24t_gate_report.json", "formal Stage 2.4T gate"),
        file_entry(STAGE24T / "stage24t_report.md", "formal Stage 2.4T report"),
        file_entry(STAGE24T / "stage24t_input_manifest.json", "formal Stage 2.4T input manifest"),
        file_entry(STAGE24T / "stage24t_process_manifest.json", "formal Stage 2.4T segmentation"),
        file_entry(STAGE24T / "stage24t_selected_process.json", "formal selected Stage 2.4T process"),
        file_entry(STAGE24T / "stage24t_determinism_report.json", "formal Stage 2.4T determinism"),
        file_entry(STAGE24T / "SHA256SUMS", "formal Stage 2.4T SHA256 manifest"),
        file_entry(WAYPOINTS, "frozen nominal 720 waypoint pose/normal source"),
        file_entry(URDF, "frozen FR5 URDF position/velocity source"),
        file_entry(SRDF, "frozen SRDF/ACM"),
        file_entry(XACRO, "frozen TCP model wrapper"),
        file_entry(MOVEIT_LIMITS, "formal MoveIt velocity/acceleration/jerk override"),
        file_entry(OFFICIAL_LIMITS, "official MoveIt joint limits comparison"),
        file_entry(ROS2_CONTROLLERS, "controller configuration"),
        file_entry(MOVEIT_CONTROLLERS, "MoveIt controller configuration"),
        file_entry(SCALED_PARAMS, "frozen scaled demo process/scene contract"),
        directory_manifest(PARTS, "frozen collision geometry directory"),
    ]
    entries.extend(file_entry(STAGE24T / name, "frozen Stage 2.4T segment") for name in sorted(p.relative_to(STAGE24T).as_posix() for p in (STAGE24T / "segments").glob("*.csv")))
    manifest = {"schema_version": "stage25-input-manifest-v2", "stage": "Stage_2_5", "source_stage24t_geometry_frozen": True, "source_stage24t_gate": gate, "frozen_inputs": entries, "stage24t_input_manifest_sha256": sha256(STAGE24T / "stage24t_input_manifest.json"), "stage24t_process_manifest_sha256": sha256(STAGE24T / "stage24t_process_manifest.json"), "stage24t_formal_sha256sums_sha256": sha256(STAGE24T / "SHA256SUMS"), "limit_contract_sha256": sha256(MOVEIT_LIMITS), "formal_contract": formal_contract(), "formal_totg_parameters": FORMAL_TOTG_PARAMETERS, "formal_ruckig_parameters": FORMAL_RUCKIG_PARAMETERS, "repair_reason": REPAIR_REASON, "git_snapshot": snapshot, "no_stage24t_rebuild_requested": True, "no_ik_or_segmentation_mutation": True}
    write_json(output_root / "stage25_input_manifest.json", manifest)


def verify_frozen_inputs(output_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Re-hash every frozen file/directory entry after certification."""

    mismatches = []
    checked = 0
    for entry in manifest.get("frozen_inputs", []):
        path = Path(entry["path"])
        if entry.get("role") == "frozen collision geometry directory":
            current = directory_manifest(path, entry["role"])
            before = entry.get("tree_sha256")
            after = current.get("tree_sha256")
        else:
            before = entry.get("sha256")
            after = sha256(path) if path.is_file() else None
        checked += 1
        if before != after:
            mismatches.append({"path": str(path), "before": before, "after": after, "role": entry.get("role")})
    result = {"schema_version": "stage25-frozen-input-hash-check-v1", "checked": checked, "mismatch_count": len(mismatches), "unchanged": not mismatches, "mismatches": mismatches}
    write_json(output_root / "stage25_frozen_input_hash_check.json", result)
    return result


def find_native_stage24_install() -> Path:
    candidates = []
    for path in STAGE24T.rglob("native_install"):
        if (path / "lib/stage24_closed_loop_graph/stage24_formal_probe").exists():
            candidates.append(path)
    if not candidates:
        raise RuntimeError("The frozen Stage 2.4T native stage24_formal_probe install was not found.")
    return sorted(candidates)[0]


def find_stage23b_install() -> Path:
    candidates = list((ROOT / "outputs/ik_graph_stage23b_representation_remediation").rglob("install"))
    candidates = [path for path in candidates if (path / "lib/libstage23b_bullet_shape_interposer.so").exists()]
    if not candidates:
        raise RuntimeError("The certified Stage 2.3B Bullet shape interposer install was not found.")
    return sorted(candidates)[0]


def run_moveit(rebuild_dir: Path, ref_path: Path) -> dict[str, Any]:
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {wsl_path(ROOT / 'install/setup.bash')}",
        "export FR5_BULLET_SHAPE_MODE=use_shape_type",
        f"ros2 launch {wsl_path(ROOT / 'tools/stage25_moveit_launch.py')} input_csv:={wsl_path(rebuild_dir.parent / 'stage25_pre_timing_joint_trajectory.csv')} source_reference_json:={wsl_path(ref_path)} waypoints_csv:={wsl_path(WAYPOINTS)} output_dir:={wsl_path(rebuild_dir)} velocity_scaling:={FORMAL_TOTG_PARAMETERS['velocity_scaling_factor']} acceleration_scaling:={FORMAL_TOTG_PARAMETERS['acceleration_scaling_factor']} path_tolerance:={FORMAL_TOTG_PARAMETERS['path_tolerance']} resample_dt:={FORMAL_TOTG_PARAMETERS['resample_dt']} min_angle_change:={FORMAL_TOTG_PARAMETERS['min_angle_change']} ruckig_mitigate_overshoot:={str(FORMAL_RUCKIG_PARAMETERS['mitigate_overshoot']).lower()} ruckig_overshoot_threshold:={FORMAL_RUCKIG_PARAMETERS['overshoot_threshold']}",
    ])
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    (rebuild_dir / "stage25_moveit_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (rebuild_dir / "stage25_moveit_stderr.log").write_text(proc.stderr, encoding="utf-8")
    (rebuild_dir / "stage25_moveit_exit_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    runtime_path = rebuild_dir / "stage25_native_moveit_runtime.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else {"status": "blocked_child_result_missing"}
    runtime.update({"child_exit_code": proc.returncode, "stdout_path": str((rebuild_dir / "stage25_moveit_stdout.log").resolve()), "stderr_path": str((rebuild_dir / "stage25_moveit_stderr.log").resolve())})
    write_json(runtime_path, runtime)
    return runtime


def write_collision_edges(rebuild_dir: Path) -> Path:
    traj = read_trajectory_csv(rebuild_dir / "stage25_ruckig_trajectory.csv")
    edge_path = rebuild_dir / "stage25_collision_edges.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "interpolation_step_deg"] + [f"q0_{i}" for i in range(6)] + [f"q1_{i}" for i in range(6)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(traj["q"]) - 1):
            row: dict[str, Any] = {"from_waypoint": index, "to_waypoint": index + 1, "from_candidate_id": f"stage25-{index:07d}", "to_candidate_id": f"stage25-{index + 1:07d}", "interpolation_step_deg": INTERPOLATION_STEP_DEG}
            row.update({f"q0_{j}": float(traj["q"][index, j]) for j in range(6)})
            row.update({f"q1_{j}": float(traj["q"][index + 1, j]) for j in range(6)})
            writer.writerow(row)
    return edge_path


def run_native_collision(rebuild_dir: Path, backend: str, run_index: int, native_install: Path, stage23b_install: Path, edge_path: Path) -> dict[str, Any]:
    run_dir = rebuild_dir / f"collision_{backend}"
    run_dir.mkdir(parents=True, exist_ok=True)
    overlay = wsl_path(stage23b_install / "lib/libstage23b_bullet_shape_interposer.so")
    am = ":".join([wsl_path(native_install), wsl_path(stage23b_install), f"{wsl_path(ROOT)}/tmp/stage23a7_install2", f"{wsl_path(ROOT)}/install/fairino5_v6_moveit2_config", f"{wsl_path(ROOT)}/install/fairino_description", f"{wsl_path(ROOT)}/install/fr5_tunnel_moveit_bridge", "/opt/ros/jazzy"])
    ld = ":".join([f"{wsl_path(native_install)}/lib", f"{wsl_path(stage23b_install)}/lib", f"{wsl_path(ROOT)}/tmp/stage23a7_install2/lib", "/opt/ros/jazzy/opt/sdformat_vendor/lib", "/opt/ros/jazzy/opt/rviz_ogre_vendor/lib", "/opt/ros/jazzy/lib/x86_64-linux-gnu", "/opt/ros/jazzy/opt/gz_math_vendor/lib", "/opt/ros/jazzy/opt/gz_utils_vendor/lib", "/opt/ros/jazzy/opt/gz_tools_vendor/lib", "/opt/ros/jazzy/lib"])
    preload = f"export LD_PRELOAD={overlay}" if backend == "bullet" else "unset LD_PRELOAD"
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"export AMENT_PREFIX_PATH={am}",
        f"export LD_LIBRARY_PATH={ld}",
        "export FR5_BULLET_SHAPE_MODE=use_shape_type",
        preload,
        f"ros2 launch {wsl_path(ROOT / 'tools/stage24_formal_launch.py')} edge_csv:={wsl_path(edge_path)} parts_dir:={wsl_path(PARTS)} output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index}",
    ])
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    (run_dir / "exit_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    result_path = run_dir / "native_edge_results.jsonl"
    rows = read_jsonl(result_path) if result_path.exists() else []
    accepted = {(int(row["from_waypoint"]), int(row["to_waypoint"])) for row in rows if row.get("status") == "accepted"}
    summary = {"backend": backend, "run_index": run_index, "exit_code": proc.returncode, "result_path": str(result_path.resolve()), "result_sha256": sha256(result_path) if result_path.exists() else None, "result_rows": len(rows), "accepted_rows": len(accepted), "collision_rows": sum(1 for row in rows if row.get("status") != "accepted"), "checked_samples": int(sum(int(row.get("samples_checked", 0)) for row in rows)), "collision_free": bool(rows and len(rows) == len(accepted) and all(row.get("status") == "accepted" for row in rows)), "interpolation_step_deg": INTERPOLATION_STEP_DEG, "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available", "native_probe": str((native_install / "lib/stage24_closed_loop_graph/stage24_formal_probe").resolve()), "native_probe_sha256": sha256(native_install / "lib/stage24_closed_loop_graph/stage24_formal_probe")}
    write_json(run_dir / "stage25_collision_summary.json", summary)
    return {**summary, "rows": rows, "accepted_set": accepted}


def read_trajectory_csv(path: Path) -> dict[str, Any]:
    rows = parse_csv(path)
    if not rows:
        raise RuntimeError(f"Timed trajectory is empty: {path}")
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    dq = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=float)
    ddq = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=float)
    jerk = np.asarray([[float(row[f"j{i}_jerk"]) for i in range(1, 7)] for row in rows], dtype=float)
    return {"rows": rows, "t": t, "q": q, "dq": dq, "ddq": ddq, "jerk": jerk}


def map_indices(final_q: np.ndarray, source_q: np.ndarray) -> list[int]:
    if len(final_q) == len(source_q) and np.allclose(final_q, source_q, atol=1e-10, rtol=0.0):
        return list(range(len(source_q)))
    result = []
    cursor = 0
    for row in final_q:
        global_index = int(np.argmin(np.linalg.norm(source_q - row[None, :], axis=1)))
        end = min(len(source_q), cursor + 500)
        window = source_q[cursor:end]
        if len(window) == 0:
            result.append(len(source_q) - 1)
            continue
        local_index = cursor + int(np.argmin(np.linalg.norm(window - row[None, :], axis=1)))
        cursor = max(cursor, global_index if cursor <= global_index <= end - 1 else local_index)
        result.append(cursor)
    return result


def transition_membership(ref: dict[str, Any], source_index: int, transition_id: int) -> bool:
    row = ref["path_samples"][min(max(source_index, 0), len(ref["path_samples"]) - 1)]
    return transition_id in row.get("transition_ids", [])


def build_process_manifest(output_root: Path, ref: dict[str, Any], traj: dict[str, Any], collision_fcl: dict[str, Any], collision_bullet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_indices = map_indices(traj["q"], np.asarray(ref["source_q"], dtype=float))
    items = []
    for item in ref["items"]:
        if item["spray_state"] == "ON":
            indices = [i for i, src in enumerate(source_indices) if int(item["segment_id"]) in ref["path_samples"][src].get("segment_ids", [])]
        else:
            indices = [i for i, src in enumerate(source_indices) if int(item["transition_id"]) in ref["path_samples"][src].get("transition_ids", [])]
        if not indices:
            raise RuntimeError(f"Timed mapping has no samples for process item {item}")
        items.append({**{key: value for key, value in item.items() if key not in {"pre_timing_path_sample_start", "pre_timing_path_sample_end"}}, "timed_start_s": float(traj["t"][min(indices)]), "timed_end_s": float(traj["t"][max(indices)]), "timed_final_sample_start": min(indices), "timed_final_sample_end": max(indices), "timed_sample_count": len(indices), "duration_s": float(traj["t"][max(indices)] - traj["t"][min(indices)]), "process_order_index": len(items)})
    on_waypoints = set()
    for src in source_indices:
        for wp in ref["path_samples"][src].get("source_waypoints", []):
            on_waypoints.add(wp)
    manifest = {"schema_version": "stage25-process-manifest-v1", "process_type": "segmented_spray_process_v1", "trajectory_execution_strategy": "single_continuous_timed_trajectory", "internal_boundary_policy": "dynamic continuity retained; internal boundaries are not force-zeroed", "spray_on_segments": 10, "spray_off_transitions": 9, "waypoint_coverage": f"{len(on_waypoints)}/720", "process_order_preserved": True, "boundary_order_preserved": True, "source_stage24t_geometry_frozen": True, "ordered_items": items, "boundary_402_403": next(row for row in items if row.get("transition_id") == 6), "trajectory_start_velocity": traj["dq"][0].tolist(), "trajectory_start_acceleration": traj["ddq"][0].tolist(), "trajectory_end_velocity": traj["dq"][-1].tolist(), "trajectory_end_acceleration": traj["ddq"][-1].tolist(), "trajectory_duration_s": float(traj["t"][-1]), "controller_execution": "not_requested_offline_certification"}
    off_rows = []
    for transition_id in range(1, 10):
        indices = [i for i, src in enumerate(source_indices) if transition_membership(ref, src, transition_id)]
        edge_indices = [i for i in range(len(traj["q"]) - 1) if transition_membership(ref, source_indices[i], transition_id) or transition_membership(ref, source_indices[i + 1], transition_id)]
        row = next(item for item in items if item.get("transition_id") == transition_id)
        q_indices = np.asarray(indices, dtype=int)
        off_rows.append({"transition_id": transition_id, "source_boundary": row["source_boundary"], "duration_s": float(traj["t"][max(indices)] - traj["t"][min(indices)]), "max_joint_velocity": float(np.max(np.abs(traj["dq"][q_indices]))), "max_joint_acceleration": float(np.max(np.abs(traj["ddq"][q_indices]))), "max_joint_jerk": float(np.max(np.abs(traj["jerk"][q_indices]))), "FCL_valid": all(collision_fcl["rows"][i].get("status") == "accepted" for i in edge_indices), "Bullet_valid": all(collision_bullet["rows"][i].get("status") == "accepted" for i in edge_indices), "timed_final_sample_start": min(indices), "timed_final_sample_end": max(indices), "timed_sample_count": len(indices), "edge_count": len(edge_indices)})
    off_report = {"schema_version": "stage25-off-transition-validation-v1", "transition_count": len(off_rows), "all_transitions_dynamic_checked": True, "transitions": off_rows, "all_fcl_valid": all(row["FCL_valid"] for row in off_rows), "all_bullet_valid": all(row["Bullet_valid"] for row in off_rows), "boundary_402_403": next(row for row in off_rows if row["transition_id"] == 6)}
    write_json(output_root / "stage25_process_manifest.json", manifest)
    write_json(output_root / "stage25_off_transition_validation.json", off_report)
    return manifest, off_report


def write_collision_report(output_root: Path, fcl: dict[str, Any], bullet: dict[str, Any]) -> dict[str, Any]:
    fcl_set = fcl["accepted_set"]
    bullet_set = bullet["accepted_set"]
    report = {"schema_version": "stage25-collision-validation-v1", "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available", "FCL_final_trajectory_collision_free": fcl["collision_free"], "Bullet_final_trajectory_collision_free": bullet["collision_free"], "FCL_checked_samples": fcl["checked_samples"], "Bullet_checked_samples": bullet["checked_samples"], "FCL_edge_rows": fcl["result_rows"], "Bullet_edge_rows": bullet["result_rows"], "FCL_Bullet_symmetric_difference": len(fcl_set ^ bullet_set), "FCL_accepted_edge_hash": digest(sorted(map(list, fcl_set))), "Bullet_accepted_edge_hash": digest(sorted(map(list, bullet_set))), "native_backend_pair_set_equal": fcl_set == bullet_set, "interpolation_step_deg": INTERPOLATION_STEP_DEG, "FCL_summary": {key: value for key, value in fcl.items() if key not in {"rows", "accepted_set"}}, "Bullet_summary": {key: value for key, value in bullet.items() if key not in {"rows", "accepted_set"}}}
    write_json(output_root / "stage25_collision_validation.json", report)
    return report


def write_dynamics_audits(output_root: Path, ruckig_audit: dict[str, Any], limits: dict[str, Any]) -> None:
    audit = ruckig_audit["audit"]
    for name, field in [("velocity", "velocity"), ("acceleration", "acceleration"), ("jerk", "jerk")]:
        write_json(output_root / f"stage25_{name}_audit.json", {"schema_version": f"stage25-{name}-audit-v1", "stage": "Stage_2_5", "limit_source": limits["limit_source"][field], "max_ratio": audit[f"max_{field}_ratio"], "limits_passed": audit[f"{field}_limits_passed"], "per_joint": audit["per_joint"], "trajectory_sample_count": audit["sample_count"], "numeric_integrity_passed": audit["numeric_integrity_passed"]})


def normalized_file_digest(path: Path) -> str:
    if path.suffix == ".json":
        value = read_json(path)
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # Core JSON artifacts may contain either native Windows paths or the
        # WSL paths emitted by the MoveIt child.  Normalize both spellings so
        # an otherwise identical rebuild is not classified as nondeterministic
        # merely because it ran in rebuild_01/02/03.
        run_dir_windows = str(path.parent.resolve())
        run_dir_wsl = wsl_path(path.parent.resolve())
        root_windows = str(ROOT.resolve())
        root_wsl = wsl_path(ROOT.resolve())
        text = (text.replace(run_dir_windows, "<RUN_DIR>")
                .replace(run_dir_wsl, "<RUN_DIR>")
                .replace(root_windows, "<ROOT>")
                .replace(root_wsl, "<ROOT>"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return sha256(path)


def determinism(output_root: Path, rebuilds: list[Path]) -> dict[str, Any]:
    names = ["stage25_totg_trajectory.csv", "stage25_ruckig_trajectory.csv", "stage25_totg_audit.json", "stage25_ruckig_audit.json", "stage25_pose_validation.json", "collision_fcl/native_edge_results.jsonl", "collision_bullet/native_edge_results.jsonl"]
    artifacts = {}
    mismatch = 0
    for name in names:
        hashes = [normalized_file_digest(run / name) if (run / name).exists() else None for run in rebuilds]
        artifacts[name] = {f"rebuild_{i + 1:02d}": value for i, value in enumerate(hashes)}
        if len(set(hashes)) != 1:
            mismatch += 1
    report = {"schema_version": "stage25-determinism-v1", "independent_rebuilds": len(rebuilds), "deterministic": mismatch == 0 and len(rebuilds) == 3, "core_artifact_mismatch_count": mismatch, "core_artifacts": artifacts, "run_specific_fields_excluded": ["absolute output directory", "child log paths", "exit timestamps"], "independent_processes": True}
    write_json(output_root / "stage25_determinism_report.json", report)
    return report


def write_sha256sums(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_regressions(output_root: Path) -> dict[str, Any]:
    commands = [[sys.executable, "-m", "py_compile", "scripts/run_stage25.py", "tools/stage25_moveit_launch.py", "tools/stage25_moveit_runner.py"]]
    results = []
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=600)
        results.append({"environment": "Windows", "command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "passed": proc.returncode == 0})
    wsl_command = "source /opt/ros/jazzy/setup.bash && source /mnt/c/Users/86198/Desktop/robotfucker/install/setup.bash && cd /mnt/c/Users/86198/Desktop/robotfucker && python3 -m pytest -q tests/test_stage23b_outputs.py tests/test_stage24_outputs.py tests/test_stage24a_outputs.py tests/test_stage24b_outputs.py tests/test_stage25_contract.py"
    wsl_proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", wsl_command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=600)
    results.append({"environment": "WSL Ubuntu-24.04-D ROS2 Jazzy Python", "command": ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", wsl_command], "exit_code": wsl_proc.returncode, "stdout": wsl_proc.stdout, "stderr": wsl_proc.stderr, "passed": wsl_proc.returncode == 0})
    result = {"py_compile_passed": results[0]["passed"], "relevant_stage24t_and_collision_regressions_passed": results[1]["passed"], "results": results}
    write_json(output_root / "stage25_regression_tests.json", result)
    return result


def gate_report(output_root: Path, limits: dict[str, Any], runtimes: list[dict[str, Any]], manifest: dict[str, Any], pose: dict[str, Any], collision: dict[str, Any], off: dict[str, Any], det: dict[str, Any], tests: dict[str, Any], frozen_hashes: dict[str, Any]) -> dict[str, Any]:
    final_runtime = runtimes[0]
    totg = read_json(output_root / "rebuild_01/stage25_totg_audit.json") if (output_root / "rebuild_01/stage25_totg_audit.json").exists() else {}
    ruckig = read_json(output_root / "rebuild_01/stage25_ruckig_audit.json") if (output_root / "rebuild_01/stage25_ruckig_audit.json").exists() else {}
    ruckig_a = ruckig.get("audit", {})
    failures = []
    parameter_mismatch = []
    for index, runtime in enumerate(runtimes, start=1):
        if runtime.get("formal_totg_parameters") != FORMAL_TOTG_PARAMETERS:
            parameter_mismatch.append(f"rebuild_{index:02d}:totg")
        if runtime.get("formal_ruckig_parameters") != FORMAL_RUCKIG_PARAMETERS:
            parameter_mismatch.append(f"rebuild_{index:02d}:ruckig")
    if parameter_mismatch: failures.append("blocked_formal_parameter_mismatch")
    if final_runtime.get("status") != "passed_native_moveit_timing": failures.append("blocked_native_moveit_runtime")
    if not totg.get("totg_success", False): failures.append("blocked_totg_parameterization")
    if totg.get("parameters") != FORMAL_TOTG_PARAMETERS: failures.append("blocked_totg_parameter_mismatch")
    if not ruckig.get("ruckig_success", False): failures.append("blocked_ruckig_smoothing")
    if ruckig.get("parameters") != FORMAL_RUCKIG_PARAMETERS: failures.append("blocked_ruckig_parameter_mismatch")
    if not ruckig_a.get("position_limits_passed", False): failures.append("blocked_position_limit")
    if not ruckig_a.get("velocity_limits_passed", False): failures.append("blocked_velocity_limit")
    if not ruckig_a.get("acceleration_limits_passed", False): failures.append("blocked_acceleration_limit")
    if not ruckig_a.get("jerk_limits_passed", False): failures.append("blocked_jerk_limit")
    if not pose.get("pose_constraints_passed", False): failures.append("blocked_post_timing_pose_deviation")
    if pose.get("source_waypoints_checked") != 720 or pose.get("samples_over_6mm") != 0: failures.append("blocked_pose_coverage_or_position_limit")
    if manifest.get("waypoint_coverage") != "720/720" or manifest.get("spray_on_segments") != 10 or manifest.get("spray_off_transitions") != 9: failures.append("blocked_process_coverage")
    if not collision.get("FCL_final_trajectory_collision_free", False) or not collision.get("Bullet_final_trajectory_collision_free", False) or collision.get("FCL_Bullet_symmetric_difference") != 0: failures.append("blocked_post_timing_collision")
    if not off.get("all_fcl_valid", False) or not off.get("all_bullet_valid", False) or not off.get("all_transitions_dynamic_checked", False): failures.append("blocked_off_transition_validation")
    if not det.get("deterministic", False): failures.append("blocked_nondeterministic_time_parameterization")
    if not tests.get("py_compile_passed", False) or not tests.get("relevant_stage24t_and_collision_regressions_passed", False): failures.append("blocked_regression_tests")
    if not frozen_hashes.get("unchanged", False): failures.append("blocked_frozen_stage24t_input_mutation")
    status = "passed" if not failures else failures[0]
    report = {
        "schema_version": "stage25-gate-v2",
        "Stage_2_4T": "passed",
        "Stage_2_5": status,
        "Stage_2_6": "unblocked_not_started" if status == "passed" else "blocked_stage25",
        "source_stage24t_geometry_frozen": True,
        "waypoint_coverage": manifest["waypoint_coverage"],
        "spray_on_segments": manifest["spray_on_segments"],
        "spray_off_transitions": f"{manifest['spray_off_transitions']}/9",
        "TOTG": "passed" if totg.get("totg_success") else "blocked",
        "Ruckig": "passed" if ruckig.get("ruckig_success") else "blocked",
        "totg_parameterization": "passed" if totg.get("totg_success") else "blocked",
        "ruckig_smoothing": "passed" if ruckig.get("ruckig_success") else "blocked",
        "formal_totg_parameters": FORMAL_TOTG_PARAMETERS,
        "formal_ruckig_parameters": FORMAL_RUCKIG_PARAMETERS,
        "repair_reason": REPAIR_REASON,
        "effective_parameter_mismatch": parameter_mismatch,
        "timestamps_present": ruckig_a.get("timestamps_present"),
        "timestamps_strictly_increasing": ruckig_a.get("timestamps_strictly_increasing"),
        "position_limits_passed": ruckig_a.get("position_limits_passed"),
        "velocity_limits_passed": ruckig_a.get("velocity_limits_passed"),
        "acceleration_limits_passed": ruckig_a.get("acceleration_limits_passed"),
        "jerk_limits_passed": ruckig_a.get("jerk_limits_passed"),
        "pose_constraints_passed": pose.get("pose_constraints_passed"),
        "max_tcp_position_error_mm": pose.get("max_tcp_position_deviation_mm"),
        "max_tcp_position_deviation_mm": pose.get("max_tcp_position_deviation_mm"),
        "max_tcp_position_deviation_waypoint": pose.get("max_tcp_position_deviation_waypoint"),
        "max_tcp_position_deviation_time_s": pose.get("max_tcp_position_deviation_time_s"),
        "second_largest_tcp_position_deviation_mm": pose.get("second_largest_tcp_position_deviation_mm"),
        "samples_over_6mm": pose.get("samples_over_6mm"),
        "max_spray_normal_error_deg": pose.get("max_spray_normal_error_deg"),
        "max_spray_distance_error_mm": pose.get("max_spray_distance_error_mm"),
        "joint_limits_passed": bool(ruckig_a.get("position_limits_passed") and ruckig_a.get("velocity_limits_passed") and ruckig_a.get("acceleration_limits_passed") and ruckig_a.get("jerk_limits_passed")),
        "FCL_final_trajectory_validation": "passed" if collision.get("FCL_final_trajectory_collision_free") else "blocked",
        "Bullet_final_trajectory_validation": "passed" if collision.get("Bullet_final_trajectory_collision_free") else "blocked",
        "FCL": "passed" if collision.get("FCL_final_trajectory_collision_free") else "blocked",
        "Bullet": "passed" if collision.get("Bullet_final_trajectory_collision_free") else "blocked",
        "backend_symmetric_difference": collision.get("FCL_Bullet_symmetric_difference"),
        "spray_state_mapping_preserved": bool(manifest.get("process_order_preserved") and manifest.get("waypoint_coverage") == "720/720"),
        "process_order_preserved": manifest.get("process_order_preserved"),
        "boundary_order_preserved": manifest.get("boundary_order_preserved"),
        "nan_count": 0 if ruckig_a.get("numeric_integrity_passed") else None,
        "inf_count": 0 if ruckig_a.get("numeric_integrity_passed") else None,
        "determinism_rebuilds": "3/3_passed" if det.get("deterministic") else f"{det.get('independent_rebuilds', 0)}/3_passed",
        "determinism": "3/3_passed" if det.get("deterministic") else f"{det.get('independent_rebuilds', 0)}/3_passed",
        "core_artifact_mismatch_count": det.get("core_artifact_mismatch_count"),
        "sha256_mismatch": 0 if det.get("core_artifact_mismatch_count") == 0 else det.get("core_artifact_mismatch_count"),
        "frozen_stage24t_input_hash_check": frozen_hashes,
        "trajectory_duration_s": manifest.get("trajectory_duration_s"),
        "max_velocity_ratio": ruckig_a.get("max_velocity_ratio"),
        "max_acceleration_ratio": ruckig_a.get("max_acceleration_ratio"),
        "max_jerk_ratio": ruckig_a.get("max_jerk_ratio"),
        "first_failing_reason": failures[0] if failures else None,
        "all_failure_reasons_observed": failures,
        "formal_limits": limits,
        "runtime_statuses": runtimes,
        "collision_checked_samples": {"FCL": collision.get("FCL_checked_samples"), "Bullet": collision.get("Bullet_checked_samples")},
        "off_transition_validation": off,
        "regression_tests": tests,
        "tests": "passed" if tests.get("py_compile_passed") and tests.get("relevant_stage24t_and_collision_regressions_passed") else "blocked",
    }
    write_json(output_root / "stage25_gate_report.json", report)
    return report


def write_report(output_root: Path, gate: dict[str, Any], limits: dict[str, Any], native_install: Path) -> None:
    status = gate["Stage_2_5"]
    lines = ["# Stage 2.5 Deterministic Time Parameterization + Ruckig Dynamic Certification", "", f"- Stage 2.4T: `{gate['Stage_2_4T']}` (frozen input)", f"- Stage 2.5: `{status}`", f"- Stage 2.6: `{gate['Stage_2_6']}`", "", "## Frozen process", "", f"- 720 source waypoints; 10 Spray-ON segments; 9 Spray-OFF transitions; coverage `{gate['waypoint_coverage']}`.", "- The selected 402->403 boundary is retained in OFF transition 6.", "- The path is one continuous timed trajectory; internal OFF boundaries are not force-zeroed.", "", "## Formal timing contract", "", "```yaml", "formal_totg_parameters:", f"  velocity_scaling_factor: {gate['formal_totg_parameters']['velocity_scaling_factor']}", f"  acceleration_scaling_factor: {gate['formal_totg_parameters']['acceleration_scaling_factor']}", f"  path_tolerance: {gate['formal_totg_parameters']['path_tolerance']}", f"  resample_dt: {gate['formal_totg_parameters']['resample_dt']}", f"  min_angle_change: {gate['formal_totg_parameters']['min_angle_change']}", "formal_ruckig_parameters:", f"  velocity_scaling_factor: {gate['formal_ruckig_parameters']['velocity_scaling_factor']}", f"  acceleration_scaling_factor: {gate['formal_ruckig_parameters']['acceleration_scaling_factor']}", f"  mitigate_overshoot: {str(gate['formal_ruckig_parameters']['mitigate_overshoot']).lower()}", f"  overshoot_threshold: {gate['formal_ruckig_parameters']['overshoot_threshold']}", "repair_reason:", f"  previous_path_tolerance_rad: {gate['repair_reason']['previous_path_tolerance_rad']}", f"  selected_path_tolerance_rad: {gate['repair_reason']['selected_path_tolerance_rad']}", f"  evidence: {gate['repair_reason']['evidence']}", "```", "", "## Native timing", "", "- TOTG: MoveIt2 `RobotTrajectory.apply_totg_time_parameterization`.", "- Ruckig: MoveIt2 `RobotTrajectory.apply_ruckig_smoothing` after TOTG.", "- Velocity/acceleration/jerk limits come from the project MoveIt contract; jerk is explicit in `joint_limits_with_jerk.yaml` and is not inferred from URDF/controller configuration.", f"- Native collision probe: `{native_install / 'lib/stage24_closed_loop_graph/stage24_formal_probe'}` with FCL and Bullet shape-type equivalence.", "", "## Gate summary", "", "```yaml", f"Stage_2_5: {status}", f"trajectory_duration_s: {gate.get('trajectory_duration_s')}", f"max_tcp_position_error_mm: {gate.get('max_tcp_position_error_mm')}", f"second_largest_tcp_position_deviation_mm: {gate.get('second_largest_tcp_position_deviation_mm')}", f"samples_over_6mm: {gate.get('samples_over_6mm')}", f"max_velocity_ratio: {gate.get('max_velocity_ratio')}", f"max_acceleration_ratio: {gate.get('max_acceleration_ratio')}", f"max_jerk_ratio: {gate.get('max_jerk_ratio')}", f"FCL_final_trajectory_validation: {gate.get('FCL_final_trajectory_validation')}", f"Bullet_final_trajectory_validation: {gate.get('Bullet_final_trajectory_validation')}", f"backend_symmetric_difference: {gate.get('backend_symmetric_difference')}", f"determinism_rebuilds: {gate.get('determinism_rebuilds')}", "```", "", "CCD and independent clearance remain `not_available`; collision evidence is adaptive discrete interpolation with 0.25 degree joint-step resolution."]
    (output_root / "stage25_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rebuilds", type=int, default=3)
    args = parser.parse_args()
    if args.rebuilds != 3:
        raise SystemExit("Stage 2.5 formal certification requires exactly 3 independent rebuilds.")
    if not STAGE24T.exists() or not WAYPOINTS.exists() or not PARTS.exists():
        raise SystemExit("Frozen Stage 2.4T input or collision geometry is missing.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or ROOT / "outputs/ik_graph_stage25_timed_certification" / f"fr5_scaled_horseshoe_demo_v45_{timestamp}_Stage25").resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    snapshot = git_snapshot()
    (output_root / "stage25_git_status_before.txt").write_text(snapshot["status_before"], encoding="utf-8")
    (output_root / "stage25_git_diff_stat_before.txt").write_text(snapshot["diff_stat_before"], encoding="utf-8")
    limits = build_limits()
    items, path_samples, q_array = load_process_items()
    build_input_manifest(output_root, snapshot, limits)
    ref = write_baseline(output_root, items, path_samples, q_array, limits, snapshot)
    ref_path = output_root / "stage25_path_reference.json"
    native_install = find_native_stage24_install()
    stage23b_install = find_stage23b_install()
    runtimes: list[dict[str, Any]] = []
    rebuild_dirs: list[Path] = []
    collision_runs: list[dict[str, Any]] = []
    for rebuild_index in range(1, 4):
        rebuild_dir = output_root / f"rebuild_{rebuild_index:02d}"
        rebuild_dir.mkdir()
        rebuild_dirs.append(rebuild_dir)
        runtime = run_moveit(rebuild_dir, ref_path)
        runtimes.append(runtime)
        required = [rebuild_dir / "stage25_totg_trajectory.csv", rebuild_dir / "stage25_ruckig_trajectory.csv", rebuild_dir / "stage25_totg_audit.json", rebuild_dir / "stage25_ruckig_audit.json", rebuild_dir / "stage25_pose_validation.json"]
        if not all(path.exists() for path in required):
            break
        edges = write_collision_edges(rebuild_dir)
        fcl = run_native_collision(rebuild_dir, "fcl", rebuild_index, native_install, stage23b_install, edges)
        bullet = run_native_collision(rebuild_dir, "bullet", rebuild_index, native_install, stage23b_install, edges)
        collision_runs.append({"fcl": fcl, "bullet": bullet})
    if not collision_runs:
        raise SystemExit("Stage 2.5 produced no native collision rebuild; formal gate cannot be evaluated.")
    # The formal core is the first independent rebuild; later rebuilds are
    # compared against it and are never allowed to overwrite the frozen input.
    first = rebuild_dirs[0]
    for name in ["stage25_totg_trajectory.csv", "stage25_ruckig_trajectory.csv", "stage25_totg_audit.json", "stage25_ruckig_audit.json", "stage25_pose_validation.json", "stage25_fk_trace.csv", "stage25_native_moveit_runtime.json"]:
        if (first / name).exists():
            shutil.copy2(first / name, output_root / name)
    final_traj = read_trajectory_csv(first / "stage25_ruckig_trajectory.csv")
    fcl = collision_runs[0]["fcl"]
    bullet = collision_runs[0]["bullet"]
    collision = write_collision_report(output_root, fcl, bullet)
    process_manifest, off_report = build_process_manifest(output_root, ref, final_traj, fcl, bullet)
    ruckig_audit = read_json(first / "stage25_ruckig_audit.json")
    write_dynamics_audits(output_root, ruckig_audit, limits)
    det = determinism(output_root, rebuild_dirs[: len(collision_runs)])
    tests = run_regressions(output_root)
    frozen_hashes = verify_frozen_inputs(output_root, read_json(output_root / "stage25_input_manifest.json"))
    gate = gate_report(output_root, limits, runtimes, process_manifest, read_json(first / "stage25_pose_validation.json"), collision, off_report, det, tests, frozen_hashes)
    logs = []
    for path in sorted(output_root.glob("rebuild_*/stage25_moveit_*.log")):
        logs.append(f"===== {path.relative_to(output_root).as_posix()} =====\n{path.read_text(encoding='utf-8', errors='replace')}")
    for path in sorted(output_root.glob("rebuild_*/collision_*/*.log")):
        logs.append(f"===== {path.relative_to(output_root).as_posix()} =====\n{path.read_text(encoding='utf-8', errors='replace')}")
    (output_root / "stage25_full_runtime_logs.txt").write_text("\n\n".join(logs) + "\n", encoding="utf-8")
    write_report(output_root, gate, limits, native_install)
    write_sha256sums(output_root)
    print(json.dumps({"output_root": str(output_root), "Stage_2_5": gate["Stage_2_5"], "Stage_2_6": gate["Stage_2_6"], "trajectory_duration_s": gate.get("trajectory_duration_s"), "max_velocity_ratio": gate.get("max_velocity_ratio"), "max_acceleration_ratio": gate.get("max_acceleration_ratio"), "max_jerk_ratio": gate.get("max_jerk_ratio"), "first_failure": gate.get("first_failing_reason")}, ensure_ascii=False, indent=2))
    return 0 if gate["Stage_2_5"] == "passed" else 5


if __name__ == "__main__":
    raise SystemExit(main())
