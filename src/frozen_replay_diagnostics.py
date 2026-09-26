"""Shared frozen-input and replay utilities for Stage 1.9.1.

This module is ROS-independent so corpus construction, hashing, comparisons,
and tests can run without silently substituting a fake IK backend.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.canonical_graph_hash import artifact_sha256, semantic_hash
from src.deterministic_ik_candidates import canonical_json, deduplicate_ik_records, explicit_seed_templates, quantize, sha256_canonical
from src.task_pose_repair import TaskPoseRepairConfig, generate_task_pose_candidates, read_pose_csv, recover_surface_frames


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "ik_graph_stage191"
PROTECTED_DIRS = ["outputs/ik_graph", "outputs/ik_graph_diagnostics", "outputs/ik_feasibility_ablation_stage1_6", "outputs/ik_graph_stage17", "outputs/ik_graph_stage18", "outputs/ik_graph_stage19"]
NOISE_KEYS = {"timestamp", "timestamp_utc", "created_time_utc", "elapsed_time_s", "elapsed_seconds", "pid", "process_id", "log_id", "discovery_order"}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_parquet(path: str | Path, rows: Sequence[Mapping[str, Any]], empty_columns: Sequence[str] | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        parquet.write_table(pa.Table.from_pylist([json_safe(dict(row)) for row in rows]), path)
    else:
        names = list(empty_columns or ["status"])
        parquet.write_table(pa.Table.from_pydict({name: pa.array([], type=pa.string()) for name in names}), path)


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return [dict(row) for row in parquet.read_table(path).to_pylist()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(json_safe(value)).encode("utf-8"))


def quantized_hash(values: Iterable[float], step: float) -> str:
    return raw_hash([quantize(float(value), step) for value in values])


def protected_hashes(root: Path = ROOT) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    groups: dict[str, Any] = {}
    for relative in PROTECTED_DIRS:
        directory = root / relative
        members = []
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    item = {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": artifact_sha256(path)}
                    members.append(item)
                    files[item["path"]] = item
        groups[relative] = {"path": relative, "exists": directory.exists(), "file_count": len(members), "files": members}
    return {"schema_version": "1.0", "files": files, "groups": groups}


def protected_hashes_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return dict(before.get("files", {})) == dict(after.get("files", {}))


def path_hash(path: str | Path) -> str | None:
    path = Path(path)
    return artifact_sha256(path) if path.exists() and path.is_file() else None


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_seed_rows(path: str | Path) -> np.ndarray:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [[float(row[f"q{i}"]) for i in range(1, 7)] for row in rows]
    return np.asarray(values, dtype=float)


def frozen_task_layers(config_path: str | Path) -> tuple[list[list[dict[str, Any]]], np.ndarray, str]:
    raw = load_yaml(config_path)
    stage17 = load_yaml(ROOT / raw["inputs"]["stage17_config"])
    repair_config = TaskPoseRepairConfig.from_mapping(stage17)
    pose_path = ROOT / raw["inputs"]["tcp_pose_csv"]
    rows, input_hash = read_pose_csv(pose_path, 181)
    frames, nominal_q, _ = recover_surface_frames(rows, repair_config.nominal_standoff_m, repair_config.tcp_points_to_wall)
    task_layers_raw, _ = generate_task_pose_candidates(frames, nominal_q, repair_config, source_phase_id="stage_1_9_1_frozen_task_pose")
    layers: list[list[dict[str, Any]]] = []
    for layer in task_layers_raw:
        records = []
        for candidate in layer:
            record = candidate.to_record()
            stable_key = {"waypoint_id": int(record["waypoint_id"]), "tangential_offset_mm": float(record.get("tangential_offset_mm", 0.0)), "longitudinal_offset_mm": float(record.get("longitudinal_offset_mm", 0.0)), "standoff_offset_mm": float(record.get("standoff_offset_mm", 0.0)), "roll_offset_deg": float(record.get("roll_offset_deg", 0.0)), "generation_type": str(record.get("generation_strategy", record.get("generation_type", "unknown")))}
            from src.deterministic_ik_candidates import stable_task_pose_identity
            encoded, stable_id = stable_task_pose_identity(record)
            record.update({"task_pose_stable_key": encoded, "task_pose_stable_id": stable_id, "target_pose_raw_sha256": raw_hash({"position": record.get("repaired_tcp_position_xyz_m"), "quaternion": record.get("repaired_quaternion_xyzw")}), "target_pose_semantic_hash": sha256_canonical(stable_key)})
            records.append(record)
        records.sort(key=lambda item: str(item["task_pose_stable_id"]))
        layers.append(records)
    return layers, load_seed_rows(ROOT / raw["inputs"]["seed_joint_csv"]), input_hash


def build_frozen_corpus(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    raw = load_yaml(config_path)
    layers, seeds, input_hash = frozen_task_layers(config_path)
    if len(layers) != 181 or len(seeds) != 181:
        raise RuntimeError(f"authoritative open-arch input must be 181/181, got {len(layers)}/{len(seeds)}")
    ik_cfg = load_yaml(ROOT / raw["inputs"]["stage19_config"])["deterministic_ik"]
    model_hashes = {
        "robot_model_sha256": path_hash(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"),
        "urdf_sha256": path_hash(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"),
        "srdf_sha256": path_hash(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"),
        "kinematics_sha256": path_hash(ROOT / "config/stage19_deterministic_ik.yaml"),
        "joint_limits_sha256": path_hash(ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"),
    }
    calls: list[dict[str, Any]] = []
    call_order = 0
    all_task_records = [record for layer in layers for record in layer]
    for waypoint, layer in enumerate(layers):
        previous_seed = seeds[waypoint - 1] if waypoint > 0 else None
        templates = explicit_seed_templates(waypoint, seeds[waypoint], previous_seed, config=ik_cfg)
        for task in layer:
            for template in templates:
                call_order += 1
                pose = {"position": task.get("repaired_tcp_position_xyz_m"), "quaternion": task.get("repaired_quaternion_xyzw")}
                seed = list(template.seed_joint_vector)
                calls.append({
                    "call_id": f"call:{call_order:08d}", "call_order": call_order, "waypoint_id": waypoint, "source_phase_id": task.get("source_phase_id", "stage_1_9_1_frozen_task_pose"), "source_csv_row": waypoint + 2,
                    "task_pose_candidate_id": task.get("task_pose_candidate_id"), "task_pose_stable_id": task["task_pose_stable_id"], "task_pose_stable_key": task["task_pose_stable_key"], "task_pose_generation_type": task.get("generation_strategy", task.get("generation_type", "unknown")), "repair_window_id": task.get("repair_window_id"),
                    "target_x": float(pose["position"][0]), "target_y": float(pose["position"][1]), "target_z": float(pose["position"][2]), "target_qx": float(pose["quaternion"][0]), "target_qy": float(pose["quaternion"][1]), "target_qz": float(pose["quaternion"][2]), "target_qw": float(pose["quaternion"][3]), "target_pose_raw_sha256": task["target_pose_raw_sha256"], "target_pose_semantic_hash": task["target_pose_semantic_hash"],
                    "seed_template_id": template.seed_template_id, "seed_template_family": template.seed_template_family, "seed_order_index": template.seed_order_index, "seed_q1": seed[0], "seed_q2": seed[1], "seed_q3": seed[2], "seed_q4": seed[3], "seed_q5": seed[4], "seed_q6": seed[5], "seed_raw_sha256": raw_hash(seed), "seed_provenance": template.to_record(),
                    **model_hashes,
                    "solver_plugin_name": "kdl", "solver_api_entry": "RobotState.set_from_ik", "solver_options": {"timeout_s": 0.0, "attempt_count": 1, "random_restart": False}, "solver_options_sha256": raw_hash({"timeout_s": 0.0, "attempt_count": 1, "random_restart": False}), "joint_group": raw["robot"]["group_name"], "tip_link": raw["robot"]["ee_link"], "frame_id": "base_link", "input_corpus_sha256": input_hash,
                })
    write_parquet(output_path, calls)
    # Hash the persisted table, not only the pre-write Python objects.  Arrow
    # normalizes nested map fields (including absent seed parameters) on write;
    # the read-back representation is the reproducible corpus contract.
    persisted = read_parquet(output_path)
    semantic = semantic_hash(persisted, sort_fields=("call_order", "call_id"), numeric_step=1.0e-12)
    return {"call_count": len(calls), "waypoint_count": 181, "task_pose_count": len(all_task_records), "semantic_hash": semantic, "file_sha256": artifact_sha256(output_path), "input_tcp_sha256": input_hash, "input_seed_sha256": path_hash(ROOT / raw["inputs"]["seed_joint_csv"])}


def compare_replays(records_by_run: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    fields = ["target_pose_raw_sha256", "seed_raw_sha256", "initial_robot_state_raw_sha256", "solver_options_sha256", "solution_raw_sha256", "solution_quantized_sha256", "solver_success"]
    by_call: dict[str, Any] = {}
    for run_id, records in records_by_run.items():
        for row in records:
            by_call.setdefault(str(row["call_id"]), {}).setdefault(run_id, []).append(row)
    first_difference = None
    classifications = {}
    for call_id in sorted(by_call):
        observations = [row for run in by_call[call_id].values() for row in run]
        input_same = len({tuple(row.get(field) for field in fields[:4]) for row in observations}) == 1
        raw_same = len({row.get("solution_raw_sha256") for row in observations}) == 1
        quant_same = len({row.get("solution_quantized_sha256") for row in observations}) == 1
        success_same = len({row.get("solver_success") for row in observations}) == 1
        if input_same and not raw_same and quant_same:
            classification = "raw_output_changed_quantized_equal"
        elif input_same and not quant_same:
            classification = "quantized_output_changed"
        elif not input_same:
            classification = "input_changed"
        elif not success_same:
            classification = "success_status_changed"
        else:
            classification = "fully_deterministic"
        classifications[call_id] = classification
        if classification != "fully_deterministic" and first_difference is None:
            first_difference = call_id
    return {"first_differing_record_id": first_difference, "classifications": classifications, "all_inputs_same": all(value != "input_changed" for value in classifications.values()), "all_fully_deterministic": all(value == "fully_deterministic" for value in classifications.values())}


def classify_first_layer(*, ik_comparison: Mapping[str, Any], node_hashes: Sequence[str], edge_hashes: Sequence[str], dp_hashes: Sequence[str], ruckig_hashes: Sequence[str], post_hashes: Sequence[str]) -> str | None:
    if not ik_comparison.get("all_inputs_same", False) or ik_comparison.get("first_differing_record_id"):
        first = ik_comparison.get("first_differing_record_id")
        if first and any(value == "input_changed" for value in ik_comparison.get("classifications", {}).values()):
            return "ik_call_input_generation"
        if first and any(value == "raw_output_changed_quantized_equal" for value in ik_comparison.get("classifications", {}).values()):
            return "low_level_numeric_noise"
        return "ik_solver_or_runtime"
    if len(set(node_hashes)) > 1:
        return "candidate_collection_dedup_or_sort"
    if len(set(edge_hashes)) > 1:
        return "edge_construction_or_collision_validation"
    if len(set(dp_hashes)) > 1:
        return "dp_search"
    if len(set(ruckig_hashes)) > 1:
        return "ruckig"
    if len(set(post_hashes)) > 1:
        return "post_validation_or_planning_scene"
    return None
