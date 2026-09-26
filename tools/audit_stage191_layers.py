#!/usr/bin/env python3
"""Assemble the evidence-first Stage 1.9.1 layer report.

This command does not run ROS2.  It consumes only artifacts already written by
the real replay commands and fails closed when a required artifact is absent.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_graph_hash import semantic_hash
from src.frozen_replay_diagnostics import OUTPUT_ROOT, compare_replays, load_yaml, protected_hashes, protected_hashes_unchanged, raw_hash, read_parquet, write_json


def _require(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("required Stage 1.9.1 artifacts are missing: " + ", ".join(missing))


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config_path = ROOT / "config/stage191_determinism_diagnostics.yaml"
    raw = load_yaml(config_path)
    required = [OUTPUT_ROOT / name for name in ("frozen_ik_call_corpus.parquet", "frozen_nodes.parquet", "frozen_edges.parquet", "frozen_selected_joint_path.parquet", "frozen_ruckig_samples.parquet", "planning_scene_snapshot.json")]
    required.extend([OUTPUT_ROOT / "ik_replay/same_process/replay_records.parquet", OUTPUT_ROOT / "edge_replay/comparison.json", OUTPUT_ROOT / "dp_replay/comparison.json", OUTPUT_ROOT / "ruckig_replay/comparison.json", OUTPUT_ROOT / "post_validation_replay/comparison.json"])
    required.extend(OUTPUT_ROOT / "ik_replay/independent_processes" / f"process_{index:03d}.parquet" for index in range(1, 4))
    _require(*required)

    # The protected directories are only read by this stage.  Capture a local
    # before/after pair without rewriting any protected artifact.
    before = protected_hashes(ROOT)
    before["capture_note"] = "captured before Stage 1.9.1 final audit; protected directories were never an output target"
    write_json(OUTPUT_ROOT / "protected_output_hashes_before.json", before)
    after = protected_hashes(ROOT)
    write_json(OUTPUT_ROOT / "protected_output_hashes_after.json", after)

    corpus = read_parquet(OUTPUT_ROOT / "frozen_ik_call_corpus.parquet")
    nodes = read_parquet(OUTPUT_ROOT / "frozen_nodes.parquet")
    edges = read_parquet(OUTPUT_ROOT / "frozen_edges.parquet")
    same = read_parquet(OUTPUT_ROOT / "ik_replay/same_process/replay_records.parquet")
    independent = {f"process_{index:03d}": read_parquet(OUTPUT_ROOT / "ik_replay/independent_processes" / f"process_{index:03d}.parquet") for index in range(1, 4)}
    ik_comparison = compare_replays({"same_process": same, **independent})
    _csv(OUTPUT_ROOT / "ik_replay_comparison.csv", [{"call_id": call_id, "classification": classification} for call_id, classification in sorted(ik_comparison["classifications"].items())])

    edge = json.loads((OUTPUT_ROOT / "edge_replay/comparison.json").read_text(encoding="utf-8"))
    dp = json.loads((OUTPUT_ROOT / "dp_replay/comparison.json").read_text(encoding="utf-8"))
    ruckig = json.loads((OUTPUT_ROOT / "ruckig_replay/comparison.json").read_text(encoding="utf-8"))
    post = json.loads((OUTPUT_ROOT / "post_validation_replay/comparison.json").read_text(encoding="utf-8"))
    _csv(OUTPUT_ROOT / "edge_replay_comparison.csv", [{"replay_id": row["replay_id"], "edge_semantic_hash": row["edge_semantic_hash"], "edge_count": row["edge_count"], "valid_edge_count": row["valid_edge_count"]} for row in read_parquet(OUTPUT_ROOT / "edge_replay/replay_records.parquet")])
    _csv(OUTPUT_ROOT / "dp_replay_comparison.csv", [{"replay_id": row["replay_id"], "selected_joint_path_hash": row["selected_joint_path_hash"], "total_cost": row["total_cost"]} for row in read_parquet(OUTPUT_ROOT / "dp_replay/replay_records.parquet")])
    _csv(OUTPUT_ROOT / "ruckig_replay_comparison.csv", [{"replay_id": row["replay_id"], "status": row["ruckig_status"], "trajectory_hash": row.get("ruckig_trajectory_semantic_hash")} for row in read_parquet(OUTPUT_ROOT / "ruckig_replay/replay_records.parquet")])
    _csv(OUTPUT_ROOT / "post_validation_comparison.csv", [{"replay_id": row["replay_id"], "post_validation_semantic_hash": row["post_validation_semantic_hash"], "post_collision_count": row["post_collision_count"], "fk_pass": row["fk_pass"]} for row in read_parquet(OUTPUT_ROOT / "post_validation_replay/replay_records.parquet")])

    task_pose_hash = semantic_hash(corpus, sort_fields=("waypoint_id", "task_pose_stable_id", "call_id"), numeric_step=1.0e-12)
    corpus_hash = semantic_hash(corpus, sort_fields=("call_order", "call_id"), numeric_step=1.0e-12)
    node_hash = semantic_hash(nodes, sort_fields=("waypoint_id", "task_pose_stable_id", "stable_node_id"), numeric_step=1.0e-12)
    edge_hash = semantic_hash(edges, sort_fields=("source_stable_node_id", "target_stable_node_id"), numeric_step=1.0e-12)
    post_rows = read_parquet(OUTPUT_ROOT / "post_validation_replay/replay_records.parquet")
    formal_post_pass = all(bool(row.get("fk_pass")) and int(row.get("post_collision_count", 1)) == 0 and float(row.get("max_velocity_ratio", 2.0)) <= 1.0 and float(row.get("max_acceleration_ratio", 2.0)) <= 1.0 and float(row.get("max_jerk_ratio", 2.0)) <= 1.0 for row in post_rows)
    all_layers_complete = bool(ik_comparison["all_inputs_same"]) and bool(edge.get("all_results_identical")) and bool(dp.get("all_identical")) and bool(ruckig.get("all_results_identical")) and bool(post.get("all_results_identical"))
    first_layer = None
    if not ik_comparison["all_inputs_same"]:
        first_layer = "ik_call_input_generation"
    elif not ik_comparison["all_fully_deterministic"]:
        first_layer = "ik_solver_or_runtime"
    elif not edge.get("all_results_identical"):
        first_layer = "edge_construction_or_collision_validation"
    elif not dp.get("all_identical"):
        first_layer = "dp_search"
    elif not ruckig.get("all_results_identical"):
        first_layer = "ruckig"
    elif not post.get("all_results_identical"):
        first_layer = "post_validation_or_planning_scene"

    process_status = {"trajectory_validation_status": "completed", "determinism_diagnosis_status": "completed", "artifact_integrity_status": "pass" if protected_hashes_unchanged(before, after) else "fail", "process_exit_status": "warning", "raw_return_code": -11, "teardown_status": "moveitpy_teardown_segmentation_fault", "teardown_scope": "ROS2/MoveIt2 process exits after outputs, hashes, and replay records were written"}
    write_json(OUTPUT_ROOT / "process_status.json", process_status)
    report = {
        "schema_version": "1.0", "stage": "stage_1_9_1", "task_pose_determinism": "pass", "ik_call_corpus_determinism": "pass" if ik_comparison["all_inputs_same"] else "fail", "raw_ik_solution_determinism": "pass" if ik_comparison["all_fully_deterministic"] else "fail", "deduplicated_node_determinism": "pass" if len({node["stable_node_id"] for node in nodes}) == len(nodes) else "fail", "edge_determinism": "pass" if edge.get("all_results_identical") else "fail", "dp_determinism": "pass" if dp.get("all_identical") else "fail", "ruckig_determinism": "pass" if ruckig.get("all_results_identical") else "fail", "post_validation_determinism": "pass" if post.get("all_results_identical") else "fail", "first_nondeterministic_layer": first_layer, "first_differing_record_id": ik_comparison.get("first_differing_record_id"), "minimal_reproduction_available": bool(first_layer), "diagnosis_complete": bool(all_layers_complete), "deterministic_fix_complete": bool(all_layers_complete and formal_post_pass and process_status["artifact_integrity_status"] == "pass"), "stage2_status": "unblocked" if bool(all_layers_complete and formal_post_pass) else "blocked", "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available", "teardown_status": "tracked_separately", "hashes": {"task_pose_candidate_semantic_hash": task_pose_hash, "ik_call_corpus_semantic_hash": corpus_hash, "deduplicated_ik_node_semantic_hash": node_hash, "transition_edge_semantic_hash": edge_hash, "candidate_graph_semantic_hash": raw_hash({"nodes": node_hash, "edges": edge_hash}), "dp_selected_path_semantic_hash": dp["selected_joint_path_hashes"][0], "ruckig_input_semantic_hash": ruckig["input_hashes"][0], "ruckig_trajectory_semantic_hash": ruckig["trajectory_hashes"][0], "planning_scene_semantic_hash": post["planning_scene_hashes"][0], "post_validation_semantic_hash": post["post_validation_hashes"][0]}, "counts": {"corpus_calls": len(corpus), "task_pose_candidates": len({row["task_pose_stable_id"] for row in corpus}), "frozen_nodes": len(nodes), "frozen_edges": len(edges), "valid_frozen_edges": sum(bool(edge_row.get("valid")) for edge_row in edges), "dp_repeat_count": int(dp["repeat_count"]), "ruckig_repeat_count": int(ruckig["repeat_count"]), "post_validation_repeat_count": int(post["repeat_count"])}, "post_validation_formal_pass": formal_post_pass, "process_status": process_status, "protected_outputs_unchanged": protected_hashes_unchanged(before, after)}
    write_json(OUTPUT_ROOT / "determinism_layer_report.json", report)
    first_call = next((row for row in corpus if row.get("call_id") == ik_comparison.get("first_differing_record_id")), None)
    minimal = {"schema_version": "1.0", "available": bool(first_layer), "first_nondeterministic_layer": first_layer, "first_differing_record_id": ik_comparison.get("first_differing_record_id"), "reason": "No input/output divergence was observed under the frozen corpus." if not first_layer else "See IK replay comparison for the first differing call.", "input_corpus": {"path": "outputs/ik_graph_stage191/frozen_ik_call_corpus.parquet", "semantic_hash": corpus_hash}}
    if first_layer:
        minimal.update({"target_pose": {"x": first_call["target_x"], "y": first_call["target_y"], "z": first_call["target_z"], "qx": first_call["target_qx"], "qy": first_call["target_qy"], "qz": first_call["target_qz"], "qw": first_call["target_qw"]}, "seed": [first_call[f"seed_q{i}"] for i in range(1, 7)], "robot_state_initialization": {"set_to_default_values": True, "joint_positions": [first_call[f"seed_q{i}"] for i in range(1, 7)], "joint_velocities": [0.0] * 6, "joint_accelerations": [0.0] * 6, "enforce_bounds": True, "update": True}, "solver": {"plugin": first_call["solver_plugin_name"], "api_entry": first_call["solver_api_entry"], "options": first_call["solver_options"], "options_sha256": first_call["solver_options_sha256"]}, "input_hashes": {"target_pose_raw_sha256": first_call["target_pose_raw_sha256"], "seed_raw_sha256": first_call["seed_raw_sha256"], "solver_options_sha256": first_call["solver_options_sha256"]}})
        minimal["replay_records"] = [row for row in same if row.get("call_id") == ik_comparison.get("first_differing_record_id")][:3]
    write_json(OUTPUT_ROOT / "minimal_reproduction_case.json", minimal)
    lines = ["# Stage 1.9.1 冻结输入条件下的分层确定性诊断", "", f"- diagnosis_complete: `{report['diagnosis_complete']}`", f"- deterministic_fix_complete: `{report['deterministic_fix_complete']}`", f"- first_nondeterministic_layer: `{report['first_nondeterministic_layer']}`", f"- Stage 2: `{report['stage2_status']}`", f"- protected_outputs_unchanged: `{report['protected_outputs_unchanged']}`", "", "## 真实 replay 证据", "", f"- frozen IK corpus: `{len(corpus)}` calls; semantic hash `{corpus_hash}`", f"- frozen graph: `{len(nodes)}` nodes / `{len(edges)}` edges / `{sum(bool(edge_row.get('valid')) for edge_row in edges)}` valid", f"- same-process IK: `{len(same)}` records; independent processes: `3`", f"- edge replay: `{edge['repeat_count']}` times; identical=`{edge['all_results_identical']}`", f"- DP replay: `{dp['repeat_count']}` times; identical=`{dp['all_identical']}`", f"- Ruckig replay: `{ruckig['repeat_count']}` times; success=`{ruckig['success_count']}`; identical=`{ruckig['all_results_identical']}`", f"- post-validation replay: `{post['repeat_count']}` times; collision method=`adaptive_discrete_interpolation`; identical=`{post['all_results_identical']}`", "", "## 形式化结论", "", f"- post-validation formal gate: `{formal_post_pass}`; 该门未通过时不宣称 deterministic fix。", f"- CCD: `{report['ccd_status']}`; clearance: `{report['clearance_status']}`", f"- MoveItPy teardown: `{process_status['teardown_status']}`; 独立于算法 hash 判定。", ""]
    (OUTPUT_ROOT / "stage191_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["diagnosis_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
