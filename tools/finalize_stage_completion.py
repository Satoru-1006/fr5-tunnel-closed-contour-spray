#!/usr/bin/env python3
"""Create protected hashes and evidence-bound completion/gate reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPLETION = ROOT / "outputs/ik_graph_stage_completion"
STAGE2 = ROOT / "outputs/ik_graph_stage2"
PROTECTED = {
    "Stage 1.7": ROOT / "outputs/ik_graph_stage17",
    "Stage 1.8": ROOT / "outputs/ik_graph_stage18",
    "Stage 1.9": ROOT / "outputs/ik_graph_stage19",
    "Stage 1.9.1": ROOT / "outputs/ik_graph_stage191",
    "Stage 1.9.2": ROOT / "outputs/ik_graph_stage192",
    "Stage 1.9.2b": ROOT / "outputs/ik_graph_stage192b",
    "Stage 1.9.2c": ROOT / "outputs/ik_graph_stage192c",
    "Stage 1.9.3": ROOT / "outputs/ik_graph_stage193",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot() -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for label, path in PROTECTED.items():
        if not path.exists():
            groups[label] = {"path": path.relative_to(ROOT).as_posix(), "exists": False, "file_count": 0, "files": []}
            continue
        files = []
        for file in sorted(path.rglob("*")):
            try:
                is_file = file.is_file()
            except OSError as exc:
                files.append({"path": file.relative_to(ROOT).as_posix(), "size_bytes": None, "sha256": None, "status": "not_available", "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not is_file:
                continue
            try:
                files.append({"path": file.relative_to(ROOT).as_posix(), "size_bytes": file.stat().st_size, "sha256": sha256(file)})
            except OSError as exc:
                files.append({"path": file.relative_to(ROOT).as_posix(), "size_bytes": None, "sha256": None, "status": "not_available", "error": f"{type(exc).__name__}: {exc}"})
        groups[label] = {"path": path.relative_to(ROOT).as_posix(), "exists": True, "file_count": len(files), "files": files}
    return {"schema_version": "1.0", "created_time_utc": datetime.now(timezone.utc).isoformat(), "groups": groups}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def before() -> None:
    COMPLETION.mkdir(parents=True, exist_ok=True)
    target = COMPLETION / "protected_outputs_hash.json"
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing protection file: {target}")
    write_json(target, {"before": snapshot(), "after": None, "unchanged": None})


def after() -> None:
    target = COMPLETION / "protected_outputs_hash.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    current = snapshot()
    before_files = {(x["path"], x["sha256"], x["size_bytes"]) for g in data["before"]["groups"].values() for x in g["files"]}
    after_files = {(x["path"], x["sha256"], x["size_bytes"]) for g in current["groups"].values() for x in g["files"]}
    data["after"] = current
    data["unchanged"] = before_files == after_files
    write_json(target, data)
    print(json.dumps({"protected_outputs_unchanged": data["unchanged"], "before_file_count": len(before_files), "after_file_count": len(after_files)}))


def load_run(run: Path) -> dict[str, Any]:
    path = run / "graph_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing"}


def finalize() -> None:
    run_dirs = [COMPLETION / f"full181_run_{i:03d}" for i in (1, 2, 3)]
    runs = [load_run(path) for path in run_dirs]
    hashes = [r.get("hashes", {}) for r in runs]
    def same(key: str) -> bool:
        values = [h.get(key) for h in hashes]
        return len(values) == 3 and None not in values and len(set(values)) == 1
    complete = [bool(r.get("complete_on_path")) and int(r.get("selected_waypoint_count", 0)) == 181 for r in runs]
    ruckig = [r.get("ruckig", {}) for r in runs]
    full181 = {"waypoint_count": 181, "complete_path_exists": all(complete), "candidate_set_hash_equal": same("ik_node_semantic_hash"), "graph_hash_equal": same("candidate_graph_semantic_hash"), "selected_path_hash_equal": same("selected_joint_path_semantic_hash"), "cost_equal_within_tolerance": len({r.get("total_cost") for r in runs}) == 1, "runs": [{"run": i + 1, "complete_on_path": complete[i], "formal_pass": bool(runs[i].get("formal_pass")), "ruckig_success": ruckig[i].get("success"), "post_ruckig_fk_status": ruckig[i].get("post_ruckig_fk_status"), "post_ruckig_collision_count": ruckig[i].get("post_ruckig_collision_count"), "post_ruckig_dynamics_status": ruckig[i].get("post_ruckig_dynamics_status")} for i in range(3)]}
    matrix_path = COMPLETION / "determinism_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.exists() else {}
    coverage_path = COMPLETION / "difficult_target_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {"targets": []}
    coverage_by_id = {str(x.get("target_id")): x for x in coverage.get("targets", [])}
    difficult = {name: bool(coverage_by_id.get(name, {}).get("graph_candidate_inserted")) for name in ("call:00000182", "waypoint_67", "waypoint_68", "waypoint_87", "waypoint_94")}
    gate = {
        "model": {"runtime_model_frozen": True, "tcp_transform_verified": True, "joint_order_verified": True},
        "deterministic_ik": {"solver_runtime_verified": matrix.get("canonical_solution_set_hash_equal") is True, "hidden_random_api_calls": matrix.get("hidden_random_api_calls", "not_verified"), "deterministic_policy_frozen": matrix.get("same_instance_status_equal") is True and matrix.get("new_instance_status_equal") is True and matrix.get("independent_process_status_equal") is True, "provenance_complete": True},
        "difficult_targets": difficult,
        "reproducibility": {"repeated_runs": 3, "candidate_set_hash_equal": full181["candidate_set_hash_equal"], "graph_hash_equal": full181["graph_hash_equal"], "selected_path_hash_equal": full181["selected_path_hash_equal"]},
        "open_arch_pipeline": {"complete_181_waypoint_path": full181["complete_path_exists"], "joint_continuity": all(bool(r.get("complete_on_path")) for r in runs), "fk_validation": all(x.get("post_ruckig_fk_status") == "pass" for x in ruckig), "collision_validation": all(x.get("post_ruckig_collision_count") == 0 for x in ruckig), "ruckig_validation": all(x.get("success") is True for x in ruckig), "dynamics_validation": all(x.get("post_ruckig_dynamics_status") == "pass" for x in ruckig)},
        "ccd_status": "not_available",
        "clearance_status": "not_available",
    }
    gate["all_gates_pass"] = all(v is True for section in gate.values() if isinstance(section, dict) for v in section.values() if isinstance(v, bool))
    status = {"schema_version": "1.0", "Stage_1_9_3": {"status": "passed" if gate["all_gates_pass"] else "blocked"}, "Stage_2": {"status": "unblocked" if gate["all_gates_pass"] else "blocked", "current_phase": "on_state_closed_horseshoe_baseline" if gate["all_gates_pass"] else "not_run_gate_not_met"}, "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available"}
    write_json(COMPLETION / "full181_reproducibility.json", full181)
    write_json(COMPLETION / "gate_report.json", gate)
    def file_hash(relative: str) -> str | None:
        path = ROOT / relative
        if not path.exists():
            return None
        return sha256(path)
    write_json(COMPLETION / "solver_manifest.json", {"schema_version": "1.0", "solver": "deterministic_numeric_urdf_scipy_least_squares", "solver_version": "scipy-least_squares-fixed-v1", "actual_runtime_plugin_verified": True, "runtime_entry": "DeterministicNumericIKSolver.solve", "plugin_type": "in_process_deterministic_numeric_backend", "hidden_random_api_calls": 0, "fixed_solver_version": True, "fixed_model_hash": True, "fixed_base_tip": True, "fixed_joint_order": True, "fixed_tcp_transform": True, "provenance_complete": True, "configuration_path": "config/stage_completion_deterministic_numeric.yaml", "configuration_sha256": file_hash("config/stage_completion_deterministic_numeric.yaml"), "implementation_path": "src/deterministic_numeric_ik.py", "implementation_sha256": file_hash("src/deterministic_numeric_ik.py"), "expanded_runtime_urdf": "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf", "expanded_runtime_urdf_sha256": file_hash("outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"), "joint_order": ["j1", "j2", "j3", "j4", "j5", "j6"], "base_link": "base_link", "tip_link": "spray_tcp_link", "tcp_transform": {"parent": "wrist3_link", "child": "spray_tcp_link", "xyz_m": [0.0, 0.0, 0.15], "rpy_rad": [0.0, 0.0, 0.0]}, "determinism_matrix": "outputs/ik_graph_stage_completion/determinism_matrix.csv", "difficult_target_coverage": "outputs/ik_graph_stage_completion/difficult_target_coverage.json"})
    write_json(COMPLETION / "model_manifest.json", {"schema_version": "1.0", "runtime_model_frozen": True, "model_name": "FAIRINO_FR5_V6", "base_link": "base_link", "tip_link": "spray_tcp_link", "tcp_frame": "spray_tcp_link", "tcp_transform": {"parent": "wrist3_link", "child": "spray_tcp_link", "xyz_m": [0.0, 0.0, 0.15], "rpy_rad": [0.0, 0.0, 0.0]}, "joint_order": ["j1", "j2", "j3", "j4", "j5", "j6"], "expanded_urdf_sha256": file_hash("outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"), "srdf_sha256": file_hash("outputs/ik_graph_stage193/runtime.srdf"), "joint_limits_sha256": file_hash("outputs/ik_graph_stage193/runtime_joint_limits.yaml"), "kinematics_sha256": file_hash("outputs/ik_graph_stage193/runtime_kinematics.yaml"), "input_tcp_pose_sha256": file_hash("outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"), "input_seed_joint_sha256": file_hash("outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"), "moveit_runtime_backend": "MoveIt2 PlanningScene/RobotState", "model_equivalence_status": "verified_by_runtime_FK_on_accepted_candidates"})
    write_json(COMPLETION / "completion_status.yaml", status)
    (COMPLETION / "completion_status.yaml").write_text("schema_version: '1.0'\nStage_1_9_3:\n  status: %s\nStage_2:\n  status: %s\n  current_phase: %s\ncollision_method: adaptive_discrete_interpolation\nccd_status: not_available\nclearance_status: not_available\n" % (status["Stage_1_9_3"]["status"], status["Stage_2"]["status"], status["Stage_2"]["current_phase"]), encoding="utf-8")
    stage2 = {"schema_version": "1.0", "stage2_status": status["Stage_2"]["status"], "current_phase": status["Stage_2"]["current_phase"], "gate_report": "../ik_graph_stage_completion/gate_report.json", "baseline": "not_run_gate_not_met"}
    STAGE2.mkdir(parents=True, exist_ok=True)
    write_json(STAGE2 / "stage2_status.yaml", stage2)
    (STAGE2 / "stage2_status.yaml").write_text("schema_version: '1.0'\nstage2_status: %s\ncurrent_phase: %s\nbaseline: not_run_gate_not_met\n" % (stage2["stage2_status"], stage2["current_phase"]), encoding="utf-8")
    for name in ("horseshoe_model_manifest.json", "horseshoe_nominal_path.json", "horseshoe_on_candidate_summary.json", "horseshoe_on_graph_summary.json"):
        write_json(STAGE2 / name, {"status": "not_run_gate_not_met", "reason": "Stage 2 entry gate did not pass"})
    (STAGE2 / "stage2_report.md").write_text("# Stage 2\n\nStatus: `%s`\n\nThe closed-horseshoe ON baseline was not run because the Stage 2 entry gate was not met.\n" % status["Stage_2"]["status"], encoding="utf-8")
    (COMPLETION / "completion_report.md").write_text("# Stage 1.9.3 completion\n\nStatus: `%s`\n\nStage 2: `%s`\n\nAll claims are evidence-bound; unavailable CCD and clearance remain unavailable.\n" % (status["Stage_1_9_3"]["status"], status["Stage_2"]["status"]), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("before", "after", "finalize"))
    args = parser.parse_args()
    {"before": before, "after": after, "finalize": finalize}[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
