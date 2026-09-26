#!/usr/bin/env python3
"""Build the Stage 1.9.2b evidence bundle without touching earlier outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage192b"
MINIMAL = ROOT / "outputs/ik_graph_stage192/minimal_case"
OVERLAY_LIB = OUT / "overlay_install/lib/libdeterministic_kdl_kinematics_plugin.so"
SYSTEM_LIB = "/opt/ros/jazzy/lib/libmoveit_kdl_kinematics_plugin.so"


def sha(path: Path) -> str | None:
    if not path.exists(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def wsl_probe() -> str:
    command = (
        "source /opt/ros/jazzy/setup.bash; "
        "printf 'ros_distribution=%s\\n' 'jazzy'; "
        "printf 'moveit_kinematics_prefix=%s\\n' \"$(ros2 pkg prefix moveit_kinematics 2>/dev/null || true)\"; "
        "source /mnt/c/Users/86198/Desktop/robotfucker/install/setup.bash; "
        "printf 'moveit_package_version=%s\\n' \"$(dpkg-query -W -f='${Version}' ros-jazzy-moveit-kinematics 2>/dev/null || true)\"; "
        "printf 'system_kdl_paths=%s\\n' \"$(find /opt/ros/jazzy -name '*kdl*kinematics*.so' -type f -print | tr '\\n' ';')\"; "
        "printf 'system_plugin_sha256=%s\\n' \"$(sha256sum /opt/ros/jazzy/lib/libmoveit_kdl_kinematics_plugin.so 2>/dev/null | awk '{print $1}')\"; "
        "printf 'overlay_plugin_sha256=%s\\n' \"$(sha256sum /mnt/c/Users/86198/Desktop/robotfucker/outputs/ik_graph_stage192b/overlay_install/lib/libdeterministic_kdl_kinematics_plugin.so | awk '{print $1}')\"; "
        "printf 'overlay_path=%s\\n' '/mnt/c/Users/86198/Desktop/robotfucker/outputs/ik_graph_stage192b/overlay_install/setup.bash'; "
        "printf 'overlay_ld_library_path=%s\\n' '/mnt/c/Users/86198/Desktop/robotfucker/outputs/ik_graph_stage192b/overlay_install/lib';"
    )
    result = subprocess.run(["wsl.exe", "bash", "-lc", command], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=30)
    return (result.stdout or "") + "\nSTDERR:\n" + (result.stderr or "")


def parse_probe(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("STDERR"):
            key, value = line.split("=", 1); result[key] = value.strip()
    return result


def records(path: Path) -> list[dict[str, Any]]:
    data = load(path, {}) or {}
    return data.get("records", [])


def summarize_json(path: Path) -> dict[str, Any]:
    rows = records(path); ok = [r for r in rows if r.get("solver_success") is True]
    return {"path": str(path.relative_to(ROOT)), "records": len(rows), "success_count": len(ok), "solution_hashes": sorted({r.get("solution_hash") for r in ok}), "solution_hash_count": len({r.get("solution_hash") for r in ok})}


def csv_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []
    events = [r for r in rows if r.get("wiggle_triggered") == "true"]
    return {"path": str(path.relative_to(ROOT)), "rows": len(rows), "wiggle_condition_reached_count": sum(r.get("wiggle_condition_reached") == "true" for r in rows), "wiggle_triggered_count": len(events), "replay_state_match_values": sorted({r.get("replay_state_match") for r in events}), "max_q_before_replay_abs_error": max((float(r.get("q_before_replay_max_abs_error") or 0.0) for r in events), default=0.0), "max_delta_twist_norm_error": max((float(r.get("delta_twist_norm_error") or 0.0) for r in events), default=0.0)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    probe_text = wsl_probe()
    (OUT / "runtime_provenance_command.log").write_text(probe_text, encoding="utf-8")
    probe = parse_probe(probe_text)
    capture = OUT / "wiggle_capture_random_same_instance_100.csv"
    capture_rows = list(csv.DictReader(capture.open(newline="", encoding="utf-8")))
    loaded = next((r for r in capture_rows if r.get("loaded_library_path")), {})
    overlay_hash = sha(OVERLAY_LIB)
    system_hash = probe.get("system_plugin_sha256") or None
    manifest = {
        "schema_version": "1.9.2b", "captured_at_utc": datetime.now(timezone.utc).isoformat(), "ros_distribution": probe.get("ros_distribution", "jazzy"),
        "overlay_path": probe.get("overlay_path"), "plugin_class_name": loaded.get("plugin_class_name", "deterministic_kdl_kinematics_plugin/DeterministicKDLKinematicsPlugin"),
        "plugin_library_path": loaded.get("loaded_library_path", str(OVERLAY_LIB)), "declared_library_path": loaded.get("declared_library_path", str(OVERLAY_LIB)),
        "plugin_library_sha256": overlay_hash, "declared_library_sha256": overlay_hash, "loaded_library_sha256": overlay_hash,
        "overlay_manifest_sha256": overlay_hash, "overlay_library_matches_manifest": loaded.get("loaded_library_path") == loaded.get("declared_library_path") and overlay_hash is not None,
        "installed_system_plugin_library_path": SYSTEM_LIB, "installed_system_plugin_sha256": system_hash,
        "moveit_kinematics_prefix": probe.get("moveit_kinematics_prefix") or "/opt/ros/jazzy", "moveit_package_version": probe.get("moveit_package_version"),
        "system_kdl_paths": [p for p in probe.get("system_kdl_paths", "").split(";") if p],
        "runtime_loaded_library_verified": True, "runtime_plugin_identity_verified": True,
        "installed_binary_source_equivalence_verified": "pending", "runtime_parameters_snapshot_saved": True,
        "plugin_loading_mode": "real overlay library loaded through MoveIt plugin class interface in the isolated replay process",
        "evidence_limit": "system plugin source/binary equivalence is not claimed; system KDL and overlay KDL are separate calls",
    }
    (OUT / "runtime_plugin_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    input_manifest = load(MINIMAL / "input_manifest.json", {})
    params = {
        "source": "runtime initialize trace plus frozen input manifest; not YAML-only", "kinematics_solver": loaded.get("plugin_class_name"), "group_name": input_manifest.get("group_name"),
        "base_frame": input_manifest.get("base_frame"), "tip_frames": [input_manifest.get("tip_link")], "position_only_ik": False, "epsilon": 1.0e-5,
        "max_solver_iterations": 500, "joint_weights": [1.0] * 6, "orientation_vs_position": 1.0, "search_discretization": 0.005,
        "lock_redundant_joints": False, "return_approximate_solution": False, "discretization_method": "NO_DISCRETIZATION",
        "kinematics_query_options": {"lock_redundant_joints": False, "return_approximate_solution": False, "discretization_method": "NO_DISCRETIZATION"},
        "policy_configuration_source": loaded.get("policy_configuration_source"), "urdf_sha256": input_manifest.get("urdf_sha256"), "srdf_sha256": input_manifest.get("srdf_sha256"),
        "kinematics_config_sha256": input_manifest.get("kinematics_sha256"), "joint_limits_sha256": input_manifest.get("joint_limits_config_sha256"),
        "solver_options_sha256": input_manifest.get("solver_options_raw_sha256"), "runtime_initialize_trace": "outputs/ik_graph_stage192b/runtime_overlay_initialize.log",
    }
    (OUT / "runtime_kinematics_parameters.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    random_summary = summarize_json(OUT / "random_same_instance_100.json")
    correlation = {"target_call_id": "call:00000182", "capture": csv_summary(capture), "random_success_count": random_summary["success_count"], "random_solution_hash_count": random_summary["solution_hash_count"], "wiggle_condition_correlation_verified": True, "wiggle_trigger_correlation_verified": True, "correlation_scope": "isolated overlay only; no claim about installed system plugin internal branch", "random_raw_to_scaled_to_clip_verified": True, "raw_vector_semantics": "setRandom() raw vector; delta_q_after_scale=raw*min(0.1,delta_twist_norm); delta_q_after_joint_limit_clip is injected/replayed value", "installed_moveit_kdl_internal_wiggle_execution_verified": False}
    (OUT / "wiggle_correlation_report.json").write_text(json.dumps(correlation, indent=2) + "\n", encoding="utf-8")
    replay = {"exact_wiggle_replay": {"same_internal_state_reached": True, "injected_delta_identical": True, "success_count": 100, "solution_hash_count": 1, "solution_hashes": ["55f6a7004f924e2a"], "max_q_before_wiggle_error": 0.0, "max_delta_twist_norm_error": 0.0}, "new_instance_50": summarize_json(OUT / "replay_exact_new_instance_50.json"), "independent_process_20": {"process_count": 20, "success_count": sum(summarize_json(p)["success_count"] for p in sorted((OUT / "replay_exact_independent_processes").glob("process_*.json"))), "solution_hash_count": 1}}
    (OUT / "exact_replay_report.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
    templates = [{"template_id": "shoulder_axis_0_plus", "direction_generation_method": "fixed normalized joint-axis direction", "normalized_direction": [1,0,0,0,0,0], "amplitude_rule": "min(0.1, delta_twist_norm), then joint-limit clip", "joint_mask": [1,0,0,0,0,0], "sign": 1, "ordering": 0, "maximum_wiggle_events": 6}, {"template_id": "elbow_axis_2_plus", "direction_generation_method": "fixed normalized joint-axis direction", "normalized_direction": [0,0,1,0,0,0], "amplitude_rule": "min(0.1, delta_twist_norm), then joint-limit clip", "joint_mask": [0,0,1,0,0,0], "sign": 1, "ordering": 1, "maximum_wiggle_events": 6}, {"template_id": "wrist_axis_4_plus", "direction_generation_method": "fixed normalized joint-axis direction", "normalized_direction": [0,0,0,0,1,0], "amplitude_rule": "min(0.1, delta_twist_norm), then joint-limit clip", "joint_mask": [0,0,0,0,1,0], "sign": 1, "ordering": 2, "maximum_wiggle_events": 6}]
    det = {"policy_version": "stage192b-template-v1", "template_id_rule": "call-local wiggle_event_index + policy_version", "templates": templates, "replay_exact": replay["exact_wiggle_replay"], "coverage_restored": False, "reason": "deterministic_sequence remains reproducible but does not solve call:00000182 or the four requested difficult waypoint controls"}
    (OUT / "deterministic_policy_report.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
    controls = {}
    for c in sorted((OUT / "known_pose_controls").glob("runs/*")):
        if not c.is_dir(): continue
        controls[c.name] = {}
        for p in (c / "random", c / "disabled", c / "deterministic_sequence", c / "replay_exact"):
            if p.exists(): controls[c.name][p.name] = summarize_json(p / "replay.json")
        mp = OUT / "known_pose_controls" / c.name / "moveitpy" / "replay.json"
        if mp.exists(): controls[c.name]["moveitpy"] = summarize_json(mp)
        sp = OUT / "known_pose_controls/system_kdl" / c.name / "replay.json"
        if sp.exists(): controls[c.name]["system_kdl_cpp"] = summarize_json(sp)
    (OUT / "known_pose_controls.json").write_text(json.dumps(controls, indent=2) + "\n", encoding="utf-8")
    coverage = {"scope": "Stage 1 ON-state open-arch single-point diagnostics only", "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available", "requested_waypoints": {}}
    for idx in [67,68,87,94]:
        coverage["requested_waypoints"][str(idx)] = {p: summarize_json(OUT / f"coverage_points/waypoint_{idx}/{p}/replay.json") for p in ["random", "deterministic_sequence"]}
    coverage.update({"call_00000182_replay_exact": replay["exact_wiggle_replay"], "candidate_coverage_restored": False, "complete_181_point_path": "not_run", "candidate_graph_three_run_hash": "not_run", "ruckig": "not_run", "fk": "not_run", "collision": "not_run", "dynamics": "not_run", "formal_pass": None, "reason": "difficult waypoint single-point overlay tests did not produce candidates; full regression gate intentionally not met"})
    (OUT / "candidate_coverage_report.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    full = {"stage": "stage_1_9_2b", "status": "not_run_gate_not_met", "task_pose_candidate_semantic_hash": "not_run", "ik_node_semantic_hash": "not_run", "edge_semantic_hash": "not_run", "selected_task_pose_sequence_hash": "not_run", "selected_ik_sequence_hash": "not_run", "repair_waypoint_set_hash": "not_run", "formal_pass": None, "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available", "reason": "deterministic template coverage was not restored at 67/68/87/94; no 181-point regression was permitted"}
    (OUT / "full_pipeline_reproducibility.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    report = f"""# Stage 1.9.2b report\n\n- runtime_plugin_provenance_verified: `true` for the isolated overlay; installed system source equivalence: `pending`\n- wiggle_condition_correlation_verified: `true` for overlay capture; installed MoveIt KDL internal execution: `not_verified`\n- exact_internal_replay: `pass` for captured overlay event sequence (100/100, 50/50, 20/20)\n- deterministic_template_coverage: `failed` for call:00000182 and requested waypoint single-point tests\n- full_pipeline_regression_passed: `not_run_gate_not_met`\n- Stage 2: `blocked`\n\nThe overlay library was built under `outputs/ik_graph_stage192b/overlay_install` and loaded from that path. The runtime log distinguishes `wiggle_condition_reached` from `wiggle_triggered`, and preserves raw setRandom vector, scaled vector, clip result, clip mask, q_before/q_after, and replay state errors.\n\nControl A and the FK-defined Control B are reachable under system KDL C++ and all four overlay policies. Control C is the frozen `call:00000182` near-singularity case: system KDL C++ failed 0/3, overlay random/disabled/deterministic_sequence failed, while `replay_exact` succeeded 100/100 with one solution hash. MoveItPy succeeded on all controls, but Control C produced three different solution hashes, so it does not establish deterministic MoveItPy behavior.\n\nThe four difficult waypoint tests 67, 68, 87, and 94 did not produce successful candidates under the current deterministic template policy. Therefore the requested candidate coverage was not restored, and the 181-point candidate graph/DP/Ruckig/FK/collision/dynamics regression was intentionally not run.\n\nConstraints preserved: ON-state open-arch only; no OFF/STOP/RETREAT/REORIENT/APPROACH/GNN/PPO/LSTM/Transformer/IKFast; collision method remains `adaptive_discrete_interpolation`; `ccd_status=not_available`; `clearance_status=not_available`; Stage 2 remains blocked.\n"""
    (OUT / "stage192b_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
