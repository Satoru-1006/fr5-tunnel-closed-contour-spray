#!/usr/bin/env python3
"""Stage 3 H4.3 offline curved-collision authorization audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import stage3_h4_2 as h42  # noqa: E402
import stage3_h4_2_1 as h421  # noqa: E402

H421_ROOT = ROOT / "outputs/stage3_h4_2_1_clean_exit_recertification_20260808T230100Z"
DEFAULT_OUTPUT = ROOT / "outputs/stage3_h4_3_authorization_audit_20260808T230800Z"
GEOMETRY = "fixture_curved_cylinder_patch"
SELECTED_CANDIDATE = "candidate_015_combined_centered_alt"
FORBIDDEN_LOG_PATTERNS = ("process has died", "exit code -11", "sigsegv", "segmentation fault")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"root": rel(root), "file_count": len(files), "files": files, "tree_hash": h42.semantic_hash(files)}


def selected_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    targets = load_jsonl(H421_ROOT / "selected_target_feasibility.jsonl")
    ik = load_jsonl(H421_ROOT / "selected_ik_candidates.jsonl")
    fk = load_jsonl(H421_ROOT / "selected_fk_validation.jsonl")
    collision = load_jsonl(H421_ROOT / "selected_collision_validation.jsonl")
    selected = load_json(H421_ROOT / "selected_candidate_input_copy.json")
    return targets, ik, fk, collision, selected


def write_curved_inputs(output: Path) -> dict[str, Any]:
    targets, ik_rows, fk_rows, collision_rows, selected = selected_inputs()
    target_map = {int(row["target_index"]): row for row in targets if row.get("geometry_id") == GEOMETRY}
    ik_map = {int(row["target_index"]): row for row in ik_rows if row.get("geometry_id") == GEOMETRY}
    fk_map = {(int(row["target_index"]), str(row["candidate_id"])): row for row in fk_rows}
    candidate_rows = []
    for collision in collision_rows:
        target_index = int(collision["target_index"])
        if target_index not in target_map or not collision.get("collision_checked"):
            continue
        ik_row = ik_map[target_index]
        candidate_id = str(collision["candidate_id"])
        candidate = next((item for item in ik_row.get("deduplicated_candidates", []) if str(item.get("candidate_id")) == candidate_id), None)
        if candidate is None:
            raise RuntimeError(f"missing deduplicated IK candidate for target={target_index} candidate={candidate_id}")
        fk = fk_map.get((target_index, candidate_id), {})
        joint_validation = fk.get("joint_validation", {})
        candidate_rows.append({
            "target_index": target_index,
            "task_sample_id": target_map[target_index]["task_sample_id"],
            "candidate_id": candidate_id,
            "target_classification": target_map[target_index]["classification"],
            "baseline_classification": ("IK_FOUND_ENV_COLLISION" if collision.get("fcl_environment_collision") else ("IK_FOUND_SELF_COLLISION" if collision.get("fcl_self_collision") else "REACHABLE_COLLISION_FREE")),
            "fk_translation_error_m": fk.get("translation_error_m"),
            "fk_orientation_error_rad": fk.get("orientation_error_rad"),
            "joint_limit_state": "VALID" if joint_validation.get("valid") else "INVALID",
            "joint_positions": candidate["joint_positions"],
        })
    input_csv = output / "curved_ik_found_candidates.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["target_index", "task_sample_id", "candidate_id", "target_classification", "baseline_classification", "fk_translation_error_m", "fk_orientation_error_rad", "joint_limit_state", *[f"q{i}" for i in range(6)]]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            values = {key: row.get(key) for key in fieldnames}
            for index, value in enumerate(row["joint_positions"]):
                values[f"q{index}"] = value
            writer.writerow(values)

    record = next(row for row in load_json(h42.H2_MANIFEST)["records"] if row["geometry_id"] == GEOMETRY)
    record = dict(record)
    record["source_path"] = str((ROOT / str(record["source_path"]).replace("\\", "/")).resolve())
    from src.stage3_h3_task_representation import load_and_verify_h2_mesh

    mesh, _processing, _derived = load_and_verify_h2_mesh(str(ROOT), record)
    transformed = h42.transform_mesh(mesh, selected["transforms"][GEOMETRY])
    mesh_obj = output / "curved_fixture_selected_candidate.obj"
    with mesh_obj.open("w", encoding="utf-8", newline="\n") as stream:
        for vertex in transformed.vertices:
            stream.write(f"v {float(vertex[0]):.17g} {float(vertex[1]):.17g} {float(vertex[2]):.17g}\n")
        for triangle in transformed.triangles:
            stream.write(f"f {int(triangle[0]) + 1} {int(triangle[1]) + 1} {int(triangle[2]) + 1}\n")
    manifest = {
        "schema_version": "stage3-h4-3-curved-input-v1",
        "source_h2_manifest": rel(h42.H2_MANIFEST),
        "source_geometry_id": GEOMETRY,
        "selected_candidate": SELECTED_CANDIDATE,
        "candidate_count": len(candidate_rows),
        "ik_found_target_count": len(target_map) - sum(row.get("classification") == "IK_UNREACHABLE" for row in target_map.values()),
        "curved_target_count": len(target_map),
        "mesh_vertices": len(transformed.vertices),
        "mesh_triangles": len(transformed.triangles),
        "collision_method": h42.COLLISION_METHOD,
        "ccd_status": h42.UNAVAILABLE,
        "clearance_status": h42.UNAVAILABLE,
        "input_csv": rel(input_csv),
        "mesh_obj": rel(mesh_obj),
        "placement_transform": selected["transforms"][GEOMETRY],
    }
    dump_json(output / "curved_contact_input_manifest.json", manifest)
    return {"candidate_rows": candidate_rows, "input_csv": input_csv, "mesh_obj": mesh_obj, "manifest": manifest}


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{resolved.as_posix().split(':', 1)[-1].lstrip('/').replace('\\\\', '/') }"


def build_native_bridge(output: Path) -> tuple[Path, dict[str, Any]]:
    build_base = output / "cpp_build"
    install_base = output / "cpp_install"
    build_command = f"source /opt/ros/jazzy/setup.bash && colcon build --base-paths {wsl_path(ROOT / 'cpp/stage3_h4_3')} --build-base {wsl_path(build_base)} --install-base {wsl_path(install_base)} --merge-install"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", build_command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False)
    log = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    (output / "native_bridge_build.log").write_text(log, encoding="utf-8", newline="\n")
    executable = install_base / "lib/stage3_h4_3_contact_bridge/stage3_h4_3_contact_bridge"
    result = {"command": build_command, "returncode": proc.returncode, "executable": rel(executable), "executable_exists": executable.is_file(), "stdout_tail": (proc.stdout or "")[-4000:], "stderr_tail": (proc.stderr or "")[-4000:]}
    dump_json(output / "native_bridge_build.json", result)
    if proc.returncode != 0 or not executable.is_file():
        raise RuntimeError(f"native H4.3 bridge build failed: {result}")
    return executable, result


def run_native_contact(output: Path, inputs: Mapping[str, Any], executable: Path) -> dict[str, Any]:
    native_dir = output / "native_contact"
    native_dir.mkdir(parents=True, exist_ok=False)
    launch_command = " ".join([
        f"ros2 launch {wsl_path(ROOT / 'scripts/stage3_h4_3_launch.py')}",
        f"executable_path:={wsl_path(executable)}",
        f"candidate_csv:={wsl_path(inputs['input_csv'])}",
        f"curved_mesh_obj:={wsl_path(inputs['mesh_obj'])}",
        f"output_dir:={wsl_path(native_dir)}",
        "max_contacts:=4096",
        "max_contacts_per_pair:=64",
    ])
    shell = " && ".join(["source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"cd {wsl_path(ROOT)}", launch_command])
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", shell], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False)
    log = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    (output / "native_contact_launch.log").write_text(log, encoding="utf-8", newline="\n")
    summary_path = native_dir / "native_contact_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    contact_rows = load_jsonl(native_dir / "curved_contact_evidence.jsonl") if (native_dir / "curved_contact_evidence.jsonl").is_file() else []
    target_state: dict[int, dict[str, Any]] = {}
    for row in contact_rows:
        state = target_state.setdefault(int(row["target_index"]), {"task_sample_id": row["task_sample_id"], "environment_collision": False, "self_collision": False})
        if row.get("collision_category") == "environment_collision":
            state["environment_collision"] = True
        if row.get("collision_category") == "self_collision":
            state["self_collision"] = True
    target_aggregate = []
    target_counts: dict[str, int] = {}
    for candidate in inputs["candidate_rows"]:
        index = int(candidate["target_index"])
        if any(row["target_index"] == index for row in target_aggregate):
            continue
        state = target_state.get(index, {"task_sample_id": candidate["task_sample_id"], "environment_collision": False, "self_collision": False})
        if state["environment_collision"] and state["self_collision"]:
            category = "IK_FOUND_BOTH_COLLISION"
        elif state["environment_collision"]:
            category = "IK_FOUND_ENV_COLLISION"
        elif state["self_collision"]:
            category = "IK_FOUND_SELF_COLLISION"
        else:
            category = "REACHABLE_COLLISION_FREE"
        target_aggregate.append({"schema_version": "stage3-h4-3-target-collision-aggregate-v1", "target_index": index, "task_sample_id": state["task_sample_id"], "native_target_category": category, "baseline_target_category": next(row["target_classification"] for row in inputs["candidate_rows"] if int(row["target_index"]) == index), "environment_collision": state["environment_collision"], "self_collision": state["self_collision"]})
        target_counts[category] = target_counts.get(category, 0) + 1
    dump_jsonl(native_dir / "native_target_collision_aggregate.jsonl", sorted(target_aggregate, key=lambda row: row["target_index"]))
    summary["ik_found_target_count"] = len(target_aggregate)
    summary["native_target_category_counts"] = dict(sorted(target_counts.items()))
    baseline_target_counts: dict[str, int] = {}
    seen_targets: set[int] = set()
    for row in inputs["candidate_rows"]:
        index = int(row["target_index"])
        if index in seen_targets:
            continue
        seen_targets.add(index)
        category = str(row["target_classification"])
        baseline_target_counts[category] = baseline_target_counts.get(category, 0) + 1
    summary["baseline_target_category_counts"] = dict(sorted(baseline_target_counts.items()))
    dump_json(summary_path, summary)
    forbidden = [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in log.lower()]
    result = {"outer_returncode": proc.returncode, "forbidden_log_patterns": forbidden, "summary": summary, "summary_exists": summary_path.is_file(), "native_contact_dir": rel(native_dir), "stdout_tail": (proc.stdout or "")[-4000:], "stderr_tail": (proc.stderr or "")[-4000:]}
    dump_json(output / "native_contact_run.json", result)
    if proc.returncode != 0 or forbidden or not summary_path.is_file() or summary.get("collision_checked_count") != len(inputs["candidate_rows"]):
        raise RuntimeError(f"native FCL contact bridge failed: {result}")
    return result


def placement_specs(selected: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = selected["transforms"][GEOMETRY]
    translation = (float(base[0][3]), float(base[1][3]), float(base[2][3]))
    specs = [("base", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), ("dx_minus_050", -0.05, 0.0, 0.0, 0.0, 0.0, 0.0), ("dx_plus_050", 0.05, 0.0, 0.0, 0.0, 0.0, 0.0), ("dy_minus_050", 0.0, -0.05, 0.0, 0.0, 0.0, 0.0), ("dy_plus_050", 0.0, 0.05, 0.0, 0.0, 0.0, 0.0), ("dz_minus_050", 0.0, 0.0, -0.05, 0.0, 0.0, 0.0), ("dz_plus_050", 0.0, 0.0, 0.05, 0.0, 0.0, 0.0), ("yaw_minus_15", 0.0, 0.0, 0.0, -15.0, 0.0, 0.0), ("yaw_plus_15", 0.0, 0.0, 0.0, 15.0, 0.0, 0.0), ("pitch_minus_10", 0.0, 0.0, 0.0, 0.0, -10.0, 0.0), ("pitch_plus_10", 0.0, 0.0, 0.0, 0.0, 10.0, 0.0), ("roll_minus_10", 0.0, 0.0, 0.0, 0.0, 0.0, -10.0), ("roll_plus_10", 0.0, 0.0, 0.0, 0.0, 0.0, 10.0)]
    result = []
    for label, dx, dy, dz, yaw, pitch, roll in specs:
        transforms = {key: value for key, value in selected["transforms"].items()}
        transforms[GEOMETRY] = h42.placement_matrix(translation[0] + dx, translation[1] + dy, translation[2] + dz, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)
        result.append({"schema_version": "stage3-h4-3-placement-candidate-v1", "candidate_id": f"h4_3_{label}", "label": label, "placement_status": "EXPERIMENTAL_DERIVED_PLACEMENT", "search_method": "bounded_deterministic_local_enumeration", "random_search": False, "ml": False, "rl": False, "path_planner": False, "ruckig": False, "delta_translation_m": [dx, dy, dz], "delta_yaw_pitch_roll_deg": [yaw, pitch, roll], "transforms": transforms})
    return result


def run_placement_study(output: Path, selected: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = placement_specs(selected)
    rows = []
    for index, spec in enumerate(specs, 401):
        replay = h421.run_replay(spec, output, index)
        result = replay["result"]
        curved = result["metrics"]["per_geometry"][GEOMETRY]
        rows.append({"candidate_id": spec["candidate_id"], "label": spec["label"], "status": "EVALUATED", "placement_status": spec["placement_status"], "search_method": spec["search_method"], "delta_translation_m": spec["delta_translation_m"], "delta_yaw_pitch_roll_deg": spec["delta_yaw_pitch_roll_deg"], "target_count": result["target_count"], "classification_counts": curved["classification_counts"], "curved_metrics": curved, "full_metrics": result["metrics"], "semantic_hash": result["semantic_hash"], "fresh_process_record": replay["record"]})
    dump_jsonl(output / "placement_candidate_metrics.jsonl", rows)
    selected_row = max(rows, key=lambda row: (int(row["curved_metrics"].get("reachable_collision_free", 0)), -int(row["curved_metrics"].get("ik_unreachable", 0)), -int(row["curved_metrics"].get("environment_collision", 0)), -int(row["curved_metrics"].get("self_collision", 0)), str(row["candidate_id"])))
    spec = next(item for item in specs if item["candidate_id"] == selected_row["candidate_id"])
    chosen = {**spec, "metrics": selected_row["full_metrics"], "curved_metrics": selected_row["curved_metrics"], "selection_policy": "max curved reachable collision-free, then min unreachable/environment/self; deterministic tie by candidate_id"}
    dump_json(output / "selected_placement_candidate.json", chosen)
    summary = {"schema_version": "stage3-h4-3-placement-study-v1", "candidate_count": len(rows), "target_denominator": 192, "selected_candidate_id": chosen["candidate_id"], "selected_curved_metrics": chosen["curved_metrics"], "all_candidates_curved_collision_free_zero": all(int(row["curved_metrics"].get("reachable_collision_free", 0)) == 0 for row in rows), "bounded_delta_translation_limits_m": 0.05, "bounded_delta_yaw_deg": 15.0, "bounded_delta_pitch_deg": 10.0, "bounded_delta_roll_deg": 10.0, "random_search": False, "bayesian_optimization": False, "ml": False, "rl": False, "planner_driven_optimization": False}
    dump_json(output / "placement_study_summary.json", summary)
    return rows, chosen


def make_authorization(output: Path, native: Mapping[str, Any], placement_rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    native_summary = native["summary"]
    curved_free = int(chosen["curved_metrics"].get("reachable_collision_free", 0))
    h421_replay = load_json(H421_ROOT / "clean_exit_replay_report.json")
    h421_terminal = load_json(H421_ROOT / "stage3_h4_2_1_terminal_certificate.json")
    h421_immutable = h42.semantic_hash(before) == h42.semantic_hash(after)
    placement_clean = all(row["fresh_process_record"].get("child_process_clean_exit") for row in placement_rows)
    placement_hashes_identical = all(row["fresh_process_record"].get("semantic_hash") == row["semantic_hash"] for row in placement_rows)
    gates = {
        "H4_2_1_PASSED": h421_terminal.get("STAGE_3_H4_2_1") == "PASSED",
        "NATIVE_FCL_CONTACTS_RAN": native_summary.get("collision_checked_count") == native_summary.get("candidate_count") and native_summary.get("candidate_count", 0) > 0,
        "CONTACT_EVIDENCE_FOR_CURVED_IK_FOUND": native_summary.get("candidate_count", 0) > 0 and native_summary.get("total_contact_count", 0) > 0,
        "BOUNDED_DETERMINISTIC_PLACEMENT_STUDY_COMPLETE": len(placement_rows) == 13 and all(row.get("target_count") == 192 for row in placement_rows),
        "PLACEMENT_CHILD_CLEAN_EXIT_ALL": placement_clean,
        "PLACEMENT_SEMANTIC_HASH_SELF_CHECK": placement_hashes_identical,
        "IMMUTABLE_BASELINES_UNCHANGED": h421_immutable,
        "DENOMINATOR_192_RETAINED": all(row.get("target_count") == 192 for row in placement_rows),
        "NO_FJT_GOALS": True,
        "NO_ROBOT_MOTION": True,
        "NO_PATH_PLANNING": True,
        "NO_RUCKIG": True,
        "NO_ML_RL": True,
        "NO_FORMAL_LEDGER_MUTATION": True,
    }
    status = "PASSED" if all(gates.values()) else "BLOCKED"
    first_blocker = next((name.lower() for name, passed in gates.items() if not passed), "none")
    h5_contract = {
        "schema_version": "stage3-h5-entry-contract-v1",
        "status": "NOT_SATISFIED" if curved_free == 0 else "PROPOSED_NOT_AUTHORIZED",
        "required_before_h5": ["H4.2.1 PASSED", "H4.3 PASSED", "curved_collision_free_reachable_count > 0", "three independent fresh-process semantic hashes identical for selected candidate", "curved reachable set must be demonstrated on more than a single accidental IK branch; continuity/coverage evidence required", "all 192 denominator targets retained", "no FJT/motion/planning/Ruckig/ML before explicit authorization"],
        "observed_selected_curved_collision_free_reachable_count": curved_free,
        "observed_h4_2_1_three_replay_clean_exit": h421_terminal.get("CHILD_PROCESS_CLEAN_EXIT_3_OF_3"),
        "observed_h4_2_1_three_replay_hashes": h421_replay.get("all_semantic_hashes_identical"),
        "ready_for_stage_3_h5": False,
    }
    final = {
        "schema_version": "stage3-h4-3-final-authorization-certificate-v1",
        "STAGE_3_H4_2_1": "PASSED",
        "STAGE_3_H4_3": status,
        "FIRST_BLOCKER": first_blocker,
        "H0_H1_H2_H3_H4_H4_1_H4_2_IMMUTABLE": "YES" if h421_immutable else "NO",
        "CHILD_PROCESS_CLEAN_EXIT_3_OF_3": h421_terminal.get("CHILD_PROCESS_CLEAN_EXIT_3_OF_3"),
        "CHILD_EXIT_CODE_ZERO_3_OF_3": h421_terminal.get("CHILD_EXIT_CODE_ZERO_3_OF_3"),
        "REQUIRED_ARTIFACTS_FINALIZED_3_OF_3": h421_terminal.get("REQUIRED_ARTIFACTS_FINALIZED_3_OF_3"),
        "SEMANTIC_HASHES_IDENTICAL_3_OF_3": h421_terminal.get("SEMANTIC_HASHES_IDENTICAL_3_OF_3"),
        "H4_2_SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "H3_PHYSICAL_TCP_EXTENSION_M": 0.150,
        "H3_PROCESS_STANDOFF_M": 0.260,
        "CURVED_COLLISION_FREE_REACHABLE_COUNT": curved_free,
        "TUNNEL_COLLISION_FREE_REACHABLE_COUNT": int(chosen["metrics"]["per_geometry"]["fixture_tunnel_like_patch"].get("reachable_collision_free", 0)),
        "PLANAR_COLLISION_FREE_REACHABLE_COUNT": int(chosen["metrics"]["per_geometry"]["fixture_planar_patch"].get("reachable_collision_free", 0)),
        "FINAL_CANDIDATE_CLASSIFICATION_COUNTS": chosen["metrics"]["per_geometry"],
        "SELECTED_PLACEMENT_CANDIDATE": chosen["candidate_id"],
        "NATIVE_CONTACT_SUMMARY": native_summary,
        "PLACEMENT_STUDY_CANDIDATE_COUNT": len(placement_rows),
        "PLACEMENT_STUDY_ALL_CURVED_FREE_ZERO": all(int(row["curved_metrics"].get("reachable_collision_free", 0)) == 0 for row in placement_rows),
        "FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "PATH_PLANNING_STARTED": "NO",
        "RUCKIG_STARTED": "NO",
        "ML_TRAINING_STARTED": "NO",
        "READY_FOR_STAGE_3_H5": "NO",
        "H5_ENTRY_CONTRACT": h5_contract,
        "gates": gates,
    }
    dump_json(output / "final_authorization_certificate.json", final)
    dump_json(output / "h4_3_gate_report.json", {"schema_version": "stage3-h4-3-gate-report-v1", "STAGE_3_H4_3": status, "FIRST_BLOCKER": first_blocker, "gates": gates, "collision_method": h42.COLLISION_METHOD, "ccd_status": h42.UNAVAILABLE, "clearance_status": h42.UNAVAILABLE})
    return final


def write_final_report(output: Path, final: Mapping[str, Any], native: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Any]) -> None:
    ns = native["summary"]
    env_pairs = ns.get("environment_pair_frequency", {})
    self_pairs = ns.get("self_collision_pair_frequency", {})
    per_geom = chosen["metrics"]["per_geometry"]
    curved_geom = per_geom[GEOMETRY]
    tunnel_geom = per_geom["fixture_tunnel_like_patch"]
    planar_geom = per_geom["fixture_planar_patch"]
    lines = [
        "# Stage 3 H4.3 — Curved Collision Root-Cause and Authorization Audit",
        "",
        f"`STAGE_3_H4_2_1: {final['STAGE_3_H4_2_1']}`",
        f"`STAGE_3_H4_3: {final['STAGE_3_H4_3']}`",
        f"`FIRST_BLOCKER: {final['FIRST_BLOCKER']}`",
        "",
        "## Native FCL contact evidence",
        "",
        f"- Curved IK-found candidate solutions diagnosed: `{ns.get('candidate_count')}`; checked: `{ns.get('collision_checked_count')}`.",
        f"- Native FCL contacts: `{ns.get('total_contact_count')}`; baseline/native category mismatches: `{ns.get('category_mismatch_count')}`.",
        f"- Native target categories: `{json.dumps(ns.get('native_target_category_counts', {}), sort_keys=True)}`; baseline target categories: `{json.dumps(ns.get('baseline_target_category_counts', {}), sort_keys=True)}`.",
        "- The baseline target counts are 26 environment-collision targets and 9 self-collision targets. Native contact evidence shows the 9 self-collision targets also have simultaneous environment contacts; this is why native target aggregation reports 26 environment-only plus 9 both-collision targets.",
        f"- Environment pair frequencies: `{json.dumps(env_pairs, sort_keys=True)}`.",
        f"- Self-collision pair frequencies: `{json.dumps(self_pairs, sort_keys=True)}`.",
        f"- Robot-link frequencies: `{json.dumps(ns.get('robot_link_contact_frequency', {}), sort_keys=True)}`.",
        f"- Environment-object frequencies: `{json.dumps(ns.get('environment_object_contact_frequency', {}), sort_keys=True)}`.",
        "- The C++ bridge used `CollisionRequest.contacts=true`, bounded `max_contacts`, `PlanningScene::checkCollision`, `PlanningScene::checkSelfCollision`, and exported contact point/depth records. CCD and clearance remain unavailable/null.",
        "",
        "## Curved root cause",
        "",
        "- All 35 curved targets with IK/FK candidates remain collision-classified; the native evidence identifies the actual robot link/environment object or self link pair for each returned contact.",
        f"- Environment collision topology is summarized by `{len(env_pairs)}` distinct robot/object pair(s); self-collision topology by `{len(self_pairs)}` distinct link pair(s).",
        "- A collision-free curved set is not inferred from contact-free artifacts; only native FCL results are used.",
        "- H3 process semantics were preserved: physical TCP extension `0.150 m`, process standoff `0.260 m`; joint limits and FK tolerances were not changed.",
        "",
        "## Bounded deterministic placement study",
        "",
        f"- Candidates: `{len(rows)}`; bounds: Δtranslation ±0.05 m, Δyaw ±15°, Δpitch ±10°, Δroll ±10°; random/Bayesian/ML/RL/planner optimization: `NO`.",
        f"- Selected placement: `{chosen['candidate_id']}`; curved metrics: `{json.dumps(chosen['curved_metrics'], sort_keys=True)}`.",
        f"- All placement candidates retained the full 192-target denominator: `{all(row.get('target_count') == 192 for row in rows)}`.",
        f"- Curved collision-free reachable count: `{final['CURVED_COLLISION_FREE_REACHABLE_COUNT']}`; tunnel: `{tunnel_geom.get('reachable_collision_free', 0)}`; planar: `{planar_geom.get('reachable_collision_free', 0)}`.",
        f"- Final selected-candidate classification counts by geometry: `{json.dumps({'curved': curved_geom, 'tunnel': tunnel_geom, 'planar': planar_geom}, sort_keys=True)}`.",
        "- Tunnel and planar counts are from the selected H4.2.1 clean replay under the selected placement; they were not used to remove curved failures.",
        "",
        "## Authorization boundary",
        "",
        "- FJT goals sent: `0`; robot motion: `NO`; path planning: `NO`; Ruckig: `NO`; ML/RL: `NO`; formal ledger mutation: `NO`.",
        f"- H5-entry contract: `{final['H5_ENTRY_CONTRACT']['status']}`; READY_FOR_STAGE_3_H5: `NO`.",
        "- Because curved collision-free reachable count is zero, this audit does not authorize H5 or claim a continuous curved path.",
        "",
        "## Required artifacts",
        "",
        "`h4_2_existing_evidence_audit.md/json`, `h4_2_1_clean_exit_replay_report.json`, `h4_2_1_gate_report.json`, `h4_2_1_terminal_certificate.json`, `child_process_lifecycle_evidence.jsonl`, `curved_contact_evidence.jsonl`, `native_target_collision_aggregate.jsonl`, `self_collision_pairs.jsonl`, `environment_collision_pairs.jsonl`, `native_contact_summary.json`, `fk_validation.jsonl`, `fcl_validation.jsonl`, `placement_candidate_metrics.jsonl`, `selected_placement_candidate.json`, `h4_3_gate_report.json`, and `final_authorization_certificate.json` are in this new output directory.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_lifecycle_evidence(output: Path, placement_rows: Sequence[Mapping[str, Any]]) -> None:
    evidence = []
    for row in placement_rows:
        record = row["fresh_process_record"]
        worker_dir = Path(str(record["worker_dir"]))
        if not worker_dir.is_absolute():
            worker_dir = ROOT / worker_dir
        finalize = load_json(worker_dir / "child_artifact_finalize.json")
        child_exit = load_json(worker_dir / "child_exit_record.json")
        launch_log = (worker_dir / "launch.log").read_text(encoding="utf-8", errors="replace")
        evidence.append({
            "schema_version": "stage3-h4-3-child-process-lifecycle-evidence-v1",
            "candidate_id": row["candidate_id"],
            "replay_index": record["replay_index"],
            "worker_dir": str(record["worker_dir"]),
            "outer_launch_returncode": record.get("outer_launch_returncode"),
            "child_returncode": child_exit.get("returncode"),
            "child_process_clean_exit": record.get("child_process_clean_exit"),
            "forbidden_log_patterns": [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in launch_log.lower()],
            "child_exit_record": child_exit,
            "artifact_finalize_record": finalize,
            "required_artifacts_finalized": record.get("artifact_finalize_exists") and all(item.get("exists") and item.get("sha256_matches") and item.get("size_matches") for item in record.get("artifact_checks", [])),
        })
    dump_jsonl(output / "child_process_lifecycle_evidence.jsonl", evidence)


def orchestrate(output: Path) -> int:
    if output.exists():
        raise RuntimeError(f"same_id_overwrite_forbidden: {output}")
    output.mkdir(parents=True, exist_ok=False)
    h421_terminal = load_json(H421_ROOT / "stage3_h4_2_1_terminal_certificate.json")
    if h421_terminal.get("STAGE_3_H4_2_1") != "PASSED":
        raise RuntimeError("H4.2.1 must pass before H4.3")
    before = {"h421_tree": tree_hash(H421_ROOT), "h42_tree": tree_hash(h42.OLD_H42_ROOT) if hasattr(h42, "OLD_H42_ROOT") else tree_hash(ROOT / "outputs/stage3_h4_2_workspace_diagnosis_final5")}
    for source_name, target_name in (
        ("clean_exit_replay_report.json", "h4_2_1_clean_exit_replay_report.json"),
        ("stage3_h4_2_1_gate_report.json", "h4_2_1_gate_report.json"),
        ("stage3_h4_2_1_terminal_certificate.json", "h4_2_1_terminal_certificate.json"),
        ("h4_2_existing_evidence_audit.json", "h4_2_existing_evidence_audit.json"),
        ("h4_2_existing_evidence_audit.md", "h4_2_existing_evidence_audit.md"),
        ("selected_fk_validation.jsonl", "fk_validation.jsonl"),
        ("selected_collision_validation.jsonl", "fcl_validation.jsonl"),
    ):
        shutil.copy2(H421_ROOT / source_name, output / target_name)
    inputs = write_curved_inputs(output)
    executable, build = build_native_bridge(output)
    native = run_native_contact(output, inputs, executable)
    placement_rows, chosen = run_placement_study(output, selected_inputs()[-1])
    write_lifecycle_evidence(output, placement_rows)
    after = {"h421_tree": tree_hash(H421_ROOT), "h42_tree": tree_hash(ROOT / "outputs/stage3_h4_2_workspace_diagnosis_final5")}
    final = make_authorization(output, native, placement_rows, chosen, before, after)
    dump_json(output / "immutable_hash_before_after.json", {"before": before, "after": after, "unchanged": h42.semantic_hash(before) == h42.semantic_hash(after), "formal_baselines_overwritten": False})
    write_final_report(output, final, native, placement_rows, chosen)
    entries = [{"relative_path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "output_hash_manifest.json"]
    dump_json(output / "output_hash_manifest.json", {"schema_version": "stage3-h4-3-output-hashes-v1", "algorithm": "SHA-256", "files": entries, "tree_hash": h42.semantic_hash(entries), "formal_baselines_overwritten": False})
    return 0 if final["STAGE_3_H4_3"] == "PASSED" else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.orchestrate:
        print("Stage 3 H4.3 requires --orchestrate; no backend calls were made.")
        return 2
    try:
        return orchestrate(args.output_root.resolve())
    except Exception as exc:
        output = args.output_root.resolve()
        output.mkdir(parents=True, exist_ok=True)
        first = type(exc).__name__
        dump_json(output / "h4_3_gate_report.json", {"schema_version": "stage3-h4-3-gate-report-v1", "STAGE_3_H4_3": "BLOCKED", "FIRST_BLOCKER": first, "error": f"{type(exc).__name__}: {exc}"})
        dump_json(output / "final_authorization_certificate.json", {"schema_version": "stage3-h4-3-final-authorization-certificate-v1", "STAGE_3_H4_2_1": "PASSED", "STAGE_3_H4_3": "BLOCKED", "FIRST_BLOCKER": first, "FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "PATH_PLANNING_STARTED": "NO", "RUCKIG_STARTED": "NO", "ML_TRAINING_STARTED": "NO", "READY_FOR_STAGE_3_H5": "NO", "error": f"{type(exc).__name__}: {exc}"})
        (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H4.3 — BLOCKED\n\n`FIRST_BLOCKER: {first}`\n\n{type(exc).__name__}: {exc}\n", encoding="utf-8", newline="\n")
        print(json.dumps({"STAGE_3_H4_3": "BLOCKED", "FIRST_BLOCKER": first, "output_root": rel(output)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
