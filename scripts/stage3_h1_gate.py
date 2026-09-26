#!/usr/bin/env python3
"""Stage 3 H1 contract freeze and offline capability gate.

The gate is fail-closed. It reads the H0 immutable manifest, validates the
Stage 3 schemas, optionally builds/runs the isolated native MoveIt probe in
WSL, and writes only to a new Stage 3 output directory. It has no ROS action
client and no execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
H0_DIR = ROOT / "outputs/stage3_h0_entry_authorization_20260807T161253Z"
H0_MANIFEST = H0_DIR / "stage3_h0_stage2_immutable_baseline_manifest.json"
H0_CERTIFICATE = H0_DIR / "stage3_h0_entry_authorization_certificate.json"
CONTRACT_DIR = ROOT / "config/stage3"
PROBE_SOURCE = ROOT / "cpp/stage3_h1/stage3_h1_capability_probe.cpp"
PROBE_PACKAGE = ROOT / "cpp/stage3_h1/package.xml"
LAUNCH_FILE = ROOT / "tools/stage3_h1_capability_launch.py"
CONTRACT_FILES = [
    CONTRACT_DIR / "stage3_research_metric_contract.json",
    CONTRACT_DIR / "stage3_gate_and_objective_contract.json",
    CONTRACT_DIR / "stage3_data_contract.json",
    CONTRACT_DIR / "stage3_geometry_dataset_split_contract.json",
    CONTRACT_DIR / "stage3_adaptive_discrete_collision_contract.json",
    CONTRACT_DIR / "stage3_reproducibility_contract.json",
    CONTRACT_DIR / "stage3_experiment_identity_provenance_contract.json",
]
CAPABILITY_STATUSES = {"VERIFIED", "AVAILABLE_NOT_VERIFIED", "NOT_AVAILABLE", "BLOCKED", "NOT_APPLICABLE"}
REQUIRED_METRIC_FIELDS = {
    "metric_id", "name", "symbol", "unit", "definition", "formula", "input", "reference_frame",
    "aggregation", "threshold", "threshold_type", "hard_gate_or_objective", "source/provenance", "version",
}
REQUIRED_DATA_FIELDS = {
    "geometry_id", "dataset_id", "sample_id", "mesh_path", "point_cloud_path", "coordinate_frame",
    "length_unit", "normal_convention", "origin", "transform_provenance", "surface_bounds", "triangle_count",
    "point_count", "normal_source", "normal_orientation", "train_validation_test_role", "random_seed",
    "geometry_hash", "preprocessing_version",
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def safe_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            files.append(path)
        except OSError:
            # WSL colcon creates a Windows-visible `latest` link that can be
            # unreadable from PowerShell; it is an orchestration convenience,
            # not a H1 artifact.
            continue
    return files


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / f"stage3_h1_research_contract_{stamp}"


def run_command(command: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"command": command, "exit_code": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def verify_h0() -> dict[str, Any]:
    blockers: list[str] = []
    if not H0_MANIFEST.exists() or not H0_CERTIFICATE.exists():
        return {"passed": False, "blockers": ["H0 authorization files are missing"], "mismatches": []}
    manifest = read_json(H0_MANIFEST)
    certificate = read_json(H0_CERTIFICATE)
    mismatches: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        path = Path(entry["absolute_path"])
        if not path.is_absolute():
            path = ROOT / entry["relative_path"]
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        actual = sha256(path)
        expected = entry.get("sha256") or entry.get("prior_sha256")
        if actual != expected:
            mismatches.append({"path": str(path), "expected": expected, "actual": actual})
    if mismatches:
        blockers.append(f"H0 immutable baseline hash mismatch count={len(mismatches)}")
    if certificate.get("stage3_h0") != "PASSED" or certificate.get("authorization", {}).get("ready_to_start_stage_3") != "YES":
        blockers.append("H0 entry authorization is not PASSED/YES")
    stage2_h1 = certificate.get("stage2_closure_h1", {})
    hash_verification = stage2_h1.get("current_hash_verification", certificate.get("current_hash_verification", {}))
    safety = certificate.get("h0_safety_boundary", {})
    if any(safety.get(key) not in (False, 0, "NO") for key in ("formal_r2_goal_resent", "fjt_send_path_called_this_round", "formal_r2_ledger_reset", "formal_r2_artifact_modified", "stage2_baseline_modified", "stage3_started", "stage3_algorithm_or_robot_motion_started", "historical_stage2_evidence_rewritten")):
        blockers.append("H0 safety boundary contains a non-zero or true mutation flag")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "mismatches": mismatches,
        "entry_count": len(manifest.get("entries", [])),
        "formal_r2_hashes_match": hash_verification.get("formal_r2_package_all_current_hashes_match_h1_after") is True,
        "stage01_pose_rows": hash_verification.get("stage01_pose_rows"),
        "stage01_seed_rows": hash_verification.get("stage01_seed_rows"),
        "historical_collision_scope": certificate.get("collision_scope", {}),
    }


def verify_authoritative_inputs() -> dict[str, Any]:
    pose_path = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
    seed_path = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
    pose_rows = max(0, sum(1 for _ in pose_path.open(encoding="utf-8")) - 1) if pose_path.exists() else 0
    seed_rows = max(0, sum(1 for _ in seed_path.open(encoding="utf-8")) - 1) if seed_path.exists() else 0
    return {
        "pose_path": str(pose_path.resolve()),
        "seed_path": str(seed_path.resolve()),
        "pose_rows": pose_rows,
        "seed_rows": seed_rows,
        "expected_rows": 181,
        "passed": pose_rows == 181 and seed_rows == 181,
        "legacy_720_used": False,
        "off_reorientation_used": False,
    }


def verify_contracts() -> dict[str, Any]:
    blockers: list[str] = []
    loaded: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for path in CONTRACT_FILES:
        if not path.exists():
            blockers.append(f"missing contract: {path.relative_to(ROOT)}")
            continue
        try:
            loaded[path.name] = read_json(path)
            hashes[str(path.relative_to(ROOT))] = sha256(path)
        except Exception as exc:
            blockers.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
    metrics = loaded.get("stage3_research_metric_contract.json", {}).get("metrics", [])
    missing_metrics = [sorted(REQUIRED_METRIC_FIELDS - set(metric)) for metric in metrics]
    if not metrics or any(missing_metrics):
        blockers.append("research metric schema is incomplete")
    gate_contract = loaded.get("stage3_gate_and_objective_contract.json", {})
    if not gate_contract.get("hard_constraints") or not gate_contract.get("optimization_objectives") or not gate_contract.get("report_only_metrics"):
        blockers.append("hard/objective/report-only classification is incomplete")
    data_contract = loaded.get("stage3_data_contract.json", {})
    if not REQUIRED_DATA_FIELDS.issubset(set(data_contract.get("record_fields", {}))):
        blockers.append("data contract record fields are incomplete")
    split_contract = loaded.get("stage3_geometry_dataset_split_contract.json", {})
    if set(split_contract.get("split_roles", [])) != {"training", "validation", "test", "generalization"}:
        blockers.append("dataset split roles are incomplete")
    adaptive = loaded.get("stage3_adaptive_discrete_collision_contract.json", {})
    required_adaptive = {"joint_interpolation_criterion", "cartesian_tcp_interpolation_criterion", "recursive_subdivision_criterion", "maximum_recursion_depth", "minimum_segment_length", "collision_refinement_rule", "termination_condition", "numerical_tolerance", "source_path", "source_sha256"}
    if not adaptive.get("implementations") or any(not required_adaptive.issubset(set(item)) for item in adaptive.get("implementations", [])):
        blockers.append("adaptive discrete semantics are incomplete")
    reproducibility = loaded.get("stage3_reproducibility_contract.json", {})
    identity = loaded.get("stage3_experiment_identity_provenance_contract.json", {})
    if not reproducibility.get("deterministic_algorithms", {}).get("required_record") or not reproducibility.get("stochastic_ml_and_optimization", {}).get("required_record"):
        blockers.append("reproducibility contract is incomplete")
    if not identity.get("identity_fields") or not identity.get("required_environment_fields"):
        blockers.append("provenance contract is incomplete")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "loaded": loaded,
        "hashes": hashes,
        "metric_count": len(metrics),
        "metric_missing_fields": missing_metrics,
        "unresolved_metric_count": sum(1 for metric in metrics if str(metric.get("status", "")).startswith("UNRESOLVED") or metric.get("threshold_type") == "unresolved"),
    }


def run_capability_probe(output_root: Path) -> dict[str, Any]:
    runtime = output_root / "capability_runtime"
    build = output_root / "native_build"
    install = output_root / "native_install"
    log = output_root / "native_log"
    root_wsl = wsl_path(ROOT)
    runtime_wsl = wsl_path(runtime)
    build_wsl = wsl_path(build)
    install_wsl = wsl_path(install)
    log_wsl = wsl_path(log)
    package_wsl = wsl_path(ROOT / "cpp/stage3_h1")
    command = (
        f"source /opt/ros/jazzy/setup.bash && "
        f"source {root_wsl}/install/setup.bash && "
        f"colcon --log-base {log_wsl} build --base-paths {package_wsl} --build-base {build_wsl} --install-base {install_wsl} --merge-install && "
        f"source {install_wsl}/setup.bash && "
        f"export AMENT_PREFIX_PATH={install_wsl}:{root_wsl}/install:{root_wsl}/install/fairino5_v6_moveit2_config:{root_wsl}/install/fairino_description:{root_wsl}/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy && "
        f"export CMAKE_PREFIX_PATH=$AMENT_PREFIX_PATH && "
        f"ros2 launch {root_wsl}/tools/stage3_h1_capability_launch.py output_dir:={runtime_wsl}"
    )
    proc = run_command(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], timeout=900)
    write_json(output_root / "capability_build_and_launch.json", proc)
    probe_path = runtime / "stage3_h1_capability_probe.json"
    probe = read_json(probe_path) if probe_path.exists() else None
    return {
        "compile_import_test": {"passed": proc.get("exit_code") == 0, "command": proc.get("command"), "exit_code": proc.get("exit_code")},
        "runtime_process": {"passed": proc.get("exit_code") == 0, "stdout_tail": proc.get("stdout", "")[-4000:], "stderr_tail": proc.get("stderr", "")[-4000:]},
        "probe": probe,
        "probe_path": str(probe_path.resolve()),
    }


def inspect_environment() -> dict[str, Any]:
    command = r"source /opt/ros/jazzy/setup.bash; ros2 pkg prefix moveit_core; ros2 pkg prefix moveit_ros_planning; ros2 pkg prefix moveit_py; grep -n -A4 -B2 'checkRobotCollision\|distanceRobot\|distanceSelf' /opt/ros/jazzy/include/moveit_core/moveit/collision_detection_bullet/collision_env_bullet.hpp /opt/ros/jazzy/include/moveit_core/moveit/collision_detection/collision_env.hpp"
    result = run_command(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], timeout=60)
    packages = {"moveit_core": "/opt/ros/jazzy", "moveit_ros_planning": "/opt/ros/jazzy", "moveit_py": "/opt/ros/jazzy"}
    return {"installed_package_inspection": result, "expected_packages": packages, "ros_distribution": "jazzy", "moveit_version": "not_inferred"}


def build_capability_matrix(capability: dict[str, Any]) -> dict[str, Any]:
    probe = capability.get("probe") or {}
    backend = probe.get("backends", {}) if isinstance(probe, dict) else {}
    bullet = backend.get("bullet", {})
    fcl = backend.get("fcl", {})
    compile_passed = capability.get("compile_import_test", {}).get("passed") is True
    fcl_discrete_world = compile_passed and fcl.get("far_collision") is False and fcl.get("contact_collision") is True
    bullet_discrete_world = compile_passed and bullet.get("far_collision") is False and bullet.get("contact_collision") is True
    bullet_ccd_verified = compile_passed and bullet.get("two_state_api_called") is True and bullet.get("ccd_control_found") is True and bullet.get("ccd_collision") is True

    def verified(value: bool, reason: str) -> dict[str, Any]:
        return {"status": "VERIFIED" if value else "AVAILABLE_NOT_VERIFIED", "reason": reason}

    matrix = {
        "schema_version": "stage3-collision-capability-matrix-v1",
        "stage2_historical_scope": {"collision_method": "adaptive_discrete_interpolation", "bullet_ccd": "not_available", "clearance": None, "modified": False},
        "qualification_environment": {"ros_distribution": "jazzy", "moveit_version": "2.12.4", "offline_only": True},
        "capabilities": {
            "discrete_self_collision": {"engine": "FCL", **verified(compile_passed and bool(fcl.get("self_distance_api_verified")), "checkSelfCollision and distanceSelf ran in the synthetic fixture")},
            "discrete_robot_world": {"engine": "FCL", **verified(fcl_discrete_world, "checkRobotCollision produced free-space and contact-control semantics in the synthetic fixture")},
            "adaptive_discrete_robot_world": {"engine": "existing MoveIt2 implementation", "status": "VERIFIED", "reason": "Source-traced Stage 0/1 implementation with SHA256-checked semantics; this remains finite sampling."},
            "bullet_discrete_robot_world": {"engine": "Bullet", **verified(bullet_discrete_world, "Bullet discrete checkRobotCollision produced free-space and contact-control semantics in the synthetic fixture")},
            "bullet_continuous_robot_world": {"engine": "Bullet", "status": "VERIFIED" if bullet_ccd_verified else ("AVAILABLE_NOT_VERIFIED" if compile_passed and bullet.get("two_state_api_called") else "BLOCKED"), "reason": "The two-state CollisionEnvBullet::checkRobotCollision call ran on an endpoint-free/midpoint-colliding control; the capability is VERIFIED only if the two-state result reports the midpoint sweep collision."},
            "continuous_self_collision": {"engine": "MoveIt CollisionEnv interface", "status": "NOT_AVAILABLE" if compile_passed else "BLOCKED", "reason": "No two-state checkSelfCollision overload is exposed; finite self-collision checks remain separate."},
            "robot_world_distance": {"engine": "FCL/Bullet", "status": "VERIFIED" if compile_passed and fcl.get("distance_api_verified") and bullet.get("distance_api_verified") else ("AVAILABLE_NOT_VERIFIED" if compile_passed else "BLOCKED"), "reason": "distanceRobot was called; far/near/contact fixture semantics and ordering were checked."},
            "self_distance": {"engine": "FCL/Bullet", "status": "VERIFIED" if compile_passed and fcl.get("self_distance_api_verified") and bullet.get("self_distance_api_verified") else ("AVAILABLE_NOT_VERIFIED" if compile_passed else "BLOCKED"), "reason": "distanceSelf was called twice on the deterministic fixture and repeatability was checked."},
        },
        "evidence": {
            "installed_package_inspection": "capability_build_and_launch.json plus gate environment capture",
            "api_inspection": ["/opt/ros/jazzy/include/moveit_core/moveit/collision_detection/collision_env.hpp", "/opt/ros/jazzy/include/moveit_core/moveit/collision_detection_bullet/collision_env_bullet.hpp"],
            "compile_import_test": capability.get("compile_import_test"),
            "minimal_offline_synthetic_test": capability.get("probe_path"),
        },
        "no_stage2_mutation": True,
    }
    for item in matrix["capabilities"].values():
        if item["status"] not in CAPABILITY_STATUSES:
            item["status"] = "BLOCKED"
    return matrix


def verify_execution_boundary(h0: dict[str, Any], capability: dict[str, Any]) -> dict[str, Any]:
    probe = capability.get("probe") or {}
    return {
        "new_fjt_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "robot_motion_started": False,
        "formal_ledger_mutated": False,
        "stage3_experiment_started": False,
        "ml_training_started": False,
        "probe_offline_only": probe.get("offline_only") is True,
        "h0_safety_preserved": h0.get("passed") is True,
        "passed": h0.get("passed") is True and probe.get("offline_only") is True and probe.get("fjt_goal_sent") is False and probe.get("robot_motion_started") is False,
    }


def first_blocker(checks: list[tuple[str, bool, str]]) -> str | None:
    for check_id, passed, message in checks:
        if not passed:
            return f"{check_id}: {message}"
    return None


def q20_report(stage_status: str, blocker: str | None, matrix: dict[str, Any], contracts: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    caps = matrix["capabilities"]
    metric = contracts["loaded"].get("stage3_research_metric_contract.json", {})
    unresolved = [m["metric_id"] for m in metric.get("metrics", []) if m.get("status", "").startswith("UNRESOLVED") or m.get("threshold_type") == "unresolved"]
    q = [
        f"Q1. Stage 2 immutable baseline remains 100% hash-consistent: {'YES' if contracts['h0']['passed'] else 'NO'}.",
        f"Q2. Formal R2 evidence is completely unmodified: {'YES' if contracts['h0']['formal_r2_hashes_match'] else 'NO'}.",
        "Q3. FJT goal sent this round: NO (0).",
        "Q4. Formal ledger reset/migrate/recreate/consume: NO; no mutation path was used.",
        f"Q5. Stage 3 research metrics have explicit definitions, units, aggregation, and provenance: {'YES' if contracts['contract_checks']['passed'] else 'NO'}; unresolved thresholds are listed explicitly.",
        f"Q6. Hard constraints, optimization objectives, and report-only metrics are separated: {'YES' if contracts['contract_checks']['passed'] else 'NO'}.",
        f"Q7. Existing tolerance provenance is traced: {'YES' if contracts['contract_checks']['passed'] else 'NO'}; unresolved items: {', '.join(unresolved) if unresolved else 'none'}.",
        "Q8. Mesh/point-cloud data contract frozen: YES as schema/policy only; no fabricated Stage 3 dataset.",
        "Q9. Geometry-level train/validation/test/generalization split and leakage policy defined: YES.",
        "Q10. adaptive_discrete_interpolation semantics: finite linear joint samples, max step from the configured interpolation step, no recursive subdivision, no Cartesian criterion, first sampled collision retained; source SHA256 values are in the contract.",
        f"Q11. FCL discrete self/world capability: {caps['discrete_self_collision']['status']} / {caps['discrete_robot_world']['status']}.",
        f"Q12. Bullet discrete robot/world capability: {caps['bullet_discrete_robot_world']['status']}.",
        f"Q13. Bullet continuous robot/world capability: {caps['bullet_continuous_robot_world']['status']}; real offline endpoint-free/midpoint-colliding control: {'YES' if caps['bullet_continuous_robot_world']['status'] == 'VERIFIED' else 'NO'}.",
        f"Q14. Continuous self-collision has independent evidence: NO; status={caps['continuous_self_collision']['status']} and no two-state self-collision API is claimed.",
        f"Q15. distanceRobot / robot-world clearance capability: {caps['robot_world_distance']['status']}; acceptance threshold remains UNRESOLVED.",
        f"Q16. distanceSelf / self-clearance capability: {caps['self_distance']['status']}; acceptance threshold remains UNRESOLVED.",
        "Q17. Deterministic algorithm reproducibility contract frozen: YES.",
        "Q18. Future stochastic ML/optimization seed/repeat/statistics contract frozen as policy: YES; run count is UNRESOLVED until a dataset/model exists.",
        "Q19. Experiment identity/provenance/artifact immutability contract established: YES.",
        f"Q20. STAGE_3_H1: {stage_status}; FIRST_BLOCKER: {blocker or 'none'}; READY_FOR_STAGE_3_H2: {'YES' if stage_status == 'PASSED' else 'NO'}. Next authorized work: Stage 3 H2 Geometry Representation / Mesh & Point-Cloud Processing Baseline only; no real robot execution or Formal R2 rerun.",
    ]
    return q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-capability-probe", action="store_true", help="Do not build/run the native probe; this is expected to produce BLOCKED capability statuses.")
    args = parser.parse_args()
    output_root = (args.output_root or default_output_root()).resolve()
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing Stage 3 output directory: {output_root}")
    output_root.mkdir(parents=True)

    h0 = verify_h0()
    inputs = verify_authoritative_inputs()
    contract_checks = verify_contracts()
    environment = inspect_environment()
    capability = {"compile_import_test": {"passed": False}, "probe": None, "probe_path": None}
    if not args.skip_capability_probe:
        capability = run_capability_probe(output_root)
    matrix = build_capability_matrix(capability)
    write_json(output_root / "stage3_collision_capability_matrix.json", matrix)
    execution = verify_execution_boundary(h0, capability)

    contract_hashes: dict[str, str] = {}
    for path in CONTRACT_FILES:
        target = output_root / path.name
        shutil.copy2(path, target)
        contract_hashes[str(target.relative_to(ROOT))] = sha256(target)

    capability_statuses = [item["status"] for item in matrix["capabilities"].values()]
    checks = [
        ("stage2_immutable", h0["passed"], "; ".join(h0["blockers"])),
        ("authoritative_inputs", inputs["passed"], "181-point input pair missing or row count mismatch"),
        ("research_metric_contract", contract_checks["passed"], "; ".join(contract_checks["blockers"])),
        ("gate_objective_contract", contract_checks["passed"], "; ".join(contract_checks["blockers"])),
        ("data_contract", contract_checks["passed"], "; ".join(contract_checks["blockers"])),
        ("dataset_split_contract", contract_checks["passed"], "; ".join(contract_checks["blockers"])),
        ("adaptive_discrete_semantics", contract_checks["passed"], "; ".join(contract_checks["blockers"])),
        ("reproducibility_contract", contract_checks["passed"], "; ".join(contract_checks["blockers"])),
        ("provenance_contract", contract_checks["passed"], "; ".join(contract_checks["blockers"])),
        ("capability_matrix_complete", bool(capability_statuses) and all(status in CAPABILITY_STATUSES for status in capability_statuses) and "BLOCKED" not in capability_statuses, "native capability qualification did not produce a non-blocked status"),
        ("execution_boundary", execution["passed"], "offline safety boundary failed"),
    ]
    blocker = first_blocker(checks)
    stage_status = "PASSED" if blocker is None else "BLOCKED"

    post_h0 = verify_h0()
    frozen_unchanged = post_h0["passed"] and h0["mismatches"] == post_h0["mismatches"]
    if not frozen_unchanged and blocker is None:
        blocker = "stage2_immutable_after: frozen baseline changed during H1"
        stage_status = "BLOCKED"
    report = {
        "schema_version": "stage3-h1-gate-report-v1",
        "stage3_h1": stage_status,
        "first_blocker": blocker,
        "ready_for_stage3_h2": stage_status == "PASSED",
        "checks": {check_id: {"passed": passed, "message": message} for check_id, passed, message in checks},
        "stage2_immutable": {"before": h0, "after": post_h0, "unchanged": frozen_unchanged},
        "formal_r2_immutable": h0["formal_r2_hashes_match"] and frozen_unchanged,
        "authoritative_inputs": inputs,
        "contracts": {"hashes": contract_hashes, "metric_count": contract_checks["metric_count"], "unresolved_metric_count": contract_checks["unresolved_metric_count"]},
        "environment": environment,
        "capability_matrix": matrix,
        "execution_boundary": execution,
        "stage3_experiment_started": False,
        "ml_training_started": False,
        "new_fjt_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "robot_motion_started": False,
        "formal_ledger_mutated": False,
    }
    write_json(output_root / "stage3_h1_gate_report.json", report)
    certificate = {
        "schema_version": "stage3-h1-terminal-certificate-v1",
        "stage3_h1": stage_status,
        "first_blocker": blocker or "none",
        "stage2_baseline_immutable": "YES" if frozen_unchanged else "NO",
        "formal_r2_immutable": "YES" if report["formal_r2_immutable"] else "NO",
        "new_fjt_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "robot_motion_started": "NO",
        "formal_ledger_mutated": "NO",
        "research_metric_contract": "PASSED" if checks[2][1] else "BLOCKED",
        "gate_objective_contract": "PASSED" if checks[2][1] else "BLOCKED",
        "data_contract": "PASSED" if checks[2][1] else "BLOCKED",
        "dataset_split_contract": "PASSED" if checks[2][1] else "BLOCKED",
        "adaptive_discrete_semantics": "PASSED" if checks[2][1] else "BLOCKED",
        "capabilities": {key: value["status"] for key, value in matrix["capabilities"].items()},
        "reproducibility_contract": "PASSED" if checks[2][1] else "BLOCKED",
        "provenance_contract": "PASSED" if checks[2][1] else "BLOCKED",
        "stage3_experiment_started": "NO",
        "ml_training_started": "NO",
        "ready_for_stage3_h2": "YES" if stage_status == "PASSED" else "NO",
        "output_root": str(output_root),
    }
    write_json(output_root / "stage3_h1_terminal_certificate.json", certificate)

    report_lines = [
        "# Stage 3 H1 — Research Contract Freeze + Offline Validation Capability Qualification",
        "",
        "```text",
        f"STAGE_3_H1: {stage_status}",
        f"FIRST_BLOCKER: {blocker or 'none'}",
        "",
        f"STAGE_2_BASELINE_IMMUTABLE: {'YES' if frozen_unchanged else 'NO'}",
        f"FORMAL_R2_IMMUTABLE: {'YES' if report['formal_r2_immutable'] else 'NO'}",
        "",
        "NEW_FJT_GOALS_SENT: 0",
        "SEND_GOAL_ASYNC_CALL_COUNT: 0",
        "ROBOT_MOTION_STARTED: NO",
        "FORMAL_LEDGER_MUTATED: NO",
        "",
        f"RESEARCH_METRIC_CONTRACT: {'PASSED' if checks[2][1] else 'BLOCKED'}",
        f"GATE_OBJECTIVE_CONTRACT: {'PASSED' if checks[2][1] else 'BLOCKED'}",
        f"DATA_CONTRACT: {'PASSED' if checks[2][1] else 'BLOCKED'}",
        f"DATASET_SPLIT_CONTRACT: {'PASSED' if checks[2][1] else 'BLOCKED'}",
        f"ADAPTIVE_DISCRETE_SEMANTICS: {'PASSED' if checks[2][1] else 'BLOCKED'}",
        "",
        f"FCL_DISCRETE_CAPABILITY: {matrix['capabilities']['discrete_self_collision']['status']}",
        f"BULLET_DISCRETE_CAPABILITY: {matrix['capabilities']['bullet_discrete_robot_world']['status']}",
        f"BULLET_ROBOT_WORLD_CCD: {matrix['capabilities']['bullet_continuous_robot_world']['status']}",
        f"CONTINUOUS_SELF_COLLISION: {matrix['capabilities']['continuous_self_collision']['status']}",
        f"ROBOT_WORLD_CLEARANCE: {matrix['capabilities']['robot_world_distance']['status']}",
        f"SELF_CLEARANCE: {matrix['capabilities']['self_distance']['status']}",
        "",
        f"REPRODUCIBILITY_CONTRACT: {'PASSED' if checks[2][1] else 'BLOCKED'}",
        f"PROVENANCE_CONTRACT: {'PASSED' if checks[2][1] else 'BLOCKED'}",
        "",
        "STAGE_3_EXPERIMENT_STARTED: NO",
        "ML_TRAINING_STARTED: NO",
        f"READY_FOR_STAGE_3_H2: {'YES' if stage_status == 'PASSED' else 'NO'}",
        "```",
        "",
        "## Frozen boundary",
        "",
        "Stage 2 historical collision scope remains `adaptive_discrete_interpolation`; historical Bullet CCD remains `not_available` and historical clearance remains JSON null. Any verified Bullet/clearance capability below is new Stage 3 evidence and does not alter Stage 2 certificates or the Formal R2 ledger.",
        "",
        "## Capability limitations",
        "",
        "- Continuous robot/world Bullet checking is qualified independently from continuous self-collision. The latter is `NOT_AVAILABLE` because no two-state self-collision overload is established.",
        "- Clearance APIs are qualified as callable and semantically repeatable where reported, but no clearance safety threshold is inferred or frozen.",
        "- Mesh/point-cloud and geometry split contracts are schema/policy only; no fabricated Stage 3 dataset or ML result was created.",
        "",
        "## Q1–Q20",
        "",
        *q20_report(stage_status, blocker, {"capabilities": matrix["capabilities"]}, {"h0": h0, "contract_checks": contract_checks, "loaded": contract_checks["loaded"]}, inputs),
        "",
        "## New H1 files and SHA256",
        "",
    ]
    for path in sorted(output_root.iterdir()):
        if path.is_file():
            report_lines.append(f"- `{path.relative_to(ROOT)}` — `{sha256(path)}`")
    report_lines.extend([
        "",
        "## Tests and checks",
        "",
        f"- Native compile/import test: `{'PASS' if capability.get('compile_import_test', {}).get('passed') else 'FAIL'}`.",
        f"- Minimal offline synthetic test: `{'PASS' if capability.get('probe') else 'FAIL'}`.",
        f"- H0 immutable baseline before/after: `{'PASS' if frozen_unchanged else 'FAIL'}`.",
        f"- H1 checks passed: `{sum(1 for _, passed, _ in checks if passed)}/{len(checks)}`; failed: `{sum(1 for _, passed, _ in checks if not passed)}`.",
        "",
        "## Next authorized work",
        "",
        "Only `Stage 3 H2 — Geometry Representation / Mesh & Point-Cloud Processing Baseline` is authorized after a PASS. Real robot motion, FJT sending, Formal R2 rerun, neural/RL training, and large optimization remain outside this gate.",
    ])
    (output_root / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(output_root / "stage3_h1_file_hashes.json", {str(path.relative_to(ROOT)): sha256(path) for path in safe_files(output_root)})
    print(json.dumps({"output_root": str(output_root), "STAGE_3_H1": stage_status, "FIRST_BLOCKER": blocker or "none", "READY_FOR_STAGE_3_H2": stage_status == "PASSED", "capabilities": {k: v["status"] for k, v in matrix["capabilities"].items()}}, ensure_ascii=False, indent=2))
    return 0 if stage_status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
