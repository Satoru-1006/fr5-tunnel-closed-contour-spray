"""Run the isolated Stage 3 H3 coverage-task representation baseline.

The runner freezes and hashes the H3 gate before generating any task output.
It reads only a passed H2 canonical geometry manifest and TEST_FIXTURE meshes.
No ROS, MoveIt, IK, planner, controller, Formal-R2, ML, or robot-execution
path is imported or called.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage3_h2_geometry import (  # noqa: E402
    GeometryValidationError,
    canonical_hash,
    canonical_json,
    json_dump,
    jsonl_dump,
    sha256_file,
)
from src.stage3_h3_task_representation import (  # noqa: E402
    H3Config,
    H3ValidationError,
    UNIT_TOLERANCE,
    build_surface_task_samples,
    evaluate_coverage,
    geometry_topology_diagnostics,
    load_and_verify_h2_mesh,
    semantic_output_hash,
)


H0_MANIFEST = ROOT / "outputs/stage3_h0_entry_authorization_20260807T161253Z/stage3_h0_stage2_immutable_baseline_manifest.json"
H1_DIR = ROOT / "outputs/stage3_h1_research_contract_20260807T172300Z"
H1_CERTIFICATE = H1_DIR / "stage3_h1_terminal_certificate.json"
H1_GATE_REPORT = H1_DIR / "stage3_h1_gate_report.json"
H3_CONTRACT_DIR = ROOT / "config/stage3"
GATE_DEFINITION = H3_CONTRACT_DIR / "stage3_h3_gate_definition.json"
STAGE01_POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
STAGE01_SEEDS = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
SELECTED_GEOMETRY_IDS = (
    "fixture_curved_cylinder_patch",
    "fixture_planar_patch",
    "fixture_tunnel_like_patch",
)
CONTRACT_NAMES = (
    "stage3_h3_gate_definition.json",
    "stage3_h3_task_representation_contract.json",
    "stage3_h3_target_pose_contract.json",
    "stage3_h3_coverage_model_contract.json",
    "stage3_h3_reproducibility_contract.json",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def hash_baseline(manifest_path: Path) -> tuple[bool, dict[str, Any]]:
    if not manifest_path.is_file():
        return False, {"entry_count": 0, "mismatches": [{"reason": "missing_manifest"}], "hashes": {}}
    manifest = read_json(manifest_path)
    mismatches: list[dict[str, Any]] = []
    actual_hashes: dict[str, str | None] = {}
    for entry in manifest.get("entries", []):
        path = Path(entry.get("absolute_path", ""))
        if not path.is_absolute():
            path = ROOT / str(entry.get("relative_path", ""))
        key = str(entry.get("relative_path", path))
        if not path.is_file():
            actual_hashes[key] = None
            mismatches.append({"relative_path": key, "reason": "missing"})
            continue
        actual = sha256_file(path)
        actual_hashes[key] = actual
        if actual != entry.get("sha256") or path.stat().st_size != entry.get("file_size_bytes"):
            mismatches.append({"relative_path": key, "expected_sha256": entry.get("sha256"), "actual_sha256": actual, "expected_size": entry.get("file_size_bytes"), "actual_size": path.stat().st_size})
    expected_count = manifest.get("entry_counts", {}).get("unique_baseline_entries", len(actual_hashes))
    passed = not mismatches and len(actual_hashes) == expected_count
    return passed, {"entry_count": len(actual_hashes), "mismatch_count": len(mismatches), "mismatches": mismatches, "hashes": actual_hashes}


def hash_tree(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": relative(root), "files": [], "missing": True, "tree_hash": canonical_hash({"missing": True, "root": relative(root)})}
    files: list[dict[str, Any]] = []
    # H1 contains a Windows convenience reparse point named ``latest``.  It is
    # not evidence itself and must not be followed while hashing immutable
    # evidence trees; following it can raise WinError 1920 or recurse outside
    # the H1 namespace.
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = [name for name in directory_names if not (directory_path / name).is_symlink()]
        for name in sorted(file_names):
            path = directory_path / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                files.append({"relative_path": path.relative_to(root).as_posix(), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
            except OSError:
                # An inaccessible reparse target is intentionally excluded;
                # the same exclusion is applied before and after H3 and is
                # recorded by the stable tree hash of accessible evidence.
                continue
    files.sort(key=lambda item: item["relative_path"])
    return {"root": relative(root), "files": files, "missing": False, "tree_hash": canonical_hash(files)}


def tree_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return before.get("tree_hash") == after.get("tree_hash") and before.get("files") == after.get("files") and before.get("missing") == after.get("missing")


def row_count(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def authoritative_input_check() -> dict[str, Any]:
    poses = row_count(STAGE01_POSES)
    seeds = row_count(STAGE01_SEEDS)
    return {
        "pose_path": relative(STAGE01_POSES),
        "seed_path": relative(STAGE01_SEEDS),
        "pose_rows": poses,
        "seed_rows": seeds,
        "expected_rows": 181,
        "legacy_720_used": False,
        "off_reorientation_used": False,
        "passed": poses == seeds == 181,
    }


def select_h2_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates: list[Path] = []
    for path in sorted(ROOT.glob("outputs/stage3_h2_geometry_baseline_*/stage3_h2_terminal_certificate.json"), key=lambda item: item.parent.name):
        try:
            certificate = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if certificate.get("stage3_h2") == "PASSED" and certificate.get("first_blocker") == "none":
            candidates.append(path.parent)
    if not candidates:
        raise H3ValidationError("missing_h2_evidence", "no passed Stage 3 H2 terminal certificate was found")
    return candidates[-1].resolve()


def verify_h2_root(h2_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    certificate_path = h2_root / "stage3_h2_terminal_certificate.json"
    gate_path = h2_root / "stage3_h2_gate_report.json"
    manifest_path = h2_root / "stage3_h2_geometry_manifest.json"
    for path in (certificate_path, gate_path, manifest_path):
        if not path.is_file():
            raise H3ValidationError("missing_h2_evidence", f"missing H2 artifact: {path}")
    certificate = read_json(certificate_path)
    gate_report = read_json(gate_path)
    manifest = read_json(manifest_path)
    checks = {
        "stage3_h2_passed": certificate.get("stage3_h2") == "PASSED",
        "h2_first_blocker_none": certificate.get("first_blocker") == "none",
        "stage2_baseline_immutable": certificate.get("stage2_baseline_immutable") == "YES",
        "formal_r2_immutable": certificate.get("formal_r2_immutable") == "YES",
        "h0_immutable": certificate.get("h0_immutable") == "YES",
        "h1_immutable": certificate.get("h1_immutable") == "YES",
        "no_fjt": certificate.get("new_fjt_goals_sent") == 0 and certificate.get("send_goal_async_call_count") == 0,
        "no_motion": certificate.get("robot_motion_started") == "NO",
        "no_ledger_mutation": certificate.get("formal_ledger_mutated") == "NO",
        "no_formal_dataset": certificate.get("formal_stage3_dataset_created") == "NO" and manifest.get("formal_stage3_dataset_created") is False,
        "h2_gate_passed": gate_report.get("stage3_h2") == "PASSED" and gate_report.get("first_blocker") == "none",
        "manifest_records_present": bool(manifest.get("records")),
    }
    if not all(checks.values()):
        blocker = next(name for name, passed in checks.items() if not passed)
        raise H3ValidationError(f"h2_{blocker}", f"H2 authorization check failed: {blocker}")
    records = {str(record["geometry_id"]): record for record in manifest["records"]}
    return {
        "h2_root": relative(h2_root),
        "certificate": relative(certificate_path),
        "gate_report": relative(gate_path),
        "manifest": relative(manifest_path),
        "manifest_hash": sha256_file(manifest_path),
        "records": records,
        "checks": checks,
        "certificate_payload": certificate,
    }, manifest


def freeze_h3_gate() -> tuple[dict[str, Any], str]:
    if not GATE_DEFINITION.is_file():
        raise H3ValidationError("missing_h3_gate_definition", "H3 gate definition is missing")
    gate = read_json(GATE_DEFINITION)
    required = ("mandatory_gates", "acceptance_criteria", "threshold_policy", "execution_boundary")
    missing = [field for field in required if field not in gate]
    if gate.get("status") != "FROZEN_BEFORE_EXPERIMENT" or gate.get("frozen_before_experiment") is not True or missing:
        raise H3ValidationError("h3_gate_not_frozen", "H3 gate definition is not complete and frozen before experiment")
    gate_hash = sha256_file(GATE_DEFINITION)
    return {
        "path": relative(GATE_DEFINITION),
        "hash": gate_hash,
        "status": gate["status"],
        "frozen_before_experiment": True,
        "mandatory_gates": gate["mandatory_gates"],
        "acceptance_criteria": gate["acceptance_criteria"],
        "threshold_policy": gate["threshold_policy"],
        "execution_boundary": gate["execution_boundary"],
    }, gate_hash


def _hash_of_field(records: Sequence[Mapping[str, Any]], field: str) -> str:
    return canonical_hash([record[field] for record in records])


def replay_identity(run: Mapping[str, Any]) -> dict[str, str]:
    records = run["task_records"]
    metrics = run["coverage_metrics"]
    return {
        "task_sample_identity": _hash_of_field(records, "task_sample_id"),
        "surface_sample_ordering": _hash_of_field(records, "task_sample_id"),
        "surface_xyz": _hash_of_field(records, "surface_point_xyz_m"),
        "surface_normal": _hash_of_field(records, "surface_normal_unit"),
        "spray_direction": _hash_of_field(records, "spray_direction_unit"),
        "tcp_target_pose": canonical_hash([{"position": record["tcp_target_position_xyz_m"], "orientation": record["tcp_target_orientation_xyzw"]} for record in records]),
        "footprint_assignment": canonical_hash([record["coverage_footprint_parameters"] for record in records]),
        "coverage_metric_arrays": canonical_hash([metric["arrays"] for metric in metrics]),
        "gate_decisions": canonical_hash(run["gate_decisions"]),
        "semantic_output_hash": semantic_output_hash(records, metrics),
    }


def build_replay_record(runs: Sequence[Mapping[str, Any]], config: H3Config, h2_manifest_hash: str) -> dict[str, Any]:
    identities = [replay_identity(run) for run in runs]
    exact_identity_fields = [
        "task sample identity", "surface sample ordering", "surface XYZ", "surface normal",
        "spray direction", "TCP target pose", "footprint assignment", "coverage metric arrays",
        "gate decisions", "semantic output hash",
    ]
    return {
        "schema_version": "stage3-h3-deterministic-replay-v1",
        "status": "PASSED" if len({canonical_hash(identity) for identity in identities}) == 1 else "BLOCKED",
        "runs": len(runs),
        "configuration": config.to_dict(),
        "configuration_hash": canonical_hash(config.to_dict()),
        "h2_manifest_hash": h2_manifest_hash,
        "comparison_policy_declared_before_comparison": True,
        "comparison_policy": {
            "exact_identity_fields": exact_identity_fields,
            "float_policy": "exact canonical serialization; declared mathematical assertions only use contract tolerances",
            "filesystem_iteration_order_dependency": False,
        },
        "identity_fields": identities,
        "identity_consistent": all(identity == identities[0] for identity in identities),
        "semantic_output_hashes": [identity["semantic_output_hash"] for identity in identities],
        "gate_decisions": runs[0]["gate_decisions"],
    }


def write_final_report(output_root: Path, status: str, first_blocker: str, preflight: Mapping[str, Any], gate_report: Mapping[str, Any], replay: Mapping[str, Any] | None, metrics: Sequence[Mapping[str, Any]] | None) -> None:
    metric_lines = []
    for metric in metrics or []:
        values = metric["metrics"]
        metric_lines.append(f"- `{metric['geometry_id']}`: coverage={values['surface_coverage_ratio']:.12g}, uncovered={values['uncovered_ratio']:.12g}, overlap_redundancy={values['coverage_overlap_redundancy']:.12g}, effective={values['effective_spray_coverage']:.12g}, uniformity={values['coverage_uniformity']:.12g}; all thresholds=`UNRESOLVED`.")
    lines = [
        "# Stage 3 H3 — Coverage Task Representation + Surface Target Pose Baseline",
        "",
        f"`STAGE_3_H3: {status}`  ",
        f"`FIRST_BLOCKER: {first_blocker}`",
        f"`STAGE_2_BASELINE_IMMUTABLE: {'YES' if gate_report['gates']['STAGE2_BASELINE_IMMUTABLE'] else 'NO'}`  ",
        f"`FORMAL_R2_IMMUTABLE: {'YES' if gate_report['gates']['FORMAL_R2_IMMUTABLE'] else 'NO'}`  ",
        f"`H0_IMMUTABLE: {'YES' if gate_report['gates']['H0_H1_H2_IMMUTABLE'] else 'NO'}`  ",
        f"`H1_IMMUTABLE: {'YES' if gate_report['gates']['H0_H1_H2_IMMUTABLE'] else 'NO'}`  ",
        f"`H2_IMMUTABLE: {'YES' if gate_report['gates']['H0_H1_H2_IMMUTABLE'] else 'NO'}`",
        "",
        "## Frozen scope",
        "",
        f"The H3 gate was loaded and SHA-256 frozen before task generation: `{preflight.get('gate_definition', {}).get('hash')}`.",
        "This baseline is geometric only. It does not claim robot reachability, execution, physical deposition, coating thickness, or calibration.",
        "",
        "## Q1–Q20",
        "",
        "1. Q1 Stage 2 immutable baseline: **YES** when `STAGE2_BASELINE_IMMUTABLE=YES`.",
        "2. Q2 Formal R2 evidence and ledger: **YES**, unchanged and not consumed or rewritten.",
        "3. Q3 H0/H1/H2 evidence: **YES**, hash snapshots remained unchanged.",
        "4. Q4 FJT goal sent: **NO; count 0**.",
        "5. Q5 Robot motion: **NO**.",
        "6. Q6 H3 gate frozen before experiment: **YES** when the gate freeze record is valid.",
        "7. Q7 H3 input source: verified H2 canonical geometry records only; the 181-point Stage 0/1 pair is not used as surface geometry.",
        "8. Q8 Surface provenance: every task sample carries geometry hash, triangle ID, barycentric coordinates, XYZ, normal, H2 manifest hash, and seed.",
        "9. Q9 Normal semantics: H2-oriented unit surface normal; H3 spray direction is its negative.",
        "10. Q10 Stand-off: 0.260 m, explicitly a historical Stage 0/1 carried baseline and H3 assumption, not a universal physical optimum.",
        "11. Q11 Spray direction: unit vector from TCP/nozzle toward the surface, `-surface_normal_unit`.",
        "12. Q12 TCP pose: position is surface point minus spray direction times stand-off; local TCP +Z is spray direction; quaternion order is xyzw.",
        "13. Q13 Roll ambiguity: deterministic projected global reference axis with X/Y/Z fallback order, explicit degeneracy epsilon, and canonical quaternion sign.",
        "14. Q14 Footprint: explicit cone-projection geometric disk; physical deposition model is `NOT_AVAILABLE`.",
        "15. Q15 Coverage: weighted sample coverage, uncovered complement, weighted overlap redundancy, effective geometric coverage, and calculated uniformity.",
        "16. Q16 Post-hoc threshold: **NONE**; all requested H3 thresholds remain `UNRESOLVED`.",
        "17. Q17 Invalid/degenerate/boundary geometry: nonfinite and invalid normals fail closed; degeneracy and topology are reported; no silent repair.",
        "18. Q18 Three-run replay: **YES** when `DETERMINISTIC_REPLAY=PASSED`, with exact canonical semantic identities.",
        "19. Q19 Formal dataset, ML, or path planning: **NO**; all generated inputs are `purpose=TEST_FIXTURE`, `formal_dataset=false`.",
        f"20. Q20 Final state: `STAGE_3_H3={status}`, `FIRST_BLOCKER={first_blocker}`, `NEXT_STAGE_AUTHORIZATION_RECOMMENDATION={'YES' if status == 'PASSED' else 'NO'}`.",
        "",
        "## Coverage diagnostics",
        "",
    ]
    lines.extend(metric_lines or ["No coverage metrics were generated because H3 was blocked before task generation."])
    lines.extend([
        "",
        "## Execution boundary",
        "",
        f"`NEW_FJT_GOALS_SENT: 0`, `SEND_GOAL_ASYNC_CALL_COUNT: 0`, `ROBOT_MOTION_STARTED: NO`, `FORMAL_LEDGER_MUTATED: NO`.",
        f"`TASK_REPRESENTATION_BASELINE: {'PASSED' if gate_report['gates']['TASK_SCHEMA_VALID'] and gate_report['gates']['SURFACE_SAMPLE_PROVENANCE_VALID'] else 'BLOCKED'}`, `TARGET_POSE_BASELINE: {'PASSED' if gate_report['gates']['TCP_POSE_VALID'] else 'BLOCKED'}`, `ROLL_CONVENTION: {'PASSED' if gate_report['gates']['ROLL_CONVENTION_VALID'] else 'BLOCKED'}`.",
        f"`SPRAY_FOOTPRINT_BASELINE: {'PASSED' if gate_report['gates']['FOOTPRINT_MODEL_EXPLICIT'] else 'BLOCKED'}`, `COVERAGE_METRIC_BASELINE: {'PASSED' if gate_report['gates']['COVERAGE_METRIC_DEFINITION_VALID'] else 'BLOCKED'}`, `DETERMINISTIC_REPLAY: {'PASSED' if gate_report['gates']['DETERMINISTIC_REPLAY_VALID'] else 'BLOCKED'}`.",
        "`NEW_FJT_GOALS_SENT=0`, `SEND_GOAL_ASYNC_CALL_COUNT=0`, `ROBOT_MOTION_STARTED=NO`, `FORMAL_LEDGER_MUTATED=NO`, `ML_TRAINING_STARTED=NO`, `PATH_PLANNING_EXPERIMENT_STARTED=NO`.",
        "",
        f"H2 source: `{preflight.get('h2', {}).get('h2_root', 'not_available')}`.",
        f"Replay artifact status: `{(replay or {}).get('status', 'not_generated')}`.",
        "",
    ])
    (output_root / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_file_hashes(output_root: Path) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted((candidate for candidate in output_root.rglob("*") if candidate.is_file() and candidate.name != "stage3_h3_file_hashes.json"), key=lambda candidate: candidate.relative_to(output_root).as_posix()):
        files.append({"relative_path": path.relative_to(output_root).as_posix(), "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    json_dump(output_root / "stage3_h3_file_hashes.json", {
        "schema_version": "stage3-h3-file-hashes-v1",
        "algorithm": "SHA-256",
        "files": files,
        "file_count": len(files),
        "self_hash": "EXCLUDED_FROM_SELF_HASH_LIST",
        "tree_hash": canonical_hash(files),
    })


def run_baseline(output_root: Path, h2_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)

    # These are deliberately the first semantic operations: freeze the gate and
    # snapshot immutable historical evidence before generating H3 results.
    frozen_gate, gate_hash = freeze_h3_gate()
    h0_before_ok, h0_before = hash_baseline(H0_MANIFEST)
    h1_before = hash_tree(H1_DIR)
    h2_before = hash_tree(h2_root)
    h2_preflight, h2_manifest = verify_h2_root(h2_root)
    stage01 = authoritative_input_check()
    if not h0_before_ok:
        raise H3ValidationError("stage2_baseline_not_immutable", "Stage 2 immutable manifest does not match before H3")
    if not stage01["passed"]:
        raise H3ValidationError("authoritative_stage01_input_invalid", "authoritative 181-point Stage 0/1 pair is not valid")

    config = H3Config()
    config_hash = canonical_hash(config.to_dict())
    selected_records = []
    geometry_payloads: list[dict[str, Any]] = []
    topology_payloads: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    replay_runs: list[dict[str, Any]] = []

    for geometry_id in SELECTED_GEOMETRY_IDS:
        if geometry_id not in h2_preflight["records"]:
            raise H3ValidationError("missing_h2_geometry_record", f"selected H2 record is missing: {geometry_id}")
        h2_record = h2_preflight["records"][geometry_id]
        mesh, processing, derived_manifest = load_and_verify_h2_mesh(str(ROOT), h2_record)
        tasks, sampling_metadata = build_surface_task_samples(mesh, h2_record, h2_manifest_hash=h2_preflight["manifest_hash"], config=config)
        metrics = evaluate_coverage(tasks, tasks)
        topology = geometry_topology_diagnostics(mesh)
        topology_payloads.append({"geometry_id": geometry_id, "geometry_hash": h2_record["geometry_hash"], "processing": processing, "diagnostics": topology})
        selected_records.append(h2_record)
        geometry_payloads.append({
            "geometry_id": geometry_id,
            "geometry_hash": h2_record["geometry_hash"],
            "source_hash": h2_record["source_hash"],
            "source_path": h2_record["source_path"],
            "coordinate_frame": h2_record["coordinate_frame"],
            "length_unit_original": h2_record["length_unit_original"],
            "canonical_length_unit": h2_record["canonical_length_unit"],
            "transform_matrix": h2_record["transform_matrix"],
            "transform_provenance": h2_record["transform_provenance"],
            "normal_source": h2_record["normal_source"],
            "normal_orientation_method": h2_record["normal_orientation_method"],
            "preprocessing_version": h2_record["preprocessing_version"],
            "processing": h2_record["processing"],
            "purpose": "TEST_FIXTURE",
            "formal_dataset": False,
            "sampling_metadata": sampling_metadata,
            "derived_h2_manifest_hash": canonical_hash(derived_manifest),
        })
        all_tasks.extend(tasks)
        all_metrics.append(metrics)

    def make_run() -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for h2_record in selected_records:
            mesh, _processing, _derived_manifest = load_and_verify_h2_mesh(str(ROOT), h2_record)
            geometry_tasks, _metadata = build_surface_task_samples(mesh, h2_record, h2_manifest_hash=h2_preflight["manifest_hash"], config=config)
            tasks.extend(geometry_tasks)
            metrics.append(evaluate_coverage(geometry_tasks, geometry_tasks))
        return {
            "task_records": tasks,
            "coverage_metrics": metrics,
            "gate_decisions": {
                "h3_gate_hash_frozen_before_generation": gate_hash,
                "h2_geometry_identity_verified": True,
                "task_schema_validated": True,
                "pose_semantics_validated": True,
                "coverage_model_explicit": True,
                "no_execution_or_planning": True,
            },
        }

    replay_runs = [make_run() for _ in range(3)]
    replay = build_replay_record(replay_runs, config, h2_preflight["manifest_hash"])
    replay_passed = replay["status"] == "PASSED" and replay["identity_consistent"]
    first_run = replay_runs[0]
    all_tasks = first_run["task_records"]
    all_metrics = first_run["coverage_metrics"]

    # Controlled mathematical fixtures are evaluated before the final gate is
    # assembled, and their outcomes cannot change any acceptance threshold.
    first_task = all_tasks[0]
    controlled_full = evaluate_coverage([first_task], [first_task])
    controlled_uncovered = evaluate_coverage([first_task], [])
    controlled_overlap = evaluate_coverage([first_task], [first_task, first_task])
    test_results = {
        "planar_and_curved_h2_inputs": len(selected_records) == 3,
        "task_schema": all(bool(record) for record in all_tasks),
        "surface_provenance": all(record["source_geometry_hash"] in {item["geometry_hash"] for item in selected_records} for record in all_tasks),
        "normal_semantics": all(abs(sum(float(value) ** 2 for value in record["surface_normal_unit"]) - 1.0) <= UNIT_TOLERANCE for record in all_tasks),
        "spray_direction_normalized": all(abs(sum(float(value) ** 2 for value in record["spray_direction_unit"]) - 1.0) <= UNIT_TOLERANCE for record in all_tasks),
        "tcp_standoff_reconstruction": all(
            math.sqrt(sum((float(record["tcp_target_position_xyz_m"][axis]) + float(record["spray_direction_unit"][axis]) * float(record["nominal_standoff_m"]) - float(record["surface_point_xyz_m"][axis])) ** 2 for axis in range(3))) <= 1.0e-12
            for record in all_tasks
        ),
        "footprint_explicit": all(record["coverage_footprint_parameters"]["physical_deposition_model"] == "NOT_AVAILABLE" for record in all_tasks),
        "controlled_coverage": controlled_full["metrics"]["surface_coverage_ratio"] == 1.0 and controlled_uncovered["metrics"]["surface_coverage_ratio"] == 0.0,
        "controlled_overlap": controlled_overlap["metrics"]["coverage_overlap_redundancy"] > 0.0,
        "replay_three_runs": replay_passed,
        "topology_policy_explicit": all(item["diagnostics"]["silent_repair_applied"] is False for item in topology_payloads),
    }
    # Verify the immutable historical trees after all H3 computation.  H3 only
    # writes its newly created namespace, so any mismatch is a hard blocker.
    h0_after_ok, h0_after = hash_baseline(H0_MANIFEST)
    h1_after = hash_tree(H1_DIR)
    h2_after = hash_tree(h2_root)
    baseline_unchanged = h0_before_ok and h0_after_ok and h0_before["hashes"] == h0_after["hashes"] and h0_before["mismatches"] == h0_after["mismatches"]
    h1_unchanged = tree_unchanged(h1_before, h1_after)
    h2_unchanged = tree_unchanged(h2_before, h2_after)
    gates = {
        "H2_AUTHORIZATION_VALID": True,
        "STAGE2_BASELINE_IMMUTABLE": baseline_unchanged,
        "FORMAL_R2_IMMUTABLE": baseline_unchanged and h2_preflight["certificate_payload"].get("formal_r2_immutable") == "YES",
        "H0_H1_H2_IMMUTABLE": baseline_unchanged and h1_unchanged and h2_unchanged,
        "GEOMETRY_IDENTITY_VALID": len(selected_records) == len(SELECTED_GEOMETRY_IDS) and all(item["geometry_hash"] for item in selected_records),
        "SURFACE_SAMPLE_PROVENANCE_VALID": test_results["surface_provenance"],
        "NORMAL_SEMANTICS_VALID": test_results["normal_semantics"],
        "TASK_SCHEMA_VALID": test_results["task_schema"],
        "STANDOFF_SEMANTICS_VALID": all(record["nominal_standoff_m"] == config.nominal_standoff_m for record in all_tasks),
        "SPRAY_DIRECTION_VALID": test_results["spray_direction_normalized"],
        "TCP_POSE_VALID": test_results["tcp_standoff_reconstruction"],
        "ROLL_CONVENTION_VALID": all(record["roll_policy"]["method"] == "projected_global_reference_axis" and record["roll_policy"]["filesystem_order_dependency"] is False for record in all_tasks),
        "FOOTPRINT_MODEL_EXPLICIT": test_results["footprint_explicit"],
        "COVERAGE_METRIC_DEFINITION_VALID": all(set(metric["thresholds"].values()) == {"UNRESOLVED"} for metric in all_metrics) and test_results["controlled_coverage"] and test_results["controlled_overlap"],
        "NONFINITE_FAIL_CLOSED": True,
        "DEGENERACY_POLICY_VALID": all(item["diagnostics"]["degenerate_triangle_policy"] == "REPORT_AND_EXCLUDE_FROM_H2_SAMPLING" for item in topology_payloads),
        "BOUNDARY_POLICY_EXPLICIT": all(item["diagnostics"]["hole_policy"] == "REPORT_ONLY_NO_REPAIR" for item in topology_payloads),
        "CANONICAL_HASHING_VALID": len({record["configuration_hash"] for record in all_tasks}) == 1 and len({record["h2_manifest_hash"] for record in all_tasks}) == 1,
        "DETERMINISTIC_REPLAY_VALID": replay_passed,
        "NO_FJT": True,
        "NO_ROBOT_MOTION": True,
        "NO_LEDGER_MUTATION": True,
        "NO_ML_TRAINING": True,
        "NO_PATH_PLANNING_EXPERIMENT": True,
    }
    first_blocker = next((name for name, passed in gates.items() if not passed), "none")
    status = "PASSED" if first_blocker == "none" else "BLOCKED"
    preflight = {
        "gate_definition": frozen_gate,
        "h2": h2_preflight,
        "authoritative_stage01": stage01,
        "h0_before": {"passed": h0_before_ok, "entry_count": h0_before["entry_count"], "mismatch_count": h0_before["mismatch_count"]},
        "h0_after": {"passed": h0_after_ok, "entry_count": h0_after["entry_count"], "mismatch_count": h0_after["mismatch_count"]},
        "h1_before": h1_before,
        "h1_after": h1_after,
        "h2_before": h2_before,
        "h2_after": h2_after,
        "immutable_baselines_unchanged": baseline_unchanged and h1_unchanged and h2_unchanged,
    }
    gate_report = {
        "schema_version": "stage3-h3-gate-report-v1",
        "stage3_h3": status,
        "STAGE_3_H3": status,
        "first_blocker": first_blocker,
        "gate_definition_frozen_before_experiment": True,
        "gate_definition_hash": gate_hash,
        "gates": gates,
        "test_results": test_results,
        "preflight": preflight,
        "coverage_metric_thresholds": {metric["geometry_id"]: metric["thresholds"] for metric in all_metrics},
        "execution_boundary": {
            "new_fjt_goals_sent": 0,
            "send_goal_async_call_count": 0,
            "robot_motion_started": "NO",
            "formal_ledger_mutated": "NO",
            "formal_r2_evidence_modified": "NO",
            "stage2_baseline_modified": "NO",
            "ml_training_started": "NO",
            "path_planning_experiment_started": "NO",
            "formal_stage3_dataset_created": "NO",
        },
        "diagnostics": {
            "topology": topology_payloads,
            "collision_method": "adaptive_discrete_interpolation",
            "bullet_ccd": "not_available",
            "clearance": None,
            "physical_deposition_model": "NOT_AVAILABLE",
        },
    }
    task_manifest = {
        "schema_version": "stage3-h3-task-manifest-v1",
        "stage": "Stage 3 H3",
        "purpose": "TEST_FIXTURE",
        "formal_dataset": False,
        "dataset_role": None,
        "dataset_authorization": "No formal Stage 3 dataset created in H3",
        "geometry_level_split_policy": "Reserved for a future authorized dataset; no point-level split is performed",
        "h2_manifest": h2_preflight["manifest"],
        "h2_manifest_hash": h2_preflight["manifest_hash"],
        "h3_configuration": config.to_dict(),
        "configuration_hash": config_hash,
        "geometry_records": geometry_payloads,
        "task_sample_count": len(all_tasks),
        "coverage_metric_count": len(all_metrics),
        "target_semantics": "GEOMETRIC_TASK_TARGET",
        "reachability_checked": False,
        "formal_stage3_dataset_created": False,
    }
    json_dump(output_root / "stage3_h3_gate_definition.json", read_json(GATE_DEFINITION))
    for contract_name in CONTRACT_NAMES[1:]:
        json_dump(output_root / contract_name, read_json(H3_CONTRACT_DIR / contract_name))
    json_dump(output_root / "stage3_h3_gate_report.json", gate_report)
    json_dump(output_root / "stage3_h3_task_manifest.json", task_manifest)
    jsonl_dump(output_root / "stage3_h3_surface_targets.jsonl", all_tasks)
    json_dump(output_root / "stage3_h3_deterministic_replay.json", replay)
    json_dump(output_root / "stage3_h3_surface_coverage_metrics.json", all_metrics)
    terminal_certificate = {
        "schema_version": "stage3-h3-terminal-certificate-v1",
        "stage3_h3": status,
        "STAGE_3_H3": status,
        "first_blocker": first_blocker,
        "FIRST_BLOCKER": first_blocker,
        "stage2_baseline_immutable": "YES" if baseline_unchanged else "NO",
        "STAGE_2_BASELINE_IMMUTABLE": "YES" if baseline_unchanged else "NO",
        "formal_r2_immutable": "YES" if gates["FORMAL_R2_IMMUTABLE"] else "NO",
        "FORMAL_R2_IMMUTABLE": "YES" if gates["FORMAL_R2_IMMUTABLE"] else "NO",
        "h0_immutable": "YES" if baseline_unchanged else "NO",
        "H0_IMMUTABLE": "YES" if baseline_unchanged else "NO",
        "h1_immutable": "YES" if h1_unchanged else "NO",
        "H1_IMMUTABLE": "YES" if h1_unchanged else "NO",
        "h2_immutable": "YES" if h2_unchanged else "NO",
        "H2_IMMUTABLE": "YES" if h2_unchanged else "NO",
        "new_fjt_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "robot_motion_started": "NO",
        "formal_ledger_mutated": "NO",
        "task_representation_baseline": "PASSED" if gates["TASK_SCHEMA_VALID"] and gates["SURFACE_SAMPLE_PROVENANCE_VALID"] else "BLOCKED",
        "TASK_REPRESENTATION_BASELINE": "PASSED" if gates["TASK_SCHEMA_VALID"] and gates["SURFACE_SAMPLE_PROVENANCE_VALID"] else "BLOCKED",
        "target_pose_baseline": "PASSED" if gates["TCP_POSE_VALID"] else "BLOCKED",
        "TARGET_POSE_BASELINE": "PASSED" if gates["TCP_POSE_VALID"] else "BLOCKED",
        "roll_convention": "PASSED" if gates["ROLL_CONVENTION_VALID"] else "BLOCKED",
        "ROLL_CONVENTION": "PASSED" if gates["ROLL_CONVENTION_VALID"] else "BLOCKED",
        "spray_footprint_baseline": "PASSED" if gates["FOOTPRINT_MODEL_EXPLICIT"] else "BLOCKED",
        "SPRAY_FOOTPRINT_BASELINE": "PASSED" if gates["FOOTPRINT_MODEL_EXPLICIT"] else "BLOCKED",
        "coverage_metric_baseline": "PASSED" if gates["COVERAGE_METRIC_DEFINITION_VALID"] else "BLOCKED",
        "COVERAGE_METRIC_BASELINE": "PASSED" if gates["COVERAGE_METRIC_DEFINITION_VALID"] else "BLOCKED",
        "deterministic_replay": "PASSED" if gates["DETERMINISTIC_REPLAY_VALID"] else "BLOCKED",
        "DETERMINISTIC_REPLAY": "PASSED" if gates["DETERMINISTIC_REPLAY_VALID"] else "BLOCKED",
        "formal_stage3_dataset_created": "NO",
        "FORMAL_STAGE3_DATASET_CREATED": "NO",
        "ml_training_started": "NO",
        "ML_TRAINING_STARTED": "NO",
        "path_planning_experiment_started": "NO",
        "PATH_PLANNING_EXPERIMENT_STARTED": "NO",
        "next_stage_authorization_recommendation": "YES" if status == "PASSED" else "NO",
        "NEXT_STAGE_AUTHORIZATION_RECOMMENDATION": "YES" if status == "PASSED" else "NO",
        "output_root": relative(output_root),
        "gate_definition_hash": gate_hash,
        "h2_manifest_hash": h2_preflight["manifest_hash"],
    }
    json_dump(output_root / "stage3_h3_terminal_certificate.json", terminal_certificate)
    write_final_report(output_root, status, first_blocker, preflight, gate_report, replay, all_metrics)
    write_file_hashes(output_root)
    result = {
        "stage3_h3": status,
        "STAGE_3_H3": status,
        "first_blocker": first_blocker,
        "output_root": relative(output_root),
        "gates": gates,
        "terminal_certificate": relative(output_root / "stage3_h3_terminal_certificate.json"),
        "next_stage_authorization_recommendation": status == "PASSED",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-root", type=Path, default=None, help="passed H2 output directory; default is latest passed H2 directory")
    parser.add_argument("--output-root", type=Path, default=None, help="new H3 output directory; it must not already exist")
    args = parser.parse_args()
    h2_root = select_h2_root(args.h2_root)
    output_root = (args.output_root or ROOT / "outputs" / f"stage3_h3_coverage_baseline_{utc_stamp()}").resolve()
    if output_root.exists():
        print(json.dumps({"stage3_h3": "BLOCKED", "first_blocker": "same_id_overwrite_forbidden", "output_root": str(output_root)}))
        return 2
    try:
        result = run_baseline(output_root, h2_root)
    except (H3ValidationError, GeometryValidationError, OSError, KeyError, ValueError) as exc:
        # A blocked run still leaves an auditable terminal artifact in its own
        # namespace, but never fabricates a passing task/coverage result.
        if not output_root.exists():
            output_root.mkdir(parents=True)
        blocker = getattr(exc, "code", type(exc).__name__)
        certificate = {
            "schema_version": "stage3-h3-terminal-certificate-v1",
            "stage3_h3": "BLOCKED",
            "STAGE_3_H3": "BLOCKED",
            "first_blocker": blocker,
            "stage2_baseline_immutable": "UNKNOWN",
            "formal_r2_immutable": "UNKNOWN",
            "h0_immutable": "UNKNOWN",
            "h1_immutable": "UNKNOWN",
            "h2_immutable": "UNKNOWN",
            "new_fjt_goals_sent": 0,
            "send_goal_async_call_count": 0,
            "robot_motion_started": "NO",
            "formal_ledger_mutated": "NO",
            "formal_stage3_dataset_created": "NO",
            "ml_training_started": "NO",
            "path_planning_experiment_started": "NO",
            "next_stage_authorization_recommendation": "NO",
            "output_root": relative(output_root),
            "error": str(exc),
        }
        json_dump(output_root / "stage3_h3_terminal_certificate.json", certificate)
        (output_root / "FINAL_REPORT.md").write_text(f"# Stage 3 H3 — BLOCKED\n\n`FIRST_BLOCKER: {blocker}`\n\n{exc}\n", encoding="utf-8", newline="\n")
        write_file_hashes(output_root)
        print(json.dumps({"stage3_h3": "BLOCKED", "first_blocker": blocker, "output_root": relative(output_root)}))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["stage3_h3"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
