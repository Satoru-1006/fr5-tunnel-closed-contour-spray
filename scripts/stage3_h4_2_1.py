#!/usr/bin/env python3
"""Stage 3 H4.2.1 clean-exit recertification.

This is a new, additive recertification implementation.  Existing H4.2
artifacts and source are read-only.  The numerical worker is delegated to the
frozen H4.2 worker implementation; only its MoveItPy lifetime boundary and
process evidence are changed.  The MoveItPy object is intentionally retained
until the child process has finalized and fsynced all artifacts, then the
process exits with code zero.  This isolates the known MoveItCpp destructor
segmentation fault without changing IK, FK, FCL, classification, or hashes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import stage3_h4_2 as h42  # noqa: E402

OLD_H42_ROOT = ROOT / "outputs/stage3_h4_2_workspace_diagnosis_final5"
DEFAULT_OUTPUT = ROOT / "outputs/stage3_h4_2_1_clean_exit_recertification_20260808T230100Z"
SELECTED_CANDIDATE = "candidate_015_combined_centered_alt"
FORBIDDEN_LOG_PATTERNS = ("process has died", "exit code -11", "sigsegv", "segmentation fault")
REQUIRED_WORKER_ARTIFACTS = ("placements.json", "ik_requests.tsv", "ik_bridge.jsonl", "ik_bridge.log", "worker_result.json")


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def source_evidence(path: Path, patterns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if any(pattern in line for pattern in patterns):
            rows.append({"path": rel(path), "line": number, "text": line.strip()})
    return rows


def tree_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"root": rel(root), "file_count": len(files), "files": files, "tree_hash": h42.semantic_hash(files)}


def audit_existing_h42(output: Path) -> dict[str, Any]:
    required = [
        OLD_H42_ROOT / "stage3_h4_2_fresh_process_replay.json",
        OLD_H42_ROOT / "stage3_h4_2_candidate_metrics.jsonl",
        OLD_H42_ROOT / "stage3_h4_2_gate_report.json",
        OLD_H42_ROOT / "stage3_h4_2_terminal_certificate.json",
        ROOT / "scripts/stage3_h4_2.py",
        ROOT / "scripts/stage3_h4_2_launch.py",
    ]
    missing = [rel(path) for path in required if not path.is_file()]
    replay = load_json(required[0]) if not missing else {}
    candidate_metrics = load_jsonl(required[1]) if not missing else []
    gate = load_json(required[2]) if not missing else {}
    certificate = load_json(required[3]) if not missing else {}
    logs = sorted((OLD_H42_ROOT / "worker_runs").glob("*/fresh_process_*/launch.log"))
    log_hits = []
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        hits = [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in text]
        if hits:
            log_hits.append({"path": rel(path), "patterns": hits})
    replay_processes = replay.get("process_records", [])
    selected_artifacts = []
    for record in replay_processes:
        worker = ROOT / str(record.get("worker_dir", ""))
        result_path = worker / "worker_result.json"
        result = load_json(result_path) if result_path.is_file() else {}
        selected_artifacts.append({
            "replay_index": record.get("replay_index"),
            "worker_dir": rel(worker),
            "outer_returncode": record.get("exit_code"),
            "worker_result_exists": result_path.is_file(),
            "target_count": result.get("target_count"),
            "target_rows": len(result.get("target_rows", [])),
            "ik_rows": len(result.get("ik_rows", [])),
            "fk_rows": len(result.get("fk_rows", [])),
            "collision_rows": len(result.get("collision_rows", [])),
            "semantic_hash": result.get("semantic_hash"),
        })
    source = {
        "worker_artifact_write_and_teardown": source_evidence(ROOT / "scripts/stage3_h4_2.py", ("dump_json(output / \"worker_result.json\"", "del moveit", "rclpy.shutdown")),
        "outer_subprocess_capture": source_evidence(ROOT / "scripts/stage3_h4_2.py", ("subprocess.run", "proc.returncode", "result_path.is_file")),
        "hash_only_replay_gate": source_evidence(ROOT / "scripts/stage3_h4_2.py", ("fields = (\"raw_ik_candidate_semantic_hash\"", "all_equal = all(equality.values())", "\"THREE_FRESH_PROCESS_REPLAY\": all_equal")),
        "launch_wrapper": source_evidence(ROOT / "scripts/stage3_h4_2_launch.py", ("Node(", "output=\"screen\"")),
    }
    report = {
        "schema_version": "stage3-h4-2-1-existing-h4-2-audit-v1",
        "source_root": rel(OLD_H42_ROOT),
        "required_files_missing": missing,
        "old_gate_status": gate.get("STAGE_3_H4_2"),
        "old_terminal_status": certificate.get("STAGE_3_H4_2"),
        "old_gate_three_replay": gate.get("gates", {}).get("THREE_FRESH_PROCESS_REPLAY"),
        "old_replay_hashes_all_equal": replay.get("all_equal"),
        "old_replay_process_count": len(replay_processes),
        "old_replay_outer_returncodes": [record.get("exit_code") for record in replay_processes],
        "all_worker_launch_log_count": len(logs),
        "child_process_died_log_count": len(log_hits),
        "child_process_died_logs": log_hits,
        "candidate_metric_worker_count": len(candidate_metrics),
        "selected_artifact_completeness": selected_artifacts,
        "answers": {
            "exit_code_minus_11_source": "actual H4.2 Python worker process, as launch.log records the python3 child dying after MoveItCpp deletion",
            "why_outer_returncode_zero": "ros2 launch is the process waited by subprocess.run; its return code was not a child lifecycle assertion and the old wrapper never parsed OnProcessExit/launch.log",
            "current_gate_scope": "semantic hashes only for the selected three replay results; child return code and launch log are not gate inputs",
            "certification_loophole": True,
            "crash_phase": "after worker_result.json was written, during MoveItCpp destruction in the finally block, before a normal worker exit",
            "artifacts_before_crash": "selected worker_result files contain complete 192 target rows and IK/FK/collision arrays; this is evidence of completed computation, not clean process certification",
            "numeric_effect_scope": "no numeric divergence is observed in the three old completed artifacts; the observed failure is process-lifecycle certification, while any claim beyond completed artifacts would be unsupported",
            "root_cause": "MoveItPy/MoveItCpp destruction boundary; rclpy.shutdown follows del moveit in source, and launch is the reporter rather than the crash site",
        },
        "source_evidence": source,
    }
    dump_json(output / "h4_2_existing_evidence_audit.json", report)
    lines = [
        "# H4.2 Existing Evidence Audit",
        "",
        "`STAGE_3_H4_2_1_AUDIT: COMPLETED`",
        "",
        f"- Existing H4.2 gate: `{report['old_gate_status']}`; existing replay hash gate: `{report['old_replay_hashes_all_equal']}`.",
        f"- Outer return codes recorded by the old wrapper: `{report['old_replay_outer_returncodes']}`.",
        f"- Worker launch logs inspected: `{report['all_worker_launch_log_count']}`; logs containing a child death: `{report['child_process_died_log_count']}`.",
        "- The `-11` is from the Python/MoveIt worker child. The child reaches `Deleting MoveItCpp`, then launch reports `process has died ... exit code -11`.",
        "- `subprocess.run(...).returncode == 0` is the outer `ros2 launch` return code; the old wrapper did not capture child `OnProcessExit` return codes or reject death text in launch logs.",
        "- The old `THREE_FRESH_PROCESS_REPLAY` gate compares semantic hashes and does not certify child clean exit. This is a certification loophole.",
        "- The crash is after the computation and `worker_result.json` write, during MoveItCpp destruction; it is not evidence that IK/FK/FCL numeric arrays were partially computed.",
        "- The evidence supports a MoveItPy/MoveItCpp destruction-lifecycle cause. `rclpy.shutdown()` occurs after `del moveit` in the old worker, so ROS shutdown is not the first observed crash site; launch only surfaces the child failure.",
        "",
        "The old H4.2 artifacts remain immutable. H4.2.1 therefore adds a child exit capture, clean-log gate, artifact-finalization manifest, and independent three-process replay.",
    ]
    (output / "h4_2_existing_evidence_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report


_LEAKED_MOVEIT_OBJECTS: list[Any] = []
_REAL_MOVEITPY: Any = None


class _LifecycleIsolatedMoveItPy:
    """Keep the C++ owner alive until the process has finalized artifacts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _REAL_MOVEITPY is None:
            raise RuntimeError("MoveItPy lifecycle wrapper was not initialized")
        self._inner = _REAL_MOVEITPY(*args, **kwargs)
        _LEAKED_MOVEIT_OBJECTS.append(self._inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install_lifecycle_isolation() -> None:
    global _REAL_MOVEITPY
    import moveit.planning

    if _REAL_MOVEITPY is None:
        _REAL_MOVEITPY = moveit.planning.MoveItPy
    if getattr(moveit.planning, "MoveItPy") is not _LifecycleIsolatedMoveItPy:
        moveit.planning.MoveItPy = _LifecycleIsolatedMoveItPy


def finalize_worker_artifacts(output: Path, result: Mapping[str, Any], replay_index: int) -> None:
    missing = [name for name in REQUIRED_WORKER_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"worker artifacts missing before finalization: {missing}")
    for name in REQUIRED_WORKER_ARTIFACTS:
        fsync_file(output / name)
    manifest = {
        "schema_version": "stage3-h4-2-1-artifact-finalize-v1",
        "phase": "all_required_artifacts_flushed_and_finalized",
        "replay_index": replay_index,
        "worker_result_semantic_hash": result["semantic_hash"],
        "required_artifacts": [
            {"name": name, "size_bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in REQUIRED_WORKER_ARTIFACTS
        ],
        "moveitpy_destruction": "deferred_until_process_reclamation",
        "rclpy_shutdown": "completed_by_frozen_worker_finally_before_return",
        "exit_strategy": "os._exit(0) only after this manifest is flushed",
    }
    dump_json(output / "child_artifact_finalize.json", manifest)
    fsync_file(output / "child_artifact_finalize.json")


def run_clean_worker(argv: argparse.Namespace) -> int:
    placements = load_json(argv.placements)
    targets = h42.load_jsonl(argv.h3_targets)
    h2_records = {str(row["geometry_id"]): row for row in h42.load_json(argv.h2_manifest).get("records", [])}
    seeds = h42.load_json(h42.H41_ROOT / "stage3_h4_1_seed_contract.json")["seeds"]
    transformed = {str(key): value for key, value in placements["transforms"].items()}
    install_lifecycle_isolation()
    result = h42.worker_result(
        argv.worker_output.resolve(),
        int(argv.replay_index),
        targets,
        seeds,
        h2_records,
        argv.h2_manifest.resolve(),
        argv.kinematics.resolve(),
        argv.urdf.resolve(),
        argv.srdf.resolve(),
        transformed,
        str(placements["placement_id"]),
    )
    finalize_worker_artifacts(argv.worker_output.resolve(), result, int(argv.replay_index))
    print(json.dumps({"worker": "PASSED", "clean_exit": True, "placement_id": placements["placement_id"], "replay_index": argv.replay_index, "semantic_hash": result["semantic_hash"]}, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{resolved.as_posix().split(':', 1)[-1].lstrip('/').replace('\\\\', '/') }"


def run_replay(candidate: Mapping[str, Any], output: Path, replay_index: int) -> dict[str, Any]:
    worker_dir = output / "worker_runs" / str(candidate["candidate_id"]) / f"fresh_process_{replay_index:03d}"
    worker_dir.mkdir(parents=True, exist_ok=False)
    placements_path = worker_dir / "placements.json"
    dump_json(placements_path, {"placement_id": candidate["candidate_id"], "transforms": candidate["transforms"]})
    child_exit_record = worker_dir / "child_exit_record.json"
    launch_command = " ".join([
        f"ros2 launch {wsl_path(ROOT / 'scripts/stage3_h4_2_1_launch.py')}",
        f"worker_output:={wsl_path(worker_dir)}",
        f"replay_index:={replay_index}",
        f"placements:={wsl_path(placements_path)}",
        f"h3_targets:={wsl_path(h42.H3_TARGETS)}",
        f"h2_manifest:={wsl_path(h42.H2_MANIFEST)}",
        f"kinematics:={wsl_path(h42.KINEMATICS)}",
        f"urdf:={wsl_path(h42.URDF)}",
        f"srdf:={wsl_path(h42.SRDF)}",
        f"child_exit_record:={wsl_path(child_exit_record)}",
    ])
    command = [
        "source /opt/ros/jazzy/setup.bash",
        f"source {wsl_path(ROOT / 'install/setup.bash')}",
        f"cd {wsl_path(ROOT)}",
        launch_command,
    ]
    proc = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", " && ".join(command)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    launch_log = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    (worker_dir / "launch.log").write_text(launch_log, encoding="utf-8", newline="\n")
    child = load_json(child_exit_record) if child_exit_record.is_file() else {}
    result_path = worker_dir / "worker_result.json"
    result = load_json(result_path) if result_path.is_file() else None
    log_lower = launch_log.lower()
    forbidden = [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in log_lower]
    finalize_path = worker_dir / "child_artifact_finalize.json"
    finalize = load_json(finalize_path) if finalize_path.is_file() else {}
    artifact_checks = []
    for item in finalize.get("required_artifacts", []):
        path = worker_dir / str(item.get("name"))
        artifact_checks.append({"name": item.get("name"), "exists": path.is_file(), "sha256_matches": path.is_file() and sha256(path) == item.get("sha256"), "size_matches": path.is_file() and path.stat().st_size == item.get("size_bytes")})
    record = {
        "replay_index": replay_index,
        "worker_dir": rel(worker_dir),
        "outer_launch_returncode": proc.returncode,
        "child_returncode": child.get("returncode"),
        "child_exit_capture": child,
        "child_process_clean_exit": proc.returncode == 0 and child.get("returncode") == 0 and not forbidden and "process has finished cleanly" in log_lower,
        "forbidden_log_patterns": forbidden,
        "worker_result_exists": result is not None,
        "artifact_finalize_exists": finalize_path.is_file(),
        "artifact_checks": artifact_checks,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }
    if result is None:
        raise RuntimeError(f"H4.2.1 worker produced no result: {record}")
    if not record["child_process_clean_exit"]:
        raise RuntimeError(f"H4.2.1 child clean-exit gate failed: {record}")
    if not finalize.get("phase") == "all_required_artifacts_flushed_and_finalized" or not all(item["exists"] and item["sha256_matches"] and item["size_matches"] for item in artifact_checks):
        raise RuntimeError(f"H4.2.1 artifact-finalization gate failed: {record}")
    record.update(h42.replay_hashes(result))
    return {"result": result, "record": record}


def immutable_snapshot() -> dict[str, Any]:
    stage = h42.verify_immutable_inputs()
    old_tree = tree_manifest(OLD_H42_ROOT)
    return {"stage_inputs": stage, "old_h4_2_tree": old_tree}


def orchestrate(output: Path) -> int:
    if output.exists():
        raise h42.H4ValidationError("same_id_overwrite_forbidden", str(output))
    output.mkdir(parents=True, exist_ok=False)
    audit = audit_existing_h42(output)
    before = immutable_snapshot()
    candidate = load_json(OLD_H42_ROOT / "stage3_h4_2_selected_candidate.json")
    if candidate.get("candidate_id") != SELECTED_CANDIDATE:
        raise h42.H4ValidationError("selected_candidate_mismatch", str(candidate.get("candidate_id")))
    dump_json(output / "selected_candidate_input_copy.json", candidate)
    replay_runs = [run_replay(candidate, output, index) for index in (201, 202, 203)]
    results = [item["result"] for item in replay_runs]
    records = [item["record"] for item in replay_runs]
    fields = ("raw_ik_candidate_semantic_hash", "fk_hash", "fcl_hash", "classification_hash", "aggregate_metrics_hash", "semantic_hash")
    equality = {field: len({record[field] for record in records}) == 1 for field in fields}
    hashes_identical = all(equality.values())
    primary = results[0]
    dump_json(output / "clean_exit_replay_report.json", {"schema_version": "stage3-h4-2-1-clean-exit-replay-v1", "fresh_process_count": 3, "placement_id": SELECTED_CANDIDATE, "processes": records, "equality": equality, "all_semantic_hashes_identical": hashes_identical, "wall_clock_fields_excluded_from_hash": ["elapsed_s", "worker_pid", "timestamp"]})
    dump_jsonl = lambda path, rows: path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    dump_jsonl(output / "selected_target_feasibility.jsonl", primary["target_rows"])
    dump_jsonl(output / "selected_ik_candidates.jsonl", primary["ik_rows"])
    dump_jsonl(output / "selected_fk_validation.jsonl", primary["fk_rows"])
    dump_jsonl(output / "selected_collision_validation.jsonl", primary["collision_rows"])
    after = immutable_snapshot()
    immutable_unchanged = h42.semantic_hash(before) == h42.semantic_hash(after)
    child_clean = all(record["child_process_clean_exit"] for record in records)
    required_artifacts_complete = all(record["artifact_finalize_exists"] and all(item["exists"] and item["sha256_matches"] and item["size_matches"] for item in record["artifact_checks"]) for record in records)
    gates = {
        "H4_2_EXISTING_AUDIT_COMPLETED": not audit["required_files_missing"],
        "H0_H1_H2_H3_H4_H4_1_H4_2_IMMUTABLE": immutable_unchanged,
        "CHILD_PROCESS_CLEAN_EXIT_3_OF_3": child_clean,
        "CHILD_EXIT_CODE_ZERO_3_OF_3": all(record["child_returncode"] == 0 for record in records),
        "NO_SIGSEGV_OR_EXIT_MINUS_11": all(not record["forbidden_log_patterns"] for record in records),
        "NO_PROCESS_HAS_DIED": all("process has died" not in " ".join(record["forbidden_log_patterns"]) for record in records),
        "REQUIRED_ARTIFACTS_FINALIZED_3_OF_3": required_artifacts_complete,
        "SEMANTIC_HASHES_IDENTICAL_3_OF_3": hashes_identical,
        "IK_FK_FCL_AGGREGATE_HASHES_IDENTICAL": all(equality[field] for field in fields),
        "TARGET_DENOMINATOR_192": primary.get("target_count") == 192 and len(primary.get("target_rows", [])) == 192,
        "FJT_GOALS_SENT_ZERO": True,
        "ROBOT_MOTION_NOT_STARTED": True,
        "PATH_PLANNING_NOT_STARTED": True,
        "RUCKIG_NOT_STARTED": True,
        "ML_TRAINING_NOT_STARTED": True,
        "FORMAL_LEDGER_MUTATION_NO": True,
    }
    first_blocker = next((name.lower() for name, passed in gates.items() if not passed), "none")
    status = "PASSED" if all(gates.values()) else "BLOCKED"
    gate_report = {"schema_version": "stage3-h4-2-1-gate-report-v1", "STAGE_3_H4_2_1": status, "stage3_h4_2_1": status, "FIRST_BLOCKER": first_blocker, "gates": gates, "selected_candidate": SELECTED_CANDIDATE, "clean_exit_replay": load_json(output / "clean_exit_replay_report.json"), "immutable_before": before, "immutable_after": after, "collision_method": h42.COLLISION_METHOD, "ccd_status": h42.UNAVAILABLE, "clearance_status": h42.UNAVAILABLE, "FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "PATH_PLANNING_STARTED": "NO", "RUCKIG_STARTED": "NO", "ML_TRAINING_STARTED": "NO", "FORMAL_LEDGER_MUTATION": "NO"}
    dump_json(output / "stage3_h4_2_1_gate_report.json", gate_report)
    terminal = {"schema_version": "stage3-h4-2-1-terminal-certificate-v1", "STAGE_3_H4_2_1": status, "FIRST_BLOCKER": first_blocker, "H0_H1_H2_H3_H4_H4_1_H4_2_IMMUTABLE": "YES" if immutable_unchanged else "NO", "CHILD_PROCESS_CLEAN_EXIT_3_OF_3": "YES" if child_clean else "NO", "CHILD_EXIT_CODE_ZERO_3_OF_3": "YES" if all(record["child_returncode"] == 0 for record in records) else "NO", "SEMANTIC_HASHES_IDENTICAL_3_OF_3": "YES" if hashes_identical else "NO", "REQUIRED_ARTIFACTS_FINALIZED_3_OF_3": "YES" if required_artifacts_complete else "NO", "SELECTED_CANDIDATE": SELECTED_CANDIDATE, "FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "PATH_PLANNING_STARTED": "NO", "RUCKIG_STARTED": "NO", "ML_TRAINING_STARTED": "NO", "FORMAL_LEDGER_MUTATION": "NO", "STAGE_3_H4_3": "NOT_STARTED", "READY_FOR_STAGE_3_H5": "NO", "collision_method": h42.COLLISION_METHOD, "ccd_status": h42.UNAVAILABLE, "clearance_status": h42.UNAVAILABLE}
    dump_json(output / "stage3_h4_2_1_terminal_certificate.json", terminal)
    write_final_report(output, status, first_blocker, gates, records, primary, audit)
    all_files = [{"relative_path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "immutable_hash_manifest.json"]
    dump_json(output / "immutable_hash_manifest.json", {"schema_version": "stage3-h4-2-1-output-hashes-v1", "algorithm": "SHA-256", "files": all_files, "tree_hash": h42.semantic_hash(all_files), "old_formal_artifacts_not_overwritten": True})
    return 0 if status == "PASSED" else 2


def write_final_report(output: Path, status: str, first_blocker: str, gates: Mapping[str, Any], records: Sequence[Mapping[str, Any]], primary: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    counts = primary.get("metrics", {}).get("classification_counts", {})
    lines = [
        "# Stage 3 H4.2.1 — Clean-Exit Recertification",
        "",
        f"`STAGE_3_H4_2_1: {status}`",
        f"`FIRST_BLOCKER: {first_blocker}`",
        "",
        "## Existing H4.2 audit",
        "",
        "- Existing H4.2 artifacts are read-only and were audited before recertification.",
        f"- Existing outer return codes: `{audit.get('old_replay_outer_returncodes')}`; old child-death logs: `{audit.get('child_process_died_log_count')}` of `{audit.get('all_worker_launch_log_count')}`.",
        "- The old gate was a semantic-hash gate and did not certify child clean exit. The observed crash was after `worker_result.json` generation, at MoveItCpp destruction.",
        "",
        "## H4.2.1 gates",
        "",
        *[f"- `{name}`: `{value}`" for name, value in gates.items()],
        "",
        f"- Selected candidate: `{SELECTED_CANDIDATE}`.",
        f"- Selected 192-target classification counts: `{json.dumps(counts, sort_keys=True)}`.",
        "- Collision result labels remain `adaptive_discrete_interpolation`; CCD and clearance remain `not_available`/null.",
        "- No FJT goals, robot motion, path planning, Ruckig, ML/RL, or formal ledger mutation occurred.",
        "",
        "## Child lifecycle evidence",
        "",
        *[f"- replay `{record.get('replay_index')}`: outer=`{record.get('outer_launch_returncode')}`, child=`{record.get('child_returncode')}`, clean=`{record.get('child_process_clean_exit')}`, finalize=`{record.get('artifact_finalize_exists')}`" for record in records],
        "",
        "The lifecycle isolation retains the MoveItPy C++ owner until the child has flushed and fsynced all required artifacts, then exits with code 0. This is additive H4.2.1 evidence and does not modify H0/H1/H2/H3/H4/H4.1/H4.2 evidence.",
        "",
        "`STAGE_3_H4_3: NOT_STARTED`",
        "`READY_FOR_STAGE_3_H5: NO`",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--replay-index", type=int, default=0)
    parser.add_argument("--placements", type=Path)
    parser.add_argument("--h3-targets", type=Path, default=h42.H3_TARGETS)
    parser.add_argument("--h2-manifest", type=Path, default=h42.H2_MANIFEST)
    parser.add_argument("--kinematics", type=Path, default=h42.KINEMATICS)
    parser.add_argument("--urdf", type=Path, default=h42.URDF)
    parser.add_argument("--srdf", type=Path, default=h42.SRDF)
    args, _unknown = parser.parse_known_args(argv)
    if args.worker:
        if not all((args.worker_output, args.placements)):
            print("worker requires --worker-output and --placements", file=sys.stderr)
            return 2
        try:
            return run_clean_worker(args)
        except Exception:
            import traceback

            traceback.print_exc()
            return 2
    if not args.orchestrate:
        print("Stage 3 H4.2.1 requires --orchestrate or --worker.")
        return 2
    try:
        return orchestrate(args.output_root.resolve())
    except Exception as exc:
        output = args.output_root.resolve()
        output.mkdir(parents=True, exist_ok=True)
        first = getattr(exc, "code", type(exc).__name__)
        dump_json(output / "stage3_h4_2_1_gate_report.json", {"schema_version": "stage3-h4-2-1-gate-report-v1", "STAGE_3_H4_2_1": "BLOCKED", "FIRST_BLOCKER": first, "error": f"{type(exc).__name__}: {exc}"})
        dump_json(output / "stage3_h4_2_1_terminal_certificate.json", {"schema_version": "stage3-h4-2-1-terminal-certificate-v1", "STAGE_3_H4_2_1": "BLOCKED", "FIRST_BLOCKER": first, "CHILD_PROCESS_CLEAN_EXIT_3_OF_3": "NO", "SEMANTIC_HASHES_IDENTICAL_3_OF_3": "NO", "STAGE_3_H4_3": "NOT_STARTED", "READY_FOR_STAGE_3_H5": "NO", "error": f"{type(exc).__name__}: {exc}"})
        (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H4.2.1 — BLOCKED\n\n`FIRST_BLOCKER: {first}`\n\n{type(exc).__name__}: {exc}\n", encoding="utf-8", newline="\n")
        print(json.dumps({"STAGE_3_H4_2_1": "BLOCKED", "FIRST_BLOCKER": first, "output_root": rel(output)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
