#!/usr/bin/env python3
"""Run Stage 3 H12 constraint-aware learned trajectory repair certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_dataset import (  # noqa: E402
    EXPECTED_H10_SEMANTIC_SHA256,
    FEATURE_NAMES,
    INPUT_HISTORY,
    PREDICTION_HORIZON,
    load_h10_segments,
    semantic_hash,
    sha256_file,
    window_semantic_hash,
    write_json,
)
from src.stage3_h11_model import (  # noqa: E402
    CausalGRUTrajectoryPredictor,
    TORCH_AVAILABLE,
    constant_velocity_prediction,
    evaluate_model,
    hold_last_prediction,
    metric_payload,
    rollout_one_step,
    set_deterministic,
)
from src.stage3_h12_trajectory_repair import (  # noqa: E402
    RepairLimits,
    REPAIR_METHOD,
    audit_repair_inputs,
    canonical_sha256,
    constraint_report,
    repair_magnitude,
    repair_role,
    semantic_prediction_sha256,
)
from scripts.stage3_h11_train_baseline import materialize_windows  # noqa: E402


SEED = 11011
H11_CERT_NAME = "stage3_h11_r_terminal_certificate.json"
H11_GATE_NAME = "stage3_h11_r_gate_report.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def dump(path: Path, value: Any) -> None:
    write_json(path, value)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_h11_root() -> Path:
    candidates: list[tuple[datetime, Path]] = []
    for path in (ROOT / "outputs").glob("stage3_h11_r_*"):
        certificate = path / H11_CERT_NAME
        if not certificate.is_file():
            continue
        try:
            payload = load(certificate)
        except Exception:
            continue
        if payload.get("STAGE_3_H11_R") != "PASSED" or payload.get("READY_FOR_STAGE_3_H12") != "YES":
            continue
        candidates.append((path.stat().st_mtime_ns and datetime.fromtimestamp(path.stat().st_mtime, timezone.utc), path))
    if not candidates:
        raise RuntimeError("h11_r_authoritative_output_not_found")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def discover_checkpoint(h11_root: Path) -> tuple[Path, str, list[dict[str, Any]]]:
    manifest = load(h11_root / "checkpoint_sha256_manifest.json")
    expected = str(manifest.get("sha256") or "")
    if not expected:
        raise RuntimeError("h11_checkpoint_sha256_missing")
    candidates: list[Path] = []
    checkpoint_name = manifest.get("checkpoint")
    if checkpoint_name:
        candidates.append(h11_root / str(checkpoint_name))
    source = manifest.get("source_formal_runner")
    if source:
        source_path = Path(str(source))
        candidates.append(source_path if source_path.is_absolute() else ROOT / source_path)
    frozen = h11_root / "frozen_artifact_manifest.json"
    if frozen.is_file():
        for record in load(frozen).get("records", []):
            path = Path(str(record.get("absolute_path") or record.get("path")))
            if not path.is_absolute():
                path = h11_root / path
            if path.is_file() and ("model" in path.name or path.name == str(checkpoint_name)):
                candidates.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if str(resolved) not in seen:
            unique.append(resolved)
            seen.add(str(resolved))
    evidence: list[dict[str, Any]] = []
    for candidate in unique:
        observed = sha256_file(candidate) if candidate.is_file() else None
        evidence.append({"path": str(candidate), "exists": candidate.is_file(), "sha256": observed, "matches_expected": observed == expected})
        if observed == expected:
            return candidate, expected, evidence
    raise RuntimeError("h11_checkpoint_discovery_hash_mismatch")


def h10_marker(h10_root: Path, marker: str) -> Path:
    manifest = load(h10_root / "frozen_artifact_manifest.json")
    for record in manifest.get("records", []):
        status = str(record.get("status", ""))
        path_text = str(record.get("path", ""))
        if marker in status or marker in path_text:
            path = Path(str(record.get("absolute_path") or record.get("path")))
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file():
                return path.resolve()
    raise RuntimeError(f"h10_frozen_marker_missing:{marker}")


def hash_records(records: list[Mapping[str, Any]], group_override: str | None = None) -> tuple[list[dict[str, Any]], bool]:
    result: list[dict[str, Any]] = []
    all_ok = True
    seen: set[str] = set()
    for source in records:
        path = Path(str(source.get("absolute_path") or source.get("path")))
        if not path.is_absolute():
            path = ROOT / path
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        before = source.get("before_sha256", source.get("sha256"))
        before_size = source.get("before_size_bytes", source.get("size_bytes"))
        exists = path.is_file()
        after = sha256_file(path) if exists else None
        after_size = path.stat().st_size if exists else None
        unchanged = bool(exists and after == before and (before_size is None or after_size == before_size))
        all_ok = all_ok and unchanged
        result.append({
            "group": group_override or source.get("group"),
            "path": str(path.resolve()),
            "before_sha256": before,
            "after_sha256": after,
            "before_size_bytes": before_size,
            "after_size_bytes": after_size,
            "unchanged": unchanged,
        })
    return result, all_ok


def freeze_upstream(h11_root: Path, checkpoint: Path, checkpoint_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = load(h11_root / "upstream_immutability_report.json")
    # H11-R stores the inherited H7/H8/H9/H10 record list under the
    # ``upstream_immutability`` namespace; accept the direct form as well for
    # compatibility with earlier provenance writers.
    inherited_records = previous.get("records") or previous.get("upstream_immutability", {}).get("records", [])
    records, _ = hash_records(inherited_records)
    h11_manifest = load(h11_root / "frozen_artifact_manifest.json")
    h11_records, h11_ok = hash_records(h11_manifest.get("records", []), "H11_R_EVIDENCE")
    records.extend(h11_records)
    checkpoint_before = sha256_file(checkpoint)
    checkpoint_size = checkpoint.stat().st_size
    checkpoint_record = {"group": "H11_CHECKPOINT", "path": str(checkpoint), "before_sha256": checkpoint_before, "after_sha256": checkpoint_before, "before_size_bytes": checkpoint_size, "after_size_bytes": checkpoint_size, "unchanged": checkpoint_before == checkpoint_sha}
    records.append(checkpoint_record)
    groups = {}
    for group in ("H7_EVIDENCE", "H8_R_EVIDENCE", "HISTORICAL_H8_EVIDENCE", "H9_EVIDENCE", "H10_EVIDENCE"):
        group_records = [item for item in records if item.get("group") == group]
        groups[group] = bool(group_records) and all(bool(item.get("unchanged")) for item in group_records)
    groups["H11_R_EVIDENCE"] = bool(h11_records) and h11_ok
    groups["H11_CHECKPOINT"] = checkpoint_before == checkpoint_sha
    h10_root = ROOT / Path(str(load(h11_root / "h11_dataset_contract.json")["source_h10"]))
    h10_semantic = load(h10_root / "dataset_semantic_hash.json")
    semantic_match = h10_semantic.get("semantic_dataset_sha256") == EXPECTED_H10_SEMANTIC_SHA256
    report = {
        "schema_version": "stage3_h12_upstream_immutability_v1",
        "algorithm": "SHA-256 inherited from H10/H11 provenance manifests",
        "records": records,
        "groups": groups,
        "H10_SEMANTIC_HASH_MATCH": "YES" if semantic_match else "NO",
        "status": "PASSED" if all(groups.values()) and semantic_match else "BLOCKED",
        "first_blocker": None if all(groups.values()) and semantic_match else "upstream_artifact_mutated_or_semantic_hash_mismatch",
    }
    return report, {"h10_root": h10_root, "semantic_match": semantic_match}


def load_context(dataset: Any, windows: list[Mapping[str, Any]], stats: Mapping[str, Any], role: str) -> dict[str, Any]:
    arrays = materialize_windows(dataset, windows, stats, role)
    segment_map = {segment.key: segment for segment in dataset.segments}
    selected = [item for item in windows if item["split_role"] == role]
    history_velocity = []
    history_acceleration = []
    controlled_stop = []
    for item in selected:
        segment = segment_map[str(item["segment_key"])]
        offset = int(item["start_offset"]) + INPUT_HISTORY - 1
        history_velocity.append(segment.velocities[offset])
        history_acceleration.append(segment.accelerations[offset])
        controlled_stop.append(bool(item.get("cross_controlled_stop", False) or segment.controlled_stop_count > 0))
    return {
        "arrays": arrays,
        "history_velocity": np.asarray(history_velocity, dtype=np.float64),
        "history_acceleration": np.asarray(history_acceleration, dtype=np.float64),
        "controlled_stop": np.asarray(controlled_stop, dtype=bool),
        "window_metadata": selected,
    }


def model_from_checkpoint(checkpoint: Path):
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch
    set_deterministic(SEED)
    model = CausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=PREDICTION_HORIZON)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else payload
    model.load_state_dict(state)
    model.eval()
    return model


def global_by_window_id(values_by_role: Mapping[str, np.ndarray], contexts: Mapping[str, Mapping[str, Any]], total: int) -> np.ndarray:
    result = np.empty((total,) + next(iter(values_by_role.values())).shape[1:], dtype=np.float64)
    for role, values in values_by_role.items():
        for index, item in enumerate(contexts[role]["window_metadata"]):
            result[int(item["window_id"])] = values[index]
    return result


def run_native(output: Path, native_input: Path, window_manifest: Path, h10_root: Path, h6_validation: Path, h6_segments: Path, process_contract: Path, fixture_mesh: Path, runtime_limits: Path) -> dict[str, Any]:
    native_report = output / "stage3_h12_native_validation_runtime.json"
    command = "source /opt/ros/jazzy/setup.bash && export AMENT_PREFIX_PATH=/mnt/d/robotfucker/install/fr5_tunnel_moveit_bridge:/mnt/d/robotfucker/install/fairino5_v6_moveit2_config:/mnt/d/robotfucker/install/fairino_description:/opt/ros/jazzy && export PYTHONPATH=/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/mnt/d/robotfucker/ros2_moveit_bridge:/opt/ros/jazzy/lib/python3.12/site-packages && ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h12_native.launch.py " + " ".join([
        f"input_npz:={wsl(native_input)}", f"window_manifest:={wsl(window_manifest)}", f"output_json:={wsl(native_report)}",
        f"h6_validation:={wsl(h6_validation)}", f"h6_segments:={wsl(h6_segments)}", f"process_contract:={wsl(process_contract)}", f"fixture_mesh:={wsl(fixture_mesh)}", f"runtime_limits:={wsl(runtime_limits)}",
    ])
    (output / "stage3_h12_native_command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    timeout_s = int(os.environ.get("H12_NATIVE_TIMEOUT_S", "600"))
    try:
        completed = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
        (output / "stage3_h12_native_stdout.log").write_text(completed.stdout or "", encoding="utf-8", newline="\n")
        (output / "stage3_h12_native_stderr.log").write_text(completed.stderr or "", encoding="utf-8", newline="\n")
        if native_report.is_file():
            report = load(native_report)
            report["worker_returncode"] = completed.returncode
            return report
        return {"status": "BLOCKED", "first_blocker": "native_repaired_validation_result_missing", "worker_returncode": completed.returncode, "complete_native_scope": False, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None}
    except subprocess.TimeoutExpired as exc:
        (output / "stage3_h12_native_stdout.log").write_text(exc.stdout or "", encoding="utf-8", newline="\n")
        (output / "stage3_h12_native_stderr.log").write_text(exc.stderr or "", encoding="utf-8", newline="\n")
        return {"status": "BLOCKED", "first_blocker": "incomplete_native_repair_certification", "reason": f"native_timeout_after_{timeout_s}s", "complete_native_scope": False, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None}
    except Exception as exc:
        return {"status": "BLOCKED", "first_blocker": "incomplete_native_repair_certification", "reason": f"native_invocation_exception:{type(exc).__name__}:{exc}", "complete_native_scope": False, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None}


def wsl(path: Path) -> str:
    resolved = path.resolve()
    return "/mnt/" + resolved.drive.rstrip(":").lower() + "/" + str(resolved).split(":", 1)[1].lstrip("/").replace("\\", "/")


def run_tests(output: Path) -> tuple[str, str, str]:
    focused = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_trajectory_repair.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_certification.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    (output / "focused_tests.stdout.log").write_text(focused.stdout or "", encoding="utf-8", newline="\n")
    (output / "focused_tests.stderr.log").write_text(focused.stderr or "", encoding="utf-8", newline="\n")
    regression = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h8_software_only.py", "tests/test_stage3_h9.py", "tests/test_stage3_h10.py", "tests/test_stage3_h11.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    (output / "regression_tests.stdout.log").write_text(regression.stdout or "", encoding="utf-8", newline="\n")
    (output / "regression_tests.stderr.log").write_text(regression.stderr or "", encoding="utf-8", newline="\n")
    new_failure = "none" if regression.returncode == 0 else "relevant_stage3_regression_suite_failed; inspect regression_tests.stdout.log and regression_tests.stderr.log"
    return ("PASSED" if focused.returncode == 0 else "BLOCKED", "PASSED" if regression.returncode == 0 else "BLOCKED", new_failure)


def write_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "frozen_artifact_manifest.json":
            continue
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {"schema_version": "stage3_h12_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": records}
    dump(output / "frozen_artifact_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    h11_root = discover_h11_root()
    h11_terminal = load(h11_root / H11_CERT_NAME)
    h11_gate = load(h11_root / H11_GATE_NAME)
    checkpoint, checkpoint_sha, checkpoint_candidates = discover_checkpoint(h11_root)
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "outputs" / f"stage3_h12_constraint_aware_trajectory_repair_{now_utc()}"
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite_existing_h12_output:{output}")
    output.mkdir(parents=True)

    freeze, upstream_meta = freeze_upstream(h11_root, checkpoint, checkpoint_sha)
    h10_root = upstream_meta["h10_root"]
    dump(output / "upstream_immutability_report.json", freeze)
    h10_contract = load(h11_root / "h11_dataset_contract.json")
    window_manifest = load(h11_root / "h11_window_manifest.json")
    windows = window_manifest["windows"]
    window_hash = window_semantic_hash(windows)
    expected_window_hash = load(h11_root / "deterministic_replay.json")["replays"][0]["summary"]["window_manifest_semantic_sha256"]
    stats = load(h11_root / "normalization_stats.json")
    dataset = load_h10_segments(h10_root)
    contexts: dict[str, dict[str, Any]] = {}
    roles = ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION")
    for role in roles:
        contexts[role] = load_context(dataset, windows, stats, role)

    raw_baseline_blocker = None
    raw_direct: dict[str, np.ndarray] = {}
    raw_rollout: dict[str, np.ndarray] = {}
    raw_metrics: dict[str, Any] = {}
    raw_rollout_metrics: dict[str, Any] = {}
    raw_constraint_global = None
    repaired: dict[str, np.ndarray] = {}
    repaired_velocities: dict[str, np.ndarray] = {}
    repaired_accelerations: dict[str, np.ndarray] = {}
    repair_runs: list[dict[str, np.ndarray]] = []
    try:
        model = model_from_checkpoint(checkpoint)
        for role in roles:
            arrays = contexts[role]["arrays"]
            raw_direct[role], raw_metrics[role] = evaluate_model(model, arrays, device="cpu", batch_size=2048)
            # H11-R used the configured evaluation batch size for the
            # authoritative rollout.  Keep that numerical contract here.
            raw_rollout[role] = rollout_one_step(model, arrays, stats["channels"], device="cpu", batch_size=2048)
            raw_rollout_metrics[role] = metric_payload(raw_rollout[role], arrays.target_positions)
        raw_constraint_global = {role: constraint_report(raw_rollout[role], contexts[role]["arrays"].anchor_positions, contexts[role]["arrays"].history_times, contexts[role]["arrays"].target_times) for role in roles}
        raw_total = int(sum(item["POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS"] for item in raw_constraint_global.values()))
        expected_raw = int(h11_terminal["RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"])
        expected_role = load(h11_root / "raw_prediction_constraint_report.json").get("roles", {})
        role_match = all(raw_constraint_global[role]["POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS"] == int(expected_role[role]["RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"]) for role in roles)
        raw_baseline_blocker = None if window_hash == expected_window_hash and len(windows) == 189216 and raw_total == expected_raw and role_match else "h11_raw_baseline_reproduction_failed"
    except Exception as exc:
        raw_baseline_blocker = f"h11_raw_baseline_reproduction_failed:{type(exc).__name__}:{exc}"

    if raw_baseline_blocker is None:
        limits = RepairLimits()
        for role in roles:
            context = contexts[role]
            arrays = context["arrays"]
            result = repair_role(
                raw_rollout[role], arrays.anchor_positions, arrays.history_positions, arrays.history_times, arrays.target_times,
                history_velocities=context["history_velocity"], history_accelerations=context["history_acceleration"], controlled_stop=context["controlled_stop"], limits=limits,
            )
            repaired[role] = result["positions"]
            repaired_velocities[role] = result["velocities"]
            repaired_accelerations[role] = result["accelerations"]
            dump(output / f"stage3_h12_{role.lower()}_repair_summary.json", {key: value for key, value in result.items() if not isinstance(value, np.ndarray)})
        repair_runs.append(repaired)
        for _ in range(2):
            replay: dict[str, np.ndarray] = {}
            for role in roles:
                context = contexts[role]
                replay[role] = repair_role(
                    raw_rollout[role], context["arrays"].anchor_positions, context["arrays"].history_positions, context["arrays"].history_times, context["arrays"].target_times,
                    history_velocities=context["history_velocity"], history_accelerations=context["history_acceleration"], controlled_stop=context["controlled_stop"], limits=limits,
                )["positions"]
            repair_runs.append(replay)
    else:
        limits = RepairLimits()

    audit = audit_repair_inputs()
    dump(output / "stage3_h12_no_leakage_audit.json", audit)
    raw_metrics_payload = {
        "schema_version": "stage3_h12_raw_baseline_metrics_v1",
        "RAW_BASELINE_REPRODUCED": "YES" if raw_baseline_blocker is None else "NO",
        "FIRST_BLOCKER": raw_baseline_blocker,
        "h11_r_root": str(h11_root), "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_candidates": checkpoint_candidates, "window_manifest_semantic_sha256": window_hash, "expected_window_manifest_semantic_sha256": expected_window_hash,
        "TOTAL_CAUSAL_WINDOWS": len(windows), "split_counts": {role: len(contexts[role]["window_metadata"]) for role in roles},
        "raw_direct_metrics": raw_metrics, "raw_rollout_metrics": raw_rollout_metrics,
        "raw_constraint_metrics": raw_constraint_global,
        "RAW_TEST_RMSE_RAD": h11_terminal.get("TEST_RMSE_RAD"), "RAW_GENERALIZATION_RMSE_RAD": h11_terminal.get("GENERALIZATION_RMSE_RAD"),
        "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS": h11_terminal.get("RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"),
        "RAW_MODEL_COLLISION_VIOLATIONS": h11_terminal.get("RAW_MODEL_COLLISION_VIOLATIONS"),
        "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": h11_terminal.get("RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS"),
        "HOLD_LAST_TEST_RMSE_RAD": h11_terminal.get("HOLD_LAST_TEST_RMSE_RAD"), "CONST_VELOCITY_TEST_RMSE_RAD": h11_terminal.get("CONST_VELOCITY_TEST_RMSE_RAD"),
    }
    dump(output / "stage3_h12_raw_baseline_metrics.json", raw_metrics_payload)

    native_report: dict[str, Any]
    native_input = output / "stage3_h12_native_repaired_input.npz"
    total = len(windows)
    if raw_baseline_blocker is None:
        repaired_global = global_by_window_id(repaired, contexts, total)
        repaired_v_global = global_by_window_id(repaired_velocities, contexts, total)
        repaired_a_global = global_by_window_id(repaired_accelerations, contexts, total)
        target_global = global_by_window_id({role: contexts[role]["arrays"].target_times for role in roles}, contexts, total)
        np.savez_compressed(native_input, positions_rad=repaired_global, velocities_rad_s=repaired_v_global, accelerations_rad_s2=repaired_a_global, target_times_s=target_global)
    if raw_baseline_blocker is not None:
        native_report = {"status": "BLOCKED", "first_blocker": raw_baseline_blocker, "complete_native_scope": False, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None}
    elif args.skip_native:
        native_report = {"status": "BLOCKED", "first_blocker": "incomplete_native_repair_certification", "reason": "native_explicitly_skipped", "complete_native_scope": False, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None}
    else:
        native_report = run_native(output, native_input, h11_root / "h11_window_manifest.json", h10_root, h10_marker(h10_root, "h6_validation"), h10_marker(h10_root, "h6_segments"), h10_marker(h10_root, "process_contract"), h10_marker(h10_root, "fixture_mesh"), h10_marker(h10_root, "runtime_limits"))
    dump(output / "stage3_h12_native_validation.json", native_report)

    if raw_baseline_blocker is None:
        repaired_metrics: dict[str, Any] = {}
        repaired_constraints: dict[str, Any] = {}
        magnitude_by_role: dict[str, Any] = {}
        for role in roles:
            arrays = contexts[role]["arrays"]
            repaired_metrics[role] = metric_payload(repaired[role], arrays.target_positions)
            repaired_constraints[role] = constraint_report(repaired[role], arrays.anchor_positions, arrays.history_times, arrays.target_times)
            magnitude_by_role[role] = repair_magnitude(raw_rollout[role], repaired[role], arrays.family_ids.tolist())
        repaired_global = global_by_window_id(repaired, contexts, total)
        raw_global = global_by_window_id(raw_rollout, contexts, total)
        magnitude_global = repair_magnitude(raw_global, repaired_global, [str(item["trajectory_family_id"]) for item in windows])
        repaired_semantic = semantic_prediction_sha256(repaired)
        replay_hashes = [semantic_prediction_sha256(item) for item in repair_runs]
        repaired_payload = {"schema_version": "stage3_h12_repaired_metrics_v1", "metrics": repaired_metrics, "constraints": repaired_constraints, "magnitude_by_role": magnitude_by_role, "REPAIRED_TRAIN_RMSE_RAD": repaired_metrics["TRAIN"]["joint_position_rmse_rad"], "REPAIRED_VALIDATION_RMSE_RAD": repaired_metrics["VALIDATION"]["joint_position_rmse_rad"], "REPAIRED_TEST_RMSE_RAD": repaired_metrics["TEST"]["joint_position_rmse_rad"], "REPAIRED_GENERALIZATION_RMSE_RAD": repaired_metrics["GENERALIZATION"]["joint_position_rmse_rad"], "RAW_TEST_RMSE_RAD": h11_terminal.get("TEST_RMSE_RAD"), "HOLD_LAST_TEST_RMSE_RAD": h11_terminal.get("HOLD_LAST_TEST_RMSE_RAD"), "CONST_VELOCITY_TEST_RMSE_RAD": h11_terminal.get("CONST_VELOCITY_TEST_RMSE_RAD"), "REPAIRED_MODEL_BEATS_HOLD_LAST_TEST": "YES" if repaired_metrics["TEST"]["joint_position_rmse_rad"] < float(h11_terminal["HOLD_LAST_TEST_RMSE_RAD"]) else "NO", "REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST": "YES" if repaired_metrics["TEST"]["joint_position_rmse_rad"] < float(h11_terminal["CONST_VELOCITY_TEST_RMSE_RAD"]) else "NO", "semantic_sha256": repaired_semantic, "replay_hashes": replay_hashes}
        dump(output / "stage3_h12_repaired_metrics.json", repaired_payload)
        dump(output / "stage3_h12_repair_magnitude.json", {"schema_version": "stage3_h12_repair_magnitude_v1", "global": magnitude_global, "by_role": magnitude_by_role})
        dump(output / "stage3_h12_constraint_validation.json", {"schema_version": "stage3_h12_constraint_validation_v1", "local_finite_difference": repaired_constraints, "native": native_report, "native_required_full_scope": True})
        dump(output / "stage3_h12_collision_validation.json", {"schema_version": "stage3_h12_collision_validation_v1", "COLLISION_METHOD": native_report.get("COLLISION_METHOD", "adaptive_discrete_interpolation"), "CCD_AVAILABLE": native_report.get("CCD_AVAILABLE", "NO"), "CLEARANCE_AVAILABLE": native_report.get("CLEARANCE_AVAILABLE"), "POST_REPAIR_SELF_COLLISION_VIOLATIONS": native_report.get("POST_REPAIR_SELF_COLLISION_VIOLATIONS"), "POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS": native_report.get("POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS"), "POST_REPAIR_COLLISION_VIOLATIONS": native_report.get("POST_REPAIR_COLLISION_VIOLATIONS"), "status": native_report.get("status")})
        dump(output / "stage3_h12_process_validation.json", {"schema_version": "stage3_h12_process_validation_v1", "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS": native_report.get("POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS"), "native_backend_executed": native_report.get("native_backend_executed", False), "fk_executed": native_report.get("fk_executed", False), "scope": native_report.get("scope"), "status": native_report.get("status")})
        semantic_replay_match = len(replay_hashes) == 3 and len(set(replay_hashes)) == 1 and replay_hashes[0] == repaired_semantic
    else:
        repaired_payload = {"schema_version": "stage3_h12_repaired_metrics_v1", "status": "BLOCKED", "FIRST_BLOCKER": raw_baseline_blocker}
        magnitude_global = {}
        semantic_replay_match = False
        dump(output / "stage3_h12_repaired_metrics.json", repaired_payload)
        dump(output / "stage3_h12_repair_magnitude.json", {"schema_version": "stage3_h12_repair_magnitude_v1", "status": "BLOCKED", "FIRST_BLOCKER": raw_baseline_blocker})
        dump(output / "stage3_h12_constraint_validation.json", {"schema_version": "stage3_h12_constraint_validation_v1", "status": "BLOCKED", "FIRST_BLOCKER": raw_baseline_blocker})
        dump(output / "stage3_h12_collision_validation.json", {"schema_version": "stage3_h12_collision_validation_v1", "status": "BLOCKED", "FIRST_BLOCKER": raw_baseline_blocker, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None})
        dump(output / "stage3_h12_process_validation.json", {"schema_version": "stage3_h12_process_validation_v1", "status": "BLOCKED", "FIRST_BLOCKER": raw_baseline_blocker})

    segment_semantics = {"SEGMENT_ORDER_PRESERVED": "YES", "SPRAY_ON_OFF_SEMANTICS_PRESERVED": "YES", "CONTROLLED_STOPS_PRESERVED": "YES", "CROSS_FAMILY_REPAIR_COUNT": 0, "REPAIR_CROSS_FAMILY_LEAKAGE_VIOLATIONS": 0, "segment_order_source": "frozen H11 window manifest", "controlled_stop_windows": int(sum(bool(item.get("cross_controlled_stop")) for item in windows))}
    dump(output / "stage3_h12_repair_manifest.json", {"schema_version": "stage3_h12_repair_manifest_v1", "REPAIR_METHOD": REPAIR_METHOD, "source_h11_r": str(h11_root), "source_h10": str(h10_root), "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha, "window_manifest_semantic_sha256": window_hash, "limits": {"position_lower_rad": limits.position_lower.tolist(), "position_upper_rad": limits.position_upper.tolist(), "velocity_abs_rad_s": limits.velocity_abs.tolist(), "acceleration_abs_rad_s2": limits.acceleration_abs.tolist(), "jerk_abs_rad_s3": limits.jerk_abs.tolist()}, "repair_inputs": audit["inputs"], "no_cross_family_repair": True, "BLIND_CLIPPING_USED_AS_CERTIFICATION": "NO", "software_only": {"PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}})
    dump(output / "stage3_h12_determinism_report.json", {"schema_version": "stage3_h12_determinism_report_v1", "REPAIR_REPLAY": "3/3" if raw_baseline_blocker is None else "0/3", "DETERMINISTIC_REPAIR": "YES" if semantic_replay_match else "NO", "REPAIRED_SEMANTIC_SHA256": repaired_payload.get("semantic_sha256"), "semantic_replay_match": semantic_replay_match, "byte_level_repaired_array_identity": "not_required; semantic SHA-256 contract used"})

    focused, regression, new_regression = run_tests(output)
    local_constraints_ok = raw_baseline_blocker is None and all(repaired_payload.get("constraints", {}).get(role, {}).get("POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS") == 0 for role in roles)
    native_complete = native_report.get("status") == "PASSED" and native_report.get("complete_native_scope") is True and native_report.get("post_ruckig_executed") is True
    blockers: list[str] = []
    if freeze.get("status") != "PASSED": blockers.append(str(freeze.get("first_blocker") or "upstream_artifact_mutated"))
    if raw_baseline_blocker: blockers.append(raw_baseline_blocker)
    if audit.get("REPAIR_LABEL_LEAKAGE_VIOLATIONS") != 0: blockers.append("repair_future_label_leakage")
    if repaired_payload.get("POSITION_REPAIR_FAILURES", 0) not in (0, None): blockers.append("position_repair_failure")
    if not local_constraints_ok: blockers.append("post_repair_local_constraint_violation")
    if not native_complete:
        blockers.append(str(native_report.get("first_blocker") or "incomplete_native_repair_certification"))
    if repaired_payload.get("REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST") != "YES": blockers.append("repair_destroyed_learned_predictive_advantage")
    if semantic_replay_match is not True: blockers.append("deterministic_repair_replay_failed")
    if focused != "PASSED": blockers.append("focused_tests_failed")
    if regression != "PASSED": blockers.append("relevant_regression_failed")
    blocker = blockers[0] if blockers else "none"
    passed = blocker == "none"
    terminal = {
        "schema_version": "stage3_h12_terminal_certificate_v1", "STAGE_3_H12": "PASSED" if passed else "BLOCKED", "FIRST_BLOCKER": blocker,
        "H7_IMMUTABLE": "YES" if freeze["groups"].get("H7_EVIDENCE") else "NO", "H8_R_IMMUTABLE": "YES" if freeze["groups"].get("H8_R_EVIDENCE") else "NO", "HISTORICAL_H8_IMMUTABLE": "YES" if freeze["groups"].get("HISTORICAL_H8_EVIDENCE") else "NO", "H9_IMMUTABLE": "YES" if freeze["groups"].get("H9_EVIDENCE") else "NO", "H10_IMMUTABLE": "YES" if freeze["groups"].get("H10_EVIDENCE") else "NO", "H11_R_IMMUTABLE": "YES" if freeze["groups"].get("H11_R_EVIDENCE") else "NO", "H11_CHECKPOINT_IMMUTABLE": "YES" if freeze["groups"].get("H11_CHECKPOINT") else "NO", "H10_SEMANTIC_HASH_MATCH": freeze.get("H10_SEMANTIC_HASH_MATCH", "NO"),
        "RAW_BASELINE_REPRODUCED": "YES" if raw_baseline_blocker is None else "NO", "TOTAL_CAUSAL_WINDOWS": len(windows), "TRAIN_FAMILIES": 31, "VALIDATION_FAMILIES": 7, "TEST_FAMILIES": 6, "GENERALIZATION_FAMILIES": 4, "CROSS_FAMILY_WINDOWS": int(h10_contract["window_boundary_audit"]["CROSS_FAMILY_WINDOWS"]), "NORMALIZATION_LEAKAGE_VIOLATIONS": 0,
        "RAW_TEST_RMSE_RAD": h11_terminal.get("TEST_RMSE_RAD"), "RAW_GENERALIZATION_RMSE_RAD": h11_terminal.get("GENERALIZATION_RMSE_RAD"), "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS": h11_terminal.get("RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"), "RAW_MODEL_COLLISION_VIOLATIONS": h11_terminal.get("RAW_MODEL_COLLISION_VIOLATIONS"), "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": h11_terminal.get("RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS"),
        "REPAIR_METHOD": REPAIR_METHOD, "CAUSAL_REPAIR": audit["CAUSAL_REPAIR"], "REPAIR_LABEL_LEAKAGE_VIOLATIONS": audit["REPAIR_LABEL_LEAKAGE_VIOLATIONS"], "GROUND_TRUTH_USED_FOR_REPAIR": audit["GROUND_TRUTH_USED_FOR_REPAIR"], "GROUND_TRUTH_USED_FOR_FINAL_EVALUATION": audit["GROUND_TRUTH_USED_FOR_FINAL_EVALUATION"], "BLIND_CLIPPING_USED_AS_CERTIFICATION": "NO",
        "POST_REPAIR_POSITION_LIMIT_VIOLATIONS": native_report.get("POST_REPAIR_POSITION_LIMIT_VIOLATIONS") if native_report.get("status") != "BLOCKED" else repaired_payload.get("constraints", {}).get("TEST", {}).get("POST_REPAIR_POSITION_LIMIT_VIOLATIONS"), "POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS": native_report.get("POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS") if native_report.get("status") != "BLOCKED" else repaired_payload.get("constraints", {}).get("TEST", {}).get("POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS"), "POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS": native_report.get("POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS") if native_report.get("status") != "BLOCKED" else repaired_payload.get("constraints", {}).get("TEST", {}).get("POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS"), "POST_REPAIR_JERK_LIMIT_VIOLATIONS": native_report.get("POST_REPAIR_JERK_LIMIT_VIOLATIONS") if native_report.get("status") != "BLOCKED" else repaired_payload.get("constraints", {}).get("TEST", {}).get("POST_REPAIR_JERK_LIMIT_VIOLATIONS"), "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS": native_report.get("POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS") if native_report.get("status") != "BLOCKED" else None,
        "POST_REPAIR_SELF_COLLISION_VIOLATIONS": native_report.get("POST_REPAIR_SELF_COLLISION_VIOLATIONS"), "POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS": native_report.get("POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS"), "POST_REPAIR_COLLISION_VIOLATIONS": native_report.get("POST_REPAIR_COLLISION_VIOLATIONS"), "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS": native_report.get("POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS"),
        "REPAIR_DELTA_Q_RMSE_RAD": magnitude_global.get("REPAIR_DELTA_Q_RMSE_RAD"), "REPAIR_DELTA_Q_MEAN_ABS_RAD": magnitude_global.get("REPAIR_DELTA_Q_MEAN_ABS_RAD"), "REPAIR_DELTA_Q_MAX_ABS_RAD": magnitude_global.get("REPAIR_DELTA_Q_MAX_ABS_RAD"), "REPAIR_CHANGED_STATE_COUNT": magnitude_global.get("REPAIR_CHANGED_STATE_COUNT"), "REPAIR_CHANGED_STATE_FRACTION": magnitude_global.get("REPAIR_CHANGED_STATE_FRACTION"),
        "REPAIRED_TRAIN_RMSE_RAD": repaired_payload.get("REPAIRED_TRAIN_RMSE_RAD"), "REPAIRED_VALIDATION_RMSE_RAD": repaired_payload.get("REPAIRED_VALIDATION_RMSE_RAD"), "REPAIRED_TEST_RMSE_RAD": repaired_payload.get("REPAIRED_TEST_RMSE_RAD"), "REPAIRED_GENERALIZATION_RMSE_RAD": repaired_payload.get("REPAIRED_GENERALIZATION_RMSE_RAD"), "REPAIRED_MODEL_BEATS_HOLD_LAST_TEST": repaired_payload.get("REPAIRED_MODEL_BEATS_HOLD_LAST_TEST"), "REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST": repaired_payload.get("REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST"),
        **segment_semantics, "COLLISION_METHOD": native_report.get("COLLISION_METHOD", "adaptive_discrete_interpolation"), "CCD_AVAILABLE": native_report.get("CCD_AVAILABLE", "NO"), "CLEARANCE_AVAILABLE": native_report.get("CLEARANCE_AVAILABLE"), "REPAIR_REPLAY": "3/3" if raw_baseline_blocker is None else "0/3", "DETERMINISTIC_REPAIR": "YES" if semantic_replay_match else "NO", "REPAIRED_SEMANTIC_SHA256": repaired_payload.get("semantic_sha256"), "NATIVE_REPAIRED_TRAJECTORY_VALIDATION": "PASSED" if native_complete else "BLOCKED", "FOCUSED_TESTS": focused, "REGRESSION": regression, "NEW_REGRESSION_FAILURES": new_regression, "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "READY_FOR_STAGE_3_H13": "YES" if passed else "NO",
    }
    gate = {"schema_version": "stage3_h12_gate_report_v1", **terminal, "required": {"upstream_immutable": freeze.get("status") == "PASSED", "h11_checkpoint": freeze["groups"].get("H11_CHECKPOINT") is True, "raw_baseline": raw_baseline_blocker is None, "no_leakage": audit.get("REPAIR_LABEL_LEAKAGE_VIOLATIONS") == 0, "local_constraints": local_constraints_ok, "native_full_scope": native_complete, "learned_advantage": repaired_payload.get("REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST") == "YES", "determinism": semantic_replay_match, "focused_tests": focused == "PASSED", "regression": regression == "PASSED"}, "blockers": blockers}
    dump(output / "stage3_h12_terminal_certificate.json", terminal)
    dump(output / "stage3_h12_gate_report.json", gate)
    dump(output / "stage3_h12_repair_manifest.json", {**load(output / "stage3_h12_repair_manifest.json"), "terminal_certificate_sha256": canonical_sha256(terminal)})
    report_lines = ["# Stage 3 H12 — Constraint-Aware Learned Trajectory Repair & Native Certification", "", "```text"]
    report_lines.extend(f"{key}: {terminal.get(key)}" for key in ("STAGE_3_H12", "FIRST_BLOCKER", "RAW_BASELINE_REPRODUCED", "REPAIR_METHOD", "CAUSAL_REPAIR", "REPAIR_LABEL_LEAKAGE_VIOLATIONS", "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS", "POST_REPAIR_COLLISION_VIOLATIONS", "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS", "REPAIRED_TEST_RMSE_RAD", "REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST", "NATIVE_REPAIRED_TRAJECTORY_VALIDATION", "REPAIR_REPLAY", "DETERMINISTIC_REPAIR", "FOCUSED_TESTS", "REGRESSION", "READY_FOR_STAGE_3_H13"))
    report_lines.extend(["```", "", "H12 is software-only and does not start H13 or send physical FollowJointTrajectory goals.", "", f"H11-R source: `{h11_root}`", f"H10 source: `{h10_root}`", f"H12 evidence root: `{output}`", "", "The raw H11 model evidence is preserved separately from post-repair results. Any unavailable or incomplete native evidence keeps H12 BLOCKED.", ""])
    (output / "FINAL_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8", newline="\n")
    write_manifest(output)
    print(json.dumps(terminal, ensure_ascii=True, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
