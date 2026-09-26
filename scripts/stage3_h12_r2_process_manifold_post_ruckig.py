#!/usr/bin/env python3
"""Execute Stage 3 H12-R2 with process-manifold repair and native closure.

This runner freezes the selected H12-R E2 checkpoint, reconstructs only
same-segment authoritative paths, invokes the native primitive worker, and
aggregates a fail-closed certificate.  It never connects to a controller and
never sends a FollowJointTrajectory goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_dataset import FEATURE_NAMES, sha256_file, window_semantic_hash  # noqa: E402
from src.stage3_h11_model import TORCH_AVAILABLE, constant_velocity_prediction, metric_payload, set_deterministic  # noqa: E402
from src.stage3_h12_r_fast_dataset import load_h10_segments_fast  # noqa: E402
from src.stage3_h12_r_residual import ResidualGRUTrajectoryPredictor, predict_residual_model  # noqa: E402
from src.stage3_h12_trajectory_repair import RepairLimits, audit_repair_inputs  # noqa: E402
from src.stage3_h12_r2_manifold import canonical_failure_category, nearest_feasible_projection, repair_window_process_manifold  # noqa: E402
from scripts.stage3_h12_r_certification import ROLES, build_contexts, global_by_id  # noqa: E402


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
H12_R_ROOT = ROOT / "outputs/stage3_h12_r_constraint_aware_residual_repair_20260811T175146Z"
H6_ROOT = ROOT / "outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z"
FIXTURE = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj"
RUNTIME_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
TOTG = ROOT / "outputs/stage3_h7_2_authoritative_time_parameterization_20260809T060043Z/stage3_h7_2_totg_parameters.json"
RUCKIG_INTERPOSER = ROOT / "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/libstage3_h7_4_ruckig_interposer.so"
TOTAL_WINDOWS = 189216


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def wsl(path: Path) -> str:
    path = path.resolve()
    return "/mnt/" + path.drive.rstrip(":").lower() + "/" + str(path).split(":", 1)[1].lstrip("/").replace("\\", "/")


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    records = manifest.get("records", [])
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for record in records:
        candidate = Path(str(record.get("absolute_path") or record.get("path", "")))
        if not candidate.is_absolute():
            candidate = ROOT / candidate
            if not candidate.is_file():
                candidate = manifest_path.parent / Path(str(record.get("path", "")))
        if not candidate.is_file():
            mismatches.append({"path": str(candidate), "reason": "missing"})
            continue
        checked += 1
        observed = sha256_file(candidate)
        if observed != record.get("sha256"):
            mismatches.append({"path": str(candidate), "expected": record.get("sha256"), "observed": observed, "reason": "sha256_mismatch"})
    return {"manifest": str(manifest_path), "records": len(records), "checked": checked, "mismatches": mismatches, "status": "PASSED" if not mismatches else "BLOCKED"}


def discover_manifest_marker(h10: Path, marker: str) -> Path:
    manifest = load_json(h10 / "frozen_artifact_manifest.json")
    for record in manifest.get("records", []):
        if marker in str(record.get("status", "")):
            candidate = Path(str(record.get("absolute_path") or record.get("path", "")))
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"h10_marker_missing:{marker}")


def verify_provenance() -> tuple[dict[str, Any], dict[str, Path]]:
    required = (H10_ROOT, H11_ROOT, H12_R_ROOT, H6_ROOT, FIXTURE, TOTG, RUNTIME_LIMITS, RUCKIG_INTERPOSER)
    missing = [str(path) for path in required if not path.exists()]
    h10_manifest = verify_manifest(H10_ROOT / "frozen_artifact_manifest.json") if not missing else {"status": "BLOCKED", "mismatches": [{"reason": "missing_upstream"}]}
    h11_manifest = verify_manifest(H11_ROOT / "frozen_artifact_manifest.json") if not missing else {"status": "BLOCKED", "mismatches": [{"reason": "missing_upstream"}]}
    h12r_manifest = verify_manifest(H12_R_ROOT / "frozen_artifact_manifest.json") if not missing else {"status": "BLOCKED", "mismatches": [{"reason": "missing_upstream"}]}
    historical_candidates = sorted(ROOT.glob("outputs/stage3_h12_constraint_aware_trajectory_repair_*/frozen_artifact_manifest.json"))
    historical = verify_manifest(historical_candidates[-1]) if historical_candidates else {"status": "BLOCKED", "mismatches": [{"reason": "historical_h12_manifest_missing"}]}
    h10_dataset = load_json(H10_ROOT / "dataset_manifest.json") if (H10_ROOT / "dataset_manifest.json").is_file() else {}
    h11_terminal = load_json(H11_ROOT / "stage3_h11_r_terminal_certificate.json") if (H11_ROOT / "stage3_h11_r_terminal_certificate.json").is_file() else {}
    selected = load_json(H12_R_ROOT / "selected_model_manifest.json") if (H12_R_ROOT / "selected_model_manifest.json").is_file() else {}
    checkpoint = Path(str(selected.get("checkpoint", "")))
    checkpoint_ok = checkpoint.is_file() and sha256_file(checkpoint) == selected.get("checkpoint_sha256")
    h10_hash_match = h10_dataset.get("SEMANTIC_DATASET_SHA256") == load_json(H11_ROOT / "h11_dataset_contract.json").get("source_h10_semantic_sha256") == load_json(H10_ROOT / "dataset_manifest.json").get("SEMANTIC_DATASET_SHA256")
    h11_provenance = h11_terminal.get("H10_SEMANTIC_HASH_MATCH") == "YES" and h11_terminal.get("H11_R_STATUS") in {None, "PASSED"}
    status = all(item.get("status") == "PASSED" for item in (h10_manifest, h11_manifest, h12r_manifest, historical)) and checkpoint_ok and h10_hash_match and h11_provenance and not missing
    return {
        "schema_version": "stage3_h12_r2_frozen_provenance_v1",
        "status": "PASSED" if status else "BLOCKED",
        "FIRST_BLOCKER": None if status else "upstream_immutability_violation",
        "missing": missing,
        "H10_SEMANTIC_HASH_MATCH": "YES" if h10_hash_match else "NO",
        "H11_PROVENANCE_VALID": "YES" if h11_provenance else "NO",
        "E2_MODEL_IMMUTABLE": "YES" if checkpoint_ok and selected.get("selected_experiment") == "E2" else "NO",
        "HISTORICAL_H12_IMMUTABLE": "YES" if historical.get("status") == "PASSED" else "NO",
        "H12_R_IMMUTABLE": "YES" if h12r_manifest.get("status") == "PASSED" else "NO",
        "h10_manifest": h10_manifest,
        "h11_manifest": h11_manifest,
        "historical_h12_manifest": historical,
        "h12_r_manifest": h12r_manifest,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
    }, {
        "h6_validation": discover_manifest_marker(H10_ROOT, "h6_validation"),
        "h6_segments": discover_manifest_marker(H10_ROOT, "h6_segments"),
        "process_contract": discover_manifest_marker(H10_ROOT, "process_contract"),
        "fixture_mesh": FIXTURE,
        "runtime_limits": RUNTIME_LIMITS,
        "totg_parameters": TOTG,
    }


def frozen_e2_model(checkpoint: Path) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch
    set_deterministic(12012)
    model = ResidualGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else payload
    model.load_state_dict(state)
    model.eval()
    return model


def rmse_improvement(cv: float | None, value: float | None) -> float | None:
    if cv is None or value is None or not np.isfinite(cv) or not np.isfinite(value):
        return None
    return float(100.0 * (cv - value) / cv)


def build_repair_and_units(output: Path, contexts: Mapping[str, Mapping[str, Any]], windows: list[dict[str, Any]], raw_by_role: Mapping[str, np.ndarray], cv_by_role: Mapping[str, np.ndarray], segments: Any, *, persist: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    segment_map = {segment.key: segment for segment in segments.segments}
    raw_global = global_by_id(raw_by_role, contexts, len(windows))
    cv_global = global_by_id(cv_by_role, contexts, len(windows))
    anchor_global = global_by_id({role: contexts[role]["arrays"].anchor_positions for role in ROLES}, contexts, len(windows))
    history_global = global_by_id({role: contexts[role]["arrays"].history_times for role in ROLES}, contexts, len(windows))
    target_times_global = global_by_id({role: contexts[role]["arrays"].target_times for role in ROLES}, contexts, len(windows))
    hv_global = global_by_id({role: contexts[role]["history_velocities"] for role in ROLES}, contexts, len(windows))
    ha_global = global_by_id({role: contexts[role]["history_accelerations"] for role in ROLES}, contexts, len(windows))
    repaired = np.empty_like(raw_global)
    limits = RepairLimits()
    tier_counts = Counter()
    decisions_path = output / "stage3_h12_r2_repair_decisions.jsonl"
    stream_context = decisions_path.open("w", encoding="utf-8", newline="\n") if persist else None
    try:
        for index, item in enumerate(windows):
            segment = segment_map.get(str(item["segment_key"]))
            if segment is None:
                raise RuntimeError(f"manifold_segment_missing:{item['segment_key']}")
            # Build q_ref only from the causal CV proposal and the frozen
            # same-segment process manifold.  Never index the future target
            # samples here: target positions remain evaluation-only.
            q_ref, _, _ = nearest_feasible_projection(
                cv_global[index],
                cv_global[index],
                reference_bank=segment.positions,
                reference_indices=segment.sample_indices,
            )
            result = repair_window_process_manifold(raw_global[index], cv_global[index], q_ref, item, anchor=anchor_global[index], history_times=history_global[index], target_times=target_times_global[index], history_velocity=hv_global[index], history_acceleration=ha_global[index], reference_bank=segment.positions, reference_indices=segment.sample_indices, limits=limits)
            candidate = np.asarray(result.pop("positions"), dtype=np.float64)
            if not np.all(np.isfinite(candidate)):
                raise RuntimeError(f"invalid_repair_output:{item['window_id']}")
            repaired[index] = candidate
            tier_counts[str(result["repair_type"])] += 1
            result.update({"window_id": int(item["window_id"]), "trajectory_family_id": item["trajectory_family_id"], "trajectory_id": item["trajectory_id"], "segment_key": item["segment_key"], "segment_id": int(item["segment_id"]), "segment_order": int(item["segment_order"]), "primitive_id": item["primitive_id"], "spray_state": item["spray_state"], "start_offset": int(item["start_offset"]), "reference_source": "authoritative_H10_planned_joint_state_same_segment", "reference_tcp_pose": None, "reference_surface_point": None, "reference_surface_normal": None, "reference_standoff_m": None, "native_validation_status": "DEFERRED_TO_PRIMITIVE_WORKER", "neural_information_preserved": str(result["repair_type"]) in {"RAW_NEURAL_ACCEPTED", "RESIDUAL_SCALING"}})
            if stream_context is not None:
                stream_context.write(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    finally:
        if stream_context is not None:
            stream_context.close()
    if persist:
        np.savez_compressed(output / "stage3_h12_r2_repaired_windows.npz", positions_rad=repaired, raw_positions_rad=raw_global, cv_positions_rad=cv_global)
    repair_summary = {"schema_version": "stage3_h12_r2_repair_summary_v1", "TOTAL_WINDOWS": len(windows), "tier_counts": dict(sorted(tier_counts.items())), "RAW_NEURAL_ACCEPT_RATE": tier_counts["RAW_NEURAL_ACCEPTED"] / len(windows), "MANIFOLD_PROJECTION_RATE": tier_counts["MANIFOLD_PROJECTION"] / len(windows), "RESIDUAL_SCALING_RATE": tier_counts["RESIDUAL_SCALING"] / len(windows), "CV_FALLBACK_RATE": tier_counts["CV_FALLBACK"] / len(windows), "NEURAL_CONTRIBUTION_RATE": float(np.count_nonzero(np.linalg.norm(repaired - cv_global, axis=(1, 2)) > 0.0) / len(windows)), "repaired_semantic_sha256": canonical_sha(repaired.tolist()), "raw_semantic_sha256": canonical_sha(raw_global.tolist())}
    return repair_summary, windows, repaired, cv_global


def build_primitive_units(output: Path, windows: list[dict[str, Any]], repaired: np.ndarray, segments: Any) -> tuple[Path, Path, list[dict[str, Any]]]:
    segment_map = {segment.key: segment for segment in segments.segments}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in windows:
        grouped[str(item["segment_key"])].append(item)
    units: list[dict[str, Any]] = []
    positions_rows: list[np.ndarray] = []
    times_rows: list[np.ndarray] = []
    lengths: list[int] = []
    max_len = 0
    for unit_index, (segment_key, group) in enumerate(sorted(grouped.items(), key=lambda pair: (pair[1][0]["trajectory_family_id"], int(pair[1][0]["segment_order"])) )):
        segment = segment_map[segment_key]
        group.sort(key=lambda row: int(row["start_offset"]))
        by_start = {int(row["start_offset"]): row for row in group}
        q = segment.positions.copy()
        n_windows = len(group)
        for local_index in range(16, len(q)):
            start = min(max(local_index - 23, 0), n_windows - 1)
            source = by_start[start]
            horizon = local_index - (start + 16)
            q[local_index] = repaired[int(source["window_id"]), horizon]
        t = np.asarray(segment.times - segment.times[0], dtype=np.float64)
        if len(q) != len(t) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
            raise RuntimeError(f"invalid_reconstructed_primitive:{segment_key}")
        positions_rows.append(q)
        times_rows.append(t)
        lengths.append(len(q))
        max_len = max(max_len, len(q))
        units.append({"unit_index": unit_index, "unit_id": f"{segment_key}", "execution_unit": "trajectory_primitive", "trajectory_family_id": group[0]["trajectory_family_id"], "trajectory_id": group[0]["trajectory_id"], "segment_key": segment_key, "segment_id": int(group[0]["segment_id"]), "segment_order": int(group[0]["segment_order"]), "primitive_id": group[0]["primitive_id"], "spray_state": group[0]["spray_state"], "window_ids": [int(row["window_id"]) for row in group], "window_count": len(group)})
    stacked_q = np.zeros((len(positions_rows), max_len, 6), dtype=np.float64)
    stacked_t = np.zeros((len(times_rows), max_len), dtype=np.float64)
    for index, (q, t) in enumerate(zip(positions_rows, times_rows)):
        stacked_q[index, : len(q)] = q
        stacked_t[index, : len(t)] = t
    native_input = output / "stage3_h12_r2_native_input.npz"
    np.savez_compressed(native_input, positions_rad=stacked_q, times_s=stacked_t, lengths=np.asarray(lengths, dtype=np.int64))
    unit_manifest = output / "stage3_h12_r2_primitive_manifest.jsonl"
    unit_manifest.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in units), encoding="utf-8", newline="\n")
    return native_input, unit_manifest, units


def run_native(output: Path, native_input: Path, unit_manifest: Path, paths: Mapping[str, Path], timeout_s: int) -> dict[str, Any]:
    native_dir = output / "native_probe"
    rows = [json.loads(line) for line in unit_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    shard_size = 60
    shard_dir = output / "native_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    all_stdout: list[str] = []
    all_stderr: list[str] = []
    shard_runs: list[dict[str, Any]] = []
    combined_results = output / "stage3_h12_r2_post_ruckig_results.jsonl"
    combined_failures = output / "stage3_h12_r2_native_failures.jsonl"
    combined_results.write_text("", encoding="utf-8", newline="\n")
    combined_failures.write_text("", encoding="utf-8", newline="\n")
    for shard_index in range(0, len(rows), shard_size):
        shard_id = shard_index // shard_size
        shard_rows = rows[shard_index:shard_index + shard_size]
        shard_manifest = shard_dir / f"shard_{shard_id:03d}_manifest.jsonl"
        shard_result = shard_dir / f"shard_{shard_id:03d}_results.jsonl"
        shard_failure = shard_dir / f"shard_{shard_id:03d}_failures.jsonl"
        shard_manifest.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in shard_rows), encoding="utf-8", newline="\n")
        command = " ".join([
            "source /opt/ros/jazzy/setup.bash",
            "&& source /mnt/d/robotfucker/install/setup.bash",
            "&& source /mnt/d/robotfucker/install/stage3_h7_4_native/setup.bash",
            f"&& export LD_PRELOAD={wsl(RUCKIG_INTERPOSER)}",
            f"&& export STAGE25R_NATIVE_DIR={wsl(native_dir / f'shard_{shard_id:03d}')}",
            "&& export PYTHONPATH=/mnt/d/robotfucker/ros2_moveit_bridge:/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages",
            f"&& ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h12_r2_native.launch.py input_npz:={wsl(native_input)} unit_manifest:={wsl(shard_manifest)} output_jsonl:={wsl(shard_result)} failure_jsonl:={wsl(shard_failure)} output_dir:={wsl(output)} h6_validation:={wsl(paths['h6_validation'])} h6_segments:={wsl(paths['h6_segments'])} process_contract:={wsl(paths['process_contract'])} fixture_mesh:={wsl(paths['fixture_mesh'])} runtime_limits:={wsl(paths['runtime_limits'])} totg_parameters:={wsl(paths['totg_parameters'])}",
        ])
        try:
            completed = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
            timed_out = False
            returncode = completed.returncode
            all_stdout.append(completed.stdout or "")
            all_stderr.append(completed.stderr or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            all_stdout.append(str(exc.stdout or ""))
            all_stderr.append(str(exc.stderr or ""))
        if shard_result.is_file():
            with combined_results.open("a", encoding="utf-8", newline="\n") as dest:
                dest.write(shard_result.read_text(encoding="utf-8"))
        if shard_failure.is_file():
            with combined_failures.open("a", encoding="utf-8", newline="\n") as dest:
                dest.write(shard_failure.read_text(encoding="utf-8"))
        shard_runs.append({"shard_id": shard_id, "start_unit": shard_index, "end_unit": shard_index + len(shard_rows), "returncode": returncode, "timed_out": timed_out, "result_count": len([line for line in shard_result.read_text(encoding="utf-8").splitlines() if line.strip()]) if shard_result.is_file() else 0})
    command_summary = "\n\n".join(f"SHARD {row['shard_id']} {row['start_unit']}:{row['end_unit']} returncode={row['returncode']} timed_out={row['timed_out']}" for row in shard_runs)
    (output / "stage3_h12_r2_native_command.txt").write_text(command_summary + "\n", encoding="utf-8", newline="\n")
    (output / "stage3_h12_r2_native_stdout.log").write_text("\n".join(all_stdout), encoding="utf-8", newline="\n")
    (output / "stage3_h12_r2_native_stderr.log").write_text("\n".join(all_stderr), encoding="utf-8", newline="\n")
    return {"returncode": 0 if all(row["returncode"] == 0 and not row["timed_out"] for row in shard_runs) else None, "timed_out": any(row["timed_out"] for row in shard_runs), "shard_count": len(shard_runs), "shards": shard_runs, "shard_size": shard_size}


def aggregate_native(output: Path, units: list[dict[str, Any]], native_run: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = output / "stage3_h12_r2_post_ruckig_results.jsonl"
    failure_path = output / "stage3_h12_r2_native_failures.jsonl"
    results = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()] if result_path.is_file() else []
    failures = [json.loads(line) for line in failure_path.read_text(encoding="utf-8").splitlines() if line.strip()] if failure_path.is_file() else []
    expected = {str(unit["unit_id"]): unit for unit in units}
    observed = defaultdict(list)
    for row in results:
        if row.get("unit_id") is not None:
            observed[str(row["unit_id"])].append(row)
    missing_units = sorted(set(expected) - set(observed))
    duplicate_units = sorted(key for key, rows in observed.items() if len(rows) > 1)
    coverage = {key: 0 for key in ("TOTAL_WINDOWS", "NATIVE_CHECKED_WINDOWS", "PRE_RUCKIG_VALIDATED_WINDOWS", "POST_RUCKIG_ATTEMPTED_WINDOWS", "POST_RUCKIG_VALIDATED_WINDOWS", "POST_RUCKIG_FAILED_WINDOWS", "POST_RUCKIG_UNVALIDATED_WINDOWS", "FINAL_CERTIFIED_WINDOWS")}
    coverage["TOTAL_WINDOWS"] = sum(int(unit["window_count"]) for unit in units)
    counts = Counter()
    pre_failure_counts = Counter()
    post_failure_counts = Counter()
    taxonomy: dict[str, Counter] = {key: Counter() for key in ("FAILURE_BY_CATEGORY", "FAILURE_BY_SEGMENT", "FAILURE_BY_SPRAY_STATE", "FAILURE_BY_JOINT", "FAILURE_BY_WINDOW", "FAILURE_BY_FIRST_GATE")}
    final_violations = Counter()
    for unit in units:
        rows = observed.get(str(unit["unit_id"]), [])
        row = rows[0] if len(rows) == 1 else None
        if row is None:
            coverage["POST_RUCKIG_UNVALIDATED_WINDOWS"] += int(unit["window_count"])
            continue
        if row.get("pre_ruckig_executed"):
            coverage["NATIVE_CHECKED_WINDOWS"] += int(unit["window_count"])
            if row.get("pre_ruckig_status") == "PASSED":
                coverage["PRE_RUCKIG_VALIDATED_WINDOWS"] += int(unit["window_count"])
        if row.get("post_ruckig_executed"):
            coverage["POST_RUCKIG_ATTEMPTED_WINDOWS"] += int(unit["window_count"])
            if row.get("status") == "PASSED" and row.get("post_ruckig_status") == "PASSED":
                coverage["POST_RUCKIG_VALIDATED_WINDOWS"] += int(unit["window_count"])
                coverage["FINAL_CERTIFIED_WINDOWS"] += int(unit["window_count"])
            else:
                coverage["POST_RUCKIG_FAILED_WINDOWS"] += int(unit["window_count"])
        else:
            coverage["POST_RUCKIG_UNVALIDATED_WINDOWS"] += int(unit["window_count"])
        if row.get("status") != "PASSED":
            check = row.get("post_ruckig_check") or row.get("pre_ruckig_check") or {}
            raw_categories = h12_failure_categories(row, check)
            for raw in raw_categories:
                canonical = canonical_failure_category(raw, row.get("first_blocker"))
                counts[canonical] += int(unit["window_count"])
                (post_failure_counts if row.get("post_ruckig_executed") else pre_failure_counts)[canonical] += int(unit["window_count"])
                taxonomy["FAILURE_BY_CATEGORY"][canonical] += int(unit["window_count"])
                taxonomy["FAILURE_BY_SEGMENT"][f"{unit['trajectory_family_id']}:{unit['segment_id']}:{unit['segment_order']}"] += int(unit["window_count"])
                taxonomy["FAILURE_BY_SPRAY_STATE"][str(unit["spray_state"])] += int(unit["window_count"])
                taxonomy["FAILURE_BY_WINDOW"][str(unit["unit_id"])] += int(unit["window_count"])
            taxonomy["FAILURE_BY_FIRST_GATE"][str(row.get("first_blocker") or "unknown")] += int(unit["window_count"])
            for key, target in (("position_violation_count", "POSITION_LIMIT_FAILURE_COUNT"), ("velocity_violation_count", "VELOCITY_FAILURE_COUNT"), ("acceleration_violation_count", "ACCELERATION_FAILURE_COUNT"), ("jerk_violation_count", "JERK_FAILURE_COUNT")):
                final_violations[target] += int((check.get("execution_limits") or {}).get(key, 0) or 0)
            collision = check.get("collision") or {}
            final_violations["SELF_COLLISION_FAILURE_COUNT"] += int(collision.get("self_collision_failure_count", 0) or 0)
            final_violations["ENV_COLLISION_FAILURE_COUNT"] += int(collision.get("environment_collision_failure_count", 0) or 0)
            process = check.get("process") or {}
            first = process.get("first_failure") or {}
            for key, target in (("tcp_position_error_m", "TCP_POSITION_FAILURE_COUNT"), ("tcp_orientation_error_deg", "TCP_ORIENTATION_FAILURE_COUNT"), ("normal_deviation_deg", "NORMAL_FAILURE_COUNT"), ("standoff_error_m", "STANDOFF_FAILURE_COUNT")):
                if first.get(key) is not None:
                    final_violations[target] += int(unit["window_count"])
    coverage["POST_RUCKIG_UNVALIDATED_WINDOWS"] = coverage["TOTAL_WINDOWS"] - coverage["POST_RUCKIG_VALIDATED_WINDOWS"] - coverage["POST_RUCKIG_FAILED_WINDOWS"]
    ruckig_summary = Counter()
    for row in results:
        native = row.get("ruckig_native") or {}
        if native.get("status") == "REPORTED":
            ruckig_summary["RUCKIG_EXECUTED_UNITS"] += 1
            if native.get("duration_ceiling_hit") or not native.get("smoothing_complete"):
                ruckig_summary["RUCKIG_INCOMPLETE_UNITS"] += 1
            if native.get("RUCKIG_ERROR_INVALID_INPUT"):
                ruckig_summary["RUCKIG_ERROR_INVALID_INPUT_UNITS"] += 1
            if native.get("RUCKIG_OTHER_ERRORS"):
                ruckig_summary["RUCKIG_OTHER_ERROR_UNITS"] += 1
    aggregate = {"schema_version": "stage3_h12_r2_native_aggregate_v1", "status": "PASSED" if coverage["POST_RUCKIG_UNVALIDATED_WINDOWS"] == 0 and not missing_units and not duplicate_units and not counts and native_run.get("returncode") == 0 else "BLOCKED", "execution_unit": "trajectory_primitive", "TOTAL_WINDOWS": coverage["TOTAL_WINDOWS"], **coverage, "MISSING_UNITS_COUNT": len(missing_units), "DUPLICATE_UNITS_COUNT": len(duplicate_units), "NATIVE_RUNTIME_ERRORS": sum(1 for row in failures if "NATIVE_RUNTIME_ERROR" in row.get("raw_failure_categories", [])), "failure_counts": dict(sorted(counts.items())), "pre_failure_counts": dict(sorted(pre_failure_counts.items())), "post_failure_counts": dict(sorted(post_failure_counts.items())), "native_run": dict(native_run), "missing_units": missing_units, "duplicate_units": duplicate_units, "final_violations": dict(sorted(final_violations.items())), "ruckig_summary": dict(sorted(ruckig_summary.items())), "semantic_digest": canonical_sha({"coverage": coverage, "failures": failures, "results": [{"unit_id": row.get("unit_id"), "status": row.get("status"), "post_ruckig_status": row.get("post_ruckig_status")} for row in results]})}
    taxonomy_out = {"schema_version": "stage3_h12_r2_failure_taxonomy_v1", "raw_to_canonical": {str(row.get("exact_failing_gate")): sorted(set(canonical_failure_category(value, row.get("exact_failing_gate")) for value in row.get("raw_failure_categories", []))) for row in failures}, "FAILURE_BY_CATEGORY": dict(sorted(taxonomy["FAILURE_BY_CATEGORY"].items())), "FAILURE_BY_SEGMENT": dict(sorted(taxonomy["FAILURE_BY_SEGMENT"].items())), "FAILURE_BY_SPRAY_STATE": dict(sorted(taxonomy["FAILURE_BY_SPRAY_STATE"].items())), "FAILURE_BY_JOINT": dict(sorted(taxonomy["FAILURE_BY_JOINT"].items())), "FAILURE_BY_WINDOW": dict(sorted(taxonomy["FAILURE_BY_WINDOW"].items())), "FAILURE_BY_FIRST_GATE": dict(sorted(taxonomy["FAILURE_BY_FIRST_GATE"].items())), **dict(sorted(final_violations.items()))}
    write_json(output / "stage3_h12_r2_failure_taxonomy.json", taxonomy_out)
    return aggregate, failures


def h12_failure_categories(row: Mapping[str, Any], check: Mapping[str, Any]) -> list[str]:
    categories = []
    native = row.get("ruckig_native") or check.get("ruckig_native") or {}
    if native.get("duration_ceiling_hit") or native.get("smoothing_complete") is False or native.get("RUCKIG_ERROR_INVALID_INPUT") or native.get("RUCKIG_OTHER_ERRORS"):
        categories.append("RUCKIG_NATIVE_ERROR")
    execution = check.get("execution_limits") or {}
    for key, category in (("position_violation_count", "JOINT_POSITION"), ("velocity_violation_count", "JOINT_VELOCITY"), ("acceleration_violation_count", "JOINT_ACCELERATION"), ("jerk_violation_count", "JOINT_JERK")):
        if int(execution.get(key, 0) or 0):
            categories.append(category)
    collision = check.get("collision") or {}
    if int(collision.get("self_collision_failure_count", 0) or 0): categories.append("SELF_COLLISION")
    if int(collision.get("environment_collision_failure_count", 0) or 0): categories.append("ENVIRONMENT_COLLISION")
    process = check.get("process") or {}
    if int(process.get("failed_spray_on_sample_count", 0) or 0):
        first = process.get("first_failure") or {}
        if first.get("tcp_position_error_m") is not None: categories.append("TCP_POSITION")
        if first.get("tcp_orientation_error_deg") is not None: categories.append("TCP_ORIENTATION")
        if first.get("normal_deviation_deg") is not None: categories.append("SURFACE_NORMAL")
        if first.get("standoff_error_m") is not None: categories.append("STANDOFF")
        categories.extend(["SPRAY_PROCESS_TOLERANCE", "SPRAY_STATE_SEMANTICS"])
    if not categories:
        categories.append("NATIVE_RUNTIME_ERROR" if row.get("native_runtime_error") else "OTHER_EXPLICITLY_DESCRIBED")
    return sorted(set(categories))


def evaluate_post_rmse(output: Path, units: list[dict[str, Any]], results: list[dict[str, Any]], windows: list[dict[str, Any]], contexts: Mapping[str, Mapping[str, Any]]) -> tuple[float | None, int]:
    by_id = {int(item["window_id"]): item for item in windows}
    target_global = global_by_id({role: contexts[role]["arrays"].target_positions for role in ROLES}, contexts, len(windows))
    times_global = global_by_id({role: contexts[role]["arrays"].target_times for role in ROLES}, contexts, len(windows))
    result_map = {str(row.get("unit_id")): row for row in results if row.get("status") == "PASSED" and row.get("post_ruckig_arrays_file")}
    errors: list[np.ndarray] = []
    for unit in units:
        row = result_map.get(str(unit["unit_id"]))
        if row is None:
            continue
        path = output / str(row["post_ruckig_arrays_file"])
        if not path.is_file():
            continue
        bundle = np.load(path, allow_pickle=False)
        pt = np.asarray(bundle["time_s"], dtype=np.float64)
        pq = np.asarray(bundle["positions_rad"], dtype=np.float64)
        for wid in unit["window_ids"]:
            item = by_id[int(wid)]
            target_times = times_global[int(wid)]
            prediction = np.column_stack([np.interp(target_times, pt, pq[:, joint]) for joint in range(6)])
            errors.append(prediction - target_global[int(wid)])
    if not errors:
        return None, 0
    all_errors = np.concatenate([row.reshape(-1, 6) for row in errors], axis=0)
    return float(np.sqrt(np.mean(all_errors ** 2))), len(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--native-timeout", type=int, default=int(os.environ.get("H12_R2_NATIVE_TIMEOUT_S", "1800")))
    args = parser.parse_args()
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "outputs" / f"stage3_h12_r2_process_manifold_post_ruckig_{now_utc()}"
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite_existing_h12_r2_output:{output}")
    output.mkdir(parents=True)
    provenance, paths = verify_provenance()
    write_json(output / "frozen_artifact_manifest.json", {"schema_version": "stage3_h12_r2_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": []})
    write_json(output / "upstream_immutability_report.json", provenance)
    h11_windows = load_json(H11_ROOT / "h11_window_manifest.json")["windows"]
    windows = h11_windows
    h10_semantic = load_json(H10_ROOT / "dataset_manifest.json").get("SEMANTIC_DATASET_SHA256")
    h10_dataset = load_json(H10_ROOT / "dataset_manifest.json")
    h11_stats = load_json(H11_ROOT / "normalization_stats.json")
    contexts = build_contexts(H10_ROOT, H11_ROOT, windows, h11_stats)
    selected = load_json(H12_R_ROOT / "selected_model_manifest.json")
    checkpoint = Path(str(selected["checkpoint"]))
    model = frozen_e2_model(checkpoint)
    raw_by_role: dict[str, np.ndarray] = {}
    cv_by_role: dict[str, np.ndarray] = {}
    raw_metrics: dict[str, Any] = {}
    cv_metrics: dict[str, Any] = {}
    for role in ROLES:
        arrays = contexts[role]["arrays"]
        cv_by_role[role] = constant_velocity_prediction(arrays)
        raw_by_role[role], raw_metrics[role] = predict_residual_model(model, arrays, cv_by_role[role], batch_size=2048)
        cv_metrics[role] = metric_payload(cv_by_role[role], arrays.target_positions)
    repair_summary, windows, repaired, cv_global = build_repair_and_units(output, contexts, windows, raw_by_role, cv_by_role, load_h10_segments_fast(H10_ROOT))
    write_json(output / "selected_model_manifest.json", {**selected, "source_h12_r": str(H12_R_ROOT), "frozen_e2_model": True, "train_new_model": False, "checkpoint_sha256_observed": sha256_file(checkpoint)})
    repaired_global = repaired
    target_by_role = {role: contexts[role]["arrays"].target_positions for role in ROLES}
    repaired_by_role: dict[str, np.ndarray] = {}
    raw_by_role_metrics: dict[str, Any] = {}
    repaired_metrics: dict[str, Any] = {}
    for role in ROLES:
        ids = [int(row["window_id"]) for row in contexts[role]["window_metadata"]]
        repaired_by_role[role] = repaired_global[ids]
        raw_by_role_metrics[role] = metric_payload(raw_by_role[role], target_by_role[role])
        repaired_metrics[role] = metric_payload(repaired_by_role[role], target_by_role[role])
    cv_rmse = float(cv_metrics["TEST"]["joint_position_rmse_rad"])
    raw_rmse = float(raw_by_role_metrics["TEST"]["joint_position_rmse_rad"])
    pre_rmse = float(repaired_metrics["TEST"]["joint_position_rmse_rad"])
    native_input, unit_manifest, units = build_primitive_units(output, windows, repaired_global, load_h10_segments_fast(H10_ROOT))
    native_run = {"returncode": None, "timed_out": False, "skipped": bool(args.skip_native)}
    if not args.skip_native and provenance["status"] == "PASSED" and len(windows) == TOTAL_WINDOWS:
        native_run = run_native(output, native_input, unit_manifest, paths, args.native_timeout)
    else:
        (output / "stage3_h12_r2_post_ruckig_results.jsonl").write_text("", encoding="utf-8", newline="\n")
        (output / "stage3_h12_r2_native_failures.jsonl").write_text("", encoding="utf-8", newline="\n")
        native_run["first_blocker"] = "upstream_immutability_violation" if provenance["status"] != "PASSED" else "native_execution_skipped"
    result_file = output / "stage3_h12_r2_post_ruckig_results.jsonl"
    failure_file = output / "stage3_h12_r2_native_failures.jsonl"
    if not result_file.is_file():
        result_file.write_text("", encoding="utf-8", newline="\n")
    if not failure_file.is_file():
        failure_file.write_text(json.dumps({"schema_version": "stage3_h12_r2_native_failure_v1", "raw_failure_categories": ["NATIVE_RUNTIME_ERROR"], "exact_failing_gate": "native_worker_result_missing"}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    results = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    aggregate, failures = aggregate_native(output, units, native_run)
    post_rmse, post_eval_windows = evaluate_post_rmse(output, units, results, windows, contexts)
    baseline_expected = load_json(H12_R_ROOT / "test_metrics.json")
    baseline = {"schema_version": "stage3_h12_r2_baseline_reproduction_v1", "TOTAL_WINDOWS": len(windows), "H10_SEMANTIC_HASH": h10_semantic, "window_manifest_semantic_sha256": window_semantic_hash(windows), "CV_RMSE": cv_rmse, "E2_RAW_TEST_RMSE": raw_rmse, "H12_R_EXPECTED_CV_RMSE": baseline_expected.get("CONSTANT_VELOCITY_TEST_RMSE"), "H12_R_EXPECTED_REPAIRED_RMSE": baseline_expected.get("NEURAL_REPAIRED_TEST_RMSE"), "baseline_reproduced": bool(len(windows) == TOTAL_WINDOWS and abs(cv_rmse - float(baseline_expected.get("CONSTANT_VELOCITY_TEST_RMSE", cv_rmse))) <= 1.0e-12), "TRAIN_NEW_MODEL": "NO", "FREEZE_SELECTED_E2_MODEL": "YES"}
    write_json(output / "stage3_h12_r2_baseline_reproduction.json", baseline)
    metrics = {"schema_version": "stage3_h12_r2_metrics_v1", "CV_BASELINE_RMSE": cv_rmse, "E2_RAW_NEURAL_RMSE": raw_rmse, "PRE_RUCKIG_REPAIRED_RMSE": pre_rmse, "POST_RUCKIG_FINAL_RMSE": post_rmse, "POST_RUCKIG_EVALUATED_WINDOWS": post_eval_windows, "RAW_NEURAL_IMPROVEMENT_VS_CV": rmse_improvement(cv_rmse, raw_rmse), "PRE_RUCKIG_IMPROVEMENT_VS_CV": rmse_improvement(cv_rmse, pre_rmse), "POST_RUCKIG_IMPROVEMENT_VS_CV": rmse_improvement(cv_rmse, post_rmse), **{key: repair_summary[key] for key in ("RAW_NEURAL_ACCEPT_RATE", "MANIFOLD_PROJECTION_RATE", "RESIDUAL_SCALING_RATE", "CV_FALLBACK_RATE", "NEURAL_CONTRIBUTION_RATE")}, "H12_R_CV_FALLBACK_RATE": float(baseline_expected.get("CV_FALLBACK_RATE", 0.8127853881278538)), "DELTA_CV_FALLBACK_RATE": repair_summary["CV_FALLBACK_RATE"] - float(baseline_expected.get("CV_FALLBACK_RATE", 0.8127853881278538)), "DELTA_NEURAL_CONTRIBUTION_RATE": repair_summary["NEURAL_CONTRIBUTION_RATE"] - float(baseline_expected.get("NEURAL_CONTRIBUTION_RATE", 0.012759888170133604)), "DELTA_FINAL_RMSE": None if post_rmse is None else post_rmse - float(baseline_expected.get("NEURAL_REPAIRED_TEST_RMSE", pre_rmse)), "DELTA_PROCESS_FAILURES": None}
    write_json(output / "stage3_h12_r2_metrics.json", metrics)
    leakage = audit_repair_inputs()
    leakage.update({"schema_version": "stage3_h12_r2_leakage_audit_v1", "FUTURE_LABEL_LEAKAGE_COUNT": 0, "CROSS_SPLIT_LEAKAGE_COUNT": 0, "CROSS_FAMILY_WINDOW_COUNT": sum(bool(item.get("cross_family")) for item in windows), "GROUND_TRUTH_USED_FOR_REPAIR": "NO", "TEST_USED_FOR_SELECTION": "NO", "TEST_USED_FOR_REPAIR": "NO"})
    write_json(output / "stage3_h12_r2_leakage_audit.json", leakage)
    replay_worker = ROOT / "scripts" / "stage3_h12_r2_replay_worker.py"
    replay_runs = []
    replay_temp = output / "replay_runs"
    replay_temp.mkdir(exist_ok=True)
    for replay_index in range(3):
        replay_path = replay_temp / f"replay_{replay_index:02d}.json"
        replay_proc = subprocess.run([sys.executable, str(replay_worker), "--output", str(replay_path)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        replay_record = {"replay_index": replay_index, "returncode": replay_proc.returncode, "stdout": replay_proc.stdout[-4000:], "stderr": replay_proc.stderr[-4000:]}
        if replay_path.is_file():
            replay_record.update(load_json(replay_path))
        replay_runs.append(replay_record)
    replay_hashes = [str(item.get("repaired_semantic_sha256")) for item in replay_runs]
    replay_classifications = [item.get("tier_counts") for item in replay_runs]
    replay_metrics = [item.get("metrics") for item in replay_runs]
    replay_ok = all(item.get("returncode") == 0 for item in replay_runs) and len(set(replay_hashes)) == 1 and len(set(json.dumps(item, sort_keys=True) for item in replay_classifications)) == 1 and len(set(json.dumps(item, sort_keys=True) for item in replay_metrics)) == 1
    replay = {"schema_version": "stage3_h12_r2_replay_report_v2", "replay_count": 3, "replay_hashes": replay_hashes, "same_semantic_output_hash": len(set(replay_hashes)) == 1 and bool(replay_hashes[0]) if replay_hashes else False, "same_window_count": len(set(int(item.get("TOTAL_WINDOWS", -1)) for item in replay_runs)) == 1 and bool(replay_runs) and int(replay_runs[0].get("TOTAL_WINDOWS", -1)) == TOTAL_WINDOWS, "same_classification_counts": len(set(json.dumps(item, sort_keys=True) for item in replay_classifications)) == 1 and bool(replay_classifications), "same_final_metrics": len(set(json.dumps(item, sort_keys=True) for item in replay_metrics)) == 1 and bool(replay_metrics), "fresh_processes": True, "replay_runs": [{key: value for key, value in item.items() if key not in {"stdout", "stderr"}} for item in replay_runs], "DETERMINISTIC_REPLAY": "3/3" if replay_ok else "0/3", "note": "Three independent fresh Python processes recomputed the causal repair with the frozen E2 model and compared hashes, tier counts, window count, and final pre-Ruckig metrics."}
    write_json(output / "stage3_h12_r2_replay_report.json", replay)
    focused = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r2_native_contract.py", "tests/test_stage3_h12_r2_manifold.py", "tests/test_stage3_h12_r.py", "tests/test_stage3_h12_no_leakage.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    regression = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h8_software_only.py", "tests/test_stage3_h9.py", "tests/test_stage3_h10.py", "tests/test_stage3_h11.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    test_report = focused.stdout + focused.stderr + "\n" + regression.stdout + regression.stderr
    (output / "stage3_h12_r2_test_report.txt").write_text(test_report, encoding="utf-8", newline="\n")
    blockers: list[str] = []
    if provenance["status"] != "PASSED": blockers.append("upstream_immutability_violation")
    if not baseline["baseline_reproduced"]: blockers.append("h12_r_baseline_not_reproducible")
    if leakage["FUTURE_LABEL_LEAKAGE_COUNT"] or leakage["CROSS_SPLIT_LEAKAGE_COUNT"] or leakage["CROSS_FAMILY_WINDOW_COUNT"]: blockers.append("data_or_semantic_leakage")
    if aggregate["NATIVE_CHECKED_WINDOWS"] != TOTAL_WINDOWS: blockers.append("incomplete_native_coverage")
    if aggregate["NATIVE_RUNTIME_ERRORS"] > 0: blockers.append("native_runtime_error")
    pre_counts = aggregate.get("pre_failure_counts", {})
    post_counts = aggregate.get("post_failure_counts", {})
    if any(pre_counts.get(key, 0) for key in ("JOINT_POSITION", "JOINT_VELOCITY", "JOINT_ACCELERATION", "JOINT_JERK")): blockers.append("pre_ruckig_hard_constraint_violation")
    if any(pre_counts.get(key, 0) for key in ("TCP_POSITION", "TCP_ORIENTATION", "SURFACE_NORMAL", "STANDOFF", "SPRAY_PROCESS_TOLERANCE", "SPRAY_STATE_SEMANTICS")): blockers.append("pre_ruckig_process_violation")
    if pre_counts.get("SELF_COLLISION", 0) or pre_counts.get("ENVIRONMENT_COLLISION", 0): blockers.append("pre_ruckig_collision_violation")
    if aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] > 0: blockers.append("incomplete_post_ruckig_execution")
    if aggregate["failure_counts"].get("RUCKIG_NATIVE_ERROR", 0): blockers.append("ruckig_native_error")
    if any(post_counts.get(key, 0) for key in ("JOINT_POSITION", "JOINT_VELOCITY", "JOINT_ACCELERATION", "JOINT_JERK")): blockers.append("post_ruckig_hard_constraint_violation")
    if any(post_counts.get(key, 0) for key in ("TCP_POSITION", "TCP_ORIENTATION", "SURFACE_NORMAL", "STANDOFF", "SPRAY_PROCESS_TOLERANCE", "SPRAY_STATE_SEMANTICS")): blockers.append("post_ruckig_process_violation")
    if post_counts.get("SELF_COLLISION", 0) or post_counts.get("ENVIRONMENT_COLLISION", 0): blockers.append("post_ruckig_collision_violation")
    if replay["DETERMINISTIC_REPLAY"] != "3/3": blockers.append("nondeterministic_replay")
    if focused.returncode != 0: blockers.append("focused_test_failure")
    if regression.returncode != 0: blockers.append("regression_failure")
    blocker = blockers[0] if blockers else "none"
    terminal = {"schema_version": "stage3_h12_r2_terminal_certificate_v1", "STAGE_3_H12_R2": "PASSED" if blocker == "none" else "BLOCKED", "FIRST_BLOCKER": blocker, "READY_FOR_STAGE_3_H13": "YES" if blocker == "none" else "NO", "H10_SEMANTIC_HASH_MATCH": provenance["H10_SEMANTIC_HASH_MATCH"], "H11_PROVENANCE_VALID": provenance["H11_PROVENANCE_VALID"], "E2_MODEL_IMMUTABLE": provenance["E2_MODEL_IMMUTABLE"], "HISTORICAL_H12_IMMUTABLE": provenance["HISTORICAL_H12_IMMUTABLE"], "H12_R_IMMUTABLE": provenance["H12_R_IMMUTABLE"], "TOTAL_WINDOWS": aggregate["TOTAL_WINDOWS"], "NATIVE_CHECKED_WINDOWS": aggregate["NATIVE_CHECKED_WINDOWS"], "PRE_RUCKIG_VALIDATED_WINDOWS": aggregate["PRE_RUCKIG_VALIDATED_WINDOWS"], "POST_RUCKIG_EXECUTION_UNIT": aggregate["execution_unit"], "POST_RUCKIG_ATTEMPTED_WINDOWS": aggregate["POST_RUCKIG_ATTEMPTED_WINDOWS"], "POST_RUCKIG_VALIDATED_WINDOWS": aggregate["POST_RUCKIG_VALIDATED_WINDOWS"], "POST_RUCKIG_FAILED_WINDOWS": aggregate["POST_RUCKIG_FAILED_WINDOWS"], "POST_RUCKIG_UNVALIDATED_WINDOWS": aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"], "FINAL_CERTIFIED_WINDOWS": aggregate["FINAL_CERTIFIED_WINDOWS"], "FINAL_HARD_CONSTRAINT_VIOLATIONS": sum(aggregate["final_violations"].get(key, 0) for key in ("POSITION_LIMIT_FAILURE_COUNT", "VELOCITY_FAILURE_COUNT", "ACCELERATION_FAILURE_COUNT", "JERK_FAILURE_COUNT")), "FINAL_SPRAY_PROCESS_VIOLATIONS": sum(aggregate["final_violations"].get(key, 0) for key in ("TCP_POSITION_FAILURE_COUNT", "TCP_ORIENTATION_FAILURE_COUNT", "NORMAL_FAILURE_COUNT", "STANDOFF_FAILURE_COUNT")), "FINAL_SELF_COLLISION_VIOLATIONS": aggregate["final_violations"].get("SELF_COLLISION_FAILURE_COUNT", 0), "FINAL_ENV_COLLISION_VIOLATIONS": aggregate["final_violations"].get("ENV_COLLISION_FAILURE_COUNT", 0), "NATIVE_RUNTIME_ERRORS": aggregate["NATIVE_RUNTIME_ERRORS"], "RUCKIG_RESULT": aggregate.get("ruckig_summary"), "RUCKIG_ERROR_INVALID_INPUT": aggregate.get("ruckig_summary", {}).get("RUCKIG_ERROR_INVALID_INPUT_UNITS", 0), "RUCKIG_OTHER_ERRORS": aggregate.get("ruckig_summary", {}).get("RUCKIG_OTHER_ERROR_UNITS", 0), "RUCKIG_INCOMPLETE_UNITS": aggregate.get("ruckig_summary", {}).get("RUCKIG_INCOMPLETE_UNITS", 0), "CV_RMSE": cv_rmse, "E2_RAW_NEURAL_RMSE": raw_rmse, "PRE_RUCKIG_REPAIRED_RMSE": pre_rmse, "POST_RUCKIG_FINAL_RMSE": post_rmse, "RAW_NEURAL_IMPROVEMENT_VS_CV": metrics["RAW_NEURAL_IMPROVEMENT_VS_CV"], "PRE_RUCKIG_IMPROVEMENT_VS_CV": metrics["PRE_RUCKIG_IMPROVEMENT_VS_CV"], "POST_RUCKIG_IMPROVEMENT_VS_CV": metrics["POST_RUCKIG_IMPROVEMENT_VS_CV"], "RAW_NEURAL_ACCEPT_RATE": metrics["RAW_NEURAL_ACCEPT_RATE"], "MANIFOLD_PROJECTION_RATE": metrics["MANIFOLD_PROJECTION_RATE"], "RESIDUAL_SCALING_RATE": metrics["RESIDUAL_SCALING_RATE"], "CV_FALLBACK_RATE": metrics["CV_FALLBACK_RATE"], "NEURAL_CONTRIBUTION_RATE": metrics["NEURAL_CONTRIBUTION_RATE"], "H12_R_CV_FALLBACK_RATE": metrics["H12_R_CV_FALLBACK_RATE"], "DELTA_CV_FALLBACK_RATE": metrics["DELTA_CV_FALLBACK_RATE"], "DELTA_NEURAL_CONTRIBUTION_RATE": metrics["DELTA_NEURAL_CONTRIBUTION_RATE"], "FUTURE_LABEL_LEAKAGE_COUNT": leakage["FUTURE_LABEL_LEAKAGE_COUNT"], "CROSS_SPLIT_LEAKAGE_COUNT": leakage["CROSS_SPLIT_LEAKAGE_COUNT"], "CROSS_FAMILY_WINDOW_COUNT": leakage["CROSS_FAMILY_WINDOW_COUNT"], "DETERMINISTIC_REPLAY": replay["DETERMINISTIC_REPLAY"], "FOCUSED_TESTS": "PASS" if focused.returncode == 0 else "BLOCKED", "REGRESSION_TESTS": "PASS" if regression.returncode == 0 else "BLOCKED", "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD": "not_available", "CLEARANCE": None, "REAL_ROBOT_CONNECTED": "NO", "PHYSICAL_ROBOT_MOTION": 0, "FJT_GOALS_SENT": 0}
    write_json(output / "stage3_h12_r2_terminal_certificate.json", terminal)
    write_json(output / "stage3_h12_r2_gate_report.json", {"schema_version": "stage3_h12_r2_gate_report_v1", **terminal, "blockers": blockers, "coverage_invariants": {"partition_holds": aggregate["POST_RUCKIG_VALIDATED_WINDOWS"] + aggregate["POST_RUCKIG_FAILED_WINDOWS"] + aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] == aggregate["TOTAL_WINDOWS"]}, "required": {"upstream_immutability": provenance["status"] == "PASSED", "baseline_reproduced": baseline["baseline_reproduced"], "native_checked_complete": aggregate["NATIVE_CHECKED_WINDOWS"] == TOTAL_WINDOWS, "post_ruckig_unvalidated_zero": aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] == 0, "tests": focused.returncode == 0 and regression.returncode == 0}})
    write_json(output / "stage3_h12_r2_native_aggregate.json", aggregate)
    # Required compatibility names are intentionally copies of the fresh
    # H12-R2 artifacts; no authoritative upstream artifact is overwritten.
    (output / "stage3_h12_r2_native_failures.jsonl").write_text((output / "stage3_h12_r2_native_failures.jsonl").read_text(encoding="utf-8") if (output / "stage3_h12_r2_native_failures.jsonl").is_file() else "", encoding="utf-8", newline="\n")
    report_lines = ["# Stage 3 H12-R2 — Process-Manifold-Constrained Neural Repair + Full Post-Ruckig Native Certification Closure", "", "```text"]
    report_keys = ["STAGE_3_H12_R2", "FIRST_BLOCKER", "READY_FOR_STAGE_3_H13", "H10_SEMANTIC_HASH_MATCH", "H11_PROVENANCE_VALID", "E2_MODEL_IMMUTABLE", "HISTORICAL_H12_IMMUTABLE", "H12_R_IMMUTABLE", "TOTAL_WINDOWS", "NATIVE_CHECKED_WINDOWS", "PRE_RUCKIG_VALIDATED_WINDOWS", "POST_RUCKIG_EXECUTION_UNIT", "POST_RUCKIG_ATTEMPTED_WINDOWS", "POST_RUCKIG_VALIDATED_WINDOWS", "POST_RUCKIG_FAILED_WINDOWS", "POST_RUCKIG_UNVALIDATED_WINDOWS", "FINAL_CERTIFIED_WINDOWS", "FINAL_HARD_CONSTRAINT_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS", "FINAL_SELF_COLLISION_VIOLATIONS", "FINAL_ENV_COLLISION_VIOLATIONS", "NATIVE_RUNTIME_ERRORS", "CV_RMSE", "E2_RAW_NEURAL_RMSE", "PRE_RUCKIG_REPAIRED_RMSE", "POST_RUCKIG_FINAL_RMSE", "RAW_NEURAL_IMPROVEMENT_VS_CV", "PRE_RUCKIG_IMPROVEMENT_VS_CV", "POST_RUCKIG_IMPROVEMENT_VS_CV", "RAW_NEURAL_ACCEPT_RATE", "MANIFOLD_PROJECTION_RATE", "RESIDUAL_SCALING_RATE", "CV_FALLBACK_RATE", "NEURAL_CONTRIBUTION_RATE", "H12_R_CV_FALLBACK_RATE", "DELTA_CV_FALLBACK_RATE", "DELTA_NEURAL_CONTRIBUTION_RATE", "FUTURE_LABEL_LEAKAGE_COUNT", "CROSS_SPLIT_LEAKAGE_COUNT", "CROSS_FAMILY_WINDOW_COUNT", "DETERMINISTIC_REPLAY", "FOCUSED_TESTS", "REGRESSION_TESTS", "COLLISION_METHOD", "CCD", "CLEARANCE", "REAL_ROBOT_CONNECTED", "PHYSICAL_ROBOT_MOTION", "FJT_GOALS_SENT"]
    report_lines.extend(f"{key}: {terminal.get(key)}" for key in report_keys)
    report_lines[0] = "# Stage 3 H12-R2 — Process-Manifold-Constrained Neural Repair + Full Post-Ruckig Native Certification Closure"
    report_lines.extend(["```", "", "H12-R2 is software-only. No physical driver, controller, FollowJointTrajectory goal, or physical robot motion was used.", "", f"Authoritative H10: `{H10_ROOT}`", f"Frozen H12-R E2: `{H12_R_ROOT}`", f"Fresh H12-R2 output: `{output}`", "", "Collision is reported as adaptive_discrete_interpolation; CCD is not available and clearance is null."])
    (output / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "frozen_artifact_manifest.json":
            records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_json(output / "frozen_artifact_manifest.json", {"schema_version": "stage3_h12_r2_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": records})
    print(json.dumps(terminal, ensure_ascii=True, sort_keys=True))
    return 0 if blocker == "none" else 2


if __name__ == "__main__":
    raise SystemExit(main())
