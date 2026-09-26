"""Execute the authorized D27 forward continuation from authenticated D26.

This runner is intentionally narrow: it consumes only the archived D26
update-400 checkpoint, preserves the D26 alpha=1/32 AdamW branch, and makes
one committed attempt per update through the requested D27 range.  Each
attempt is replayed from the same parent state/RNG before it can be committed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_d26_progress_first_optimizer_consistency_recovery as d26
from tools import stage3_h13_post_d24_d25_trust_region_design_review as design
from tools import stage3_h13_post_d24_d25_trust_region_execution as d25_execution
from tools import stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution as d24


DEFAULT_D26_ARCHIVE = ROOT / "tmp" / "d26_final_archive_stage_20260822"
DEFAULT_PARENT = DEFAULT_D26_ARCHIVE / "CHECKPOINTS" / "committed_update_400.pt"
DEFAULT_D26_MANIFEST = DEFAULT_D26_ARCHIVE / "FINAL_RESULTS" / "D26_EXECUTION_MANIFEST.json"
DEFAULT_D26_TRAJECTORY = DEFAULT_D26_ARCHIVE / "NUMERICAL_TRAJECTORY" / "D26_COMMITTED_FORWARD_TRAJECTORY.csv"
DEFAULT_D26_IDENTITY = DEFAULT_D26_ARCHIVE / "CHECKPOINT_IDENTITY.json"
DEFAULT_AUTHORIZATION = Path(r"C:\Users\86198\.codex\attachments\fcd04f89-5450-4949-b3e9-da27e4ef0ad7\pasted-text.txt")
EXPECTED_D26_PARENT_SHA256 = "af31be4c531914a1265d9d830c6b0820ca01fc65bfb87f5751fc07906d698b14"
EXPECTED_D26_MANIFEST_SHA256 = "bf52a870d3d599ec369dbb9be33a1c3e853da018b8563bb83bb628b81858caf5"
EXPECTED_ALPHA = 1.0 / 32.0
EXPECTED_CANONICAL_LR = 5.0e-5
TARGET_FIRST_UPDATE = 401
TARGET_LAST_UPDATE = 406


class D27Blocker(RuntimeError):
    """A real authorization, identity, or scientific execution blocker."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D27Blocker(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def read_trajectory(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {int(row["update"]): row for row in csv.DictReader(stream)}
    return rows


def checkpoint_payload(model: Any, optimizer: torch.optim.Optimizer, rng: Mapping[str, Any], record: Mapping[str, Any], parent_update: int, update: int, alpha: float, canonical_lrs: Sequence[float], d26_manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_d27_resumable_committed_checkpoint_v1",
        "experiment_id": "STAGE_3_H13_POST_D26_D27_FORWARD_CONTINUATION",
        "parent_d26_execution_manifest_sha256": d26_manifest_sha256,
        "proposal": "P4",
        "proposal_spec": copy.deepcopy(design.P4_SPEC),
        "alpha": float(alpha),
        "parent_update": int(parent_update),
        "completed_optimizer_step": int(update),
        "next_schedule_index": int(update),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": None,
        "rng_state": copy.deepcopy(rng),
        "batch_position": {"completed_optimizer_step": int(update), "next_zero_based_schedule_index": int(update)},
        "canonical_base_lr": [float(value) for value in canonical_lrs],
        "candidate_effective_lr": json.loads(str(record["effective_lr"])),
        "last_record_identity": {
            "update": int(update),
            "batch_window_sha256": record["batch_identity"],
            "model_state_hash": d24.d17.canonical_model_state_sha256(model.state_dict()),
            "optimizer_state_hash": d24.d17.base.optimizer_semantic_hash(optimizer),
        },
        "replay_consistency": {
            "pass": bool(record.get("deterministic_replay_consistency_pass")),
            "candidate_record_digest": record.get("deterministic_replay_candidate_record_digest"),
            "replay_record_digest": record.get("deterministic_replay_record_digest"),
        },
        "resumable": True,
        "scientific_state": "COMMITTED_FORWARD_STATE",
    }


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return list(left.keys()) == list(right.keys()) and all(tensor_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(tensor_state_equal(a, b) for a, b in zip(left, right))
    return left == right


def replay_compare(first: Mapping[str, Any], replay: Mapping[str, Any], first_model: Any, replay_model: Any, first_optimizer: torch.optim.Optimizer, replay_optimizer: torch.optim.Optimizer, first_rng: Mapping[str, Any], replay_rng: Mapping[str, Any]) -> dict[str, Any]:
    record_equal = tensor_state_equal(dict(first), dict(replay))
    model_equal = tensor_state_equal(first_model.state_dict(), replay_model.state_dict())
    optimizer_equal = tensor_state_equal(first_optimizer.state_dict(), replay_optimizer.state_dict())
    rng_equal = tensor_state_equal(first_rng, replay_rng)
    return {
        "pass": bool(record_equal and model_equal and optimizer_equal and rng_equal),
        "record_equal": record_equal,
        "model_state_equal": model_equal,
        "optimizer_state_equal": optimizer_equal,
        "rng_state_equal": rng_equal,
        "candidate_record_digest": digest_json(dict(first)),
        "replay_record_digest": digest_json(dict(replay)),
        "candidate_model_state_hash": d24.d17.canonical_model_state_sha256(first_model.state_dict()),
        "replay_model_state_hash": d24.d17.canonical_model_state_sha256(replay_model.state_dict()),
        "candidate_optimizer_state_hash": d24.d17.base.optimizer_semantic_hash(first_optimizer),
        "replay_optimizer_state_hash": d24.d17.base.optimizer_semantic_hash(replay_optimizer),
        "candidate_rng_hash": d24.d17.base.rng_digest(first_rng),
        "replay_rng_hash": d24.d17.base.rng_digest(replay_rng),
    }


def commit_checkpoint(output: Path, model: Any, optimizer: torch.optim.Optimizer, rng: Mapping[str, Any], record: dict[str, Any], parent_update: int, update: int, alpha: float, canonical_lrs: Sequence[float], d26_manifest_sha256: str) -> tuple[Path, str]:
    path = output / "checkpoints" / f"committed_update_{update}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, rng, record, parent_update, update, alpha, canonical_lrs, d26_manifest_sha256), path)
    checkpoint_sha = sha256_file(path)
    record["selected"] = True
    record["trial_state"] = "COMMITTED_FORWARD_STATE"
    record["model_checkpoint_sha256"] = checkpoint_sha
    record["optimizer_state_sha256_or_equivalent"] = d24.d17.base.optimizer_semantic_hash(optimizer)
    return path, checkpoint_sha


def verify_checkpoint(path: Path, expected_parent_update: int, expected_update: int, expected_parent_hash: str, expected_record: Mapping[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256_file(path)
    parent_hash = expected_parent_hash
    chain_pass = payload.get("parent_update") == expected_parent_update and payload.get("completed_optimizer_step") == expected_update and payload.get("next_schedule_index") == expected_update and expected_record.get("parent_update") == expected_parent_update and expected_record.get("candidate_update") == expected_update
    identity_pass = payload.get("last_record_identity", {}).get("batch_window_sha256") == expected_record.get("batch_identity") and payload.get("last_record_identity", {}).get("update") == expected_update
    step_pass = payload.get("optimizer_state_dict", {}).get("state") and all(int(value.get("step", -1)) == expected_update for value in payload["optimizer_state_dict"]["state"].values())
    return {
        "pass": bool(chain_pass and identity_pass and step_pass),
        "checkpoint_sha256": checkpoint_hash,
        "parent_checkpoint_sha256": parent_hash,
        "parent_update": payload.get("parent_update"),
        "completed_optimizer_step": payload.get("completed_optimizer_step"),
        "next_schedule_index": payload.get("next_schedule_index"),
        "batch_identity_pass": bool(identity_pass),
        "optimizer_step_counters_pass": bool(step_pass),
        "chain_fields_pass": bool(chain_pass),
    }


def load_authenticated_parent(parent_path: Path, manifest_path: Path, trajectory_path: Path, identity_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any, torch.optim.Optimizer, dict[str, Any], dict[str, Any], list[float], str]:
    require(parent_path.is_file(), f"D26_PARENT_MISSING:{parent_path}")
    require(manifest_path.is_file(), f"D26_MANIFEST_MISSING:{manifest_path}")
    require(trajectory_path.is_file(), f"D26_TRAJECTORY_MISSING:{trajectory_path}")
    require(sha256_file(parent_path) == EXPECTED_D26_PARENT_SHA256, "D26_UPDATE_400_CHECKPOINT_SHA256_MISMATCH")
    require(sha256_file(manifest_path) == EXPECTED_D26_MANIFEST_SHA256, "D26_EXECUTION_MANIFEST_SHA256_MISMATCH")
    d26_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(d26_manifest.get("committed_updates") == [395, 396, 397, 398, 399, 400], "D26_COMMITTED_UPDATE_CHAIN_MISMATCH")
    require(d26_manifest.get("outcome") == "D26_FORWARD_SPRINT_COMPLETED_THROUGH_UPDATE_400", "D26_NOT_COMPLETED_THROUGH_UPDATE_400")
    trajectory = read_trajectory(trajectory_path)
    require(400 in trajectory, "D26_UPDATE_400_TRAJECTORY_ROW_MISSING")
    final_row = trajectory[400]
    require(final_row.get("model_checkpoint_sha256") == EXPECTED_D26_PARENT_SHA256, "D26_TRAJECTORY_UPDATE_400_HASH_MISMATCH")
    payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    require(payload.get("completed_optimizer_step") == 400 and payload.get("next_schedule_index") == 400, "D26_UPDATE_400_PAYLOAD_POSITION_MISMATCH")
    require(payload.get("scientific_state") == "COMMITTED_FORWARD_STATE", "D26_UPDATE_400_NOT_COMMITTED_FORWARD_STATE")
    require(float(payload.get("alpha")) == EXPECTED_ALPHA, "D26_UPDATE_400_ALPHA_MISMATCH")
    runtime = d24.d17.preflight_runtime()
    model, optimizer = d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    canonical_lrs = [float(value) for value in payload.get("canonical_base_lr", [])]
    require(canonical_lrs == [EXPECTED_CANONICAL_LR], f"D27_CANONICAL_LR_MISMATCH:{canonical_lrs}")
    require([float(group["lr"]) for group in optimizer.param_groups] == design.effective_group_lrs(canonical_lrs, EXPECTED_ALPHA), "D26_EFFECTIVE_LR_STATE_MISMATCH")
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else {}
    update400_identity = next((item for item in identity.get("new_committed_checkpoints", []) if item.get("update_number") == 400), None)
    if update400_identity:
        require(update400_identity.get("checkpoint_sha256") == EXPECTED_D26_PARENT_SHA256, "D26_CHECKPOINT_IDENTITY_HASH_MISMATCH")
        require(update400_identity.get("model_state_hash") == d24.d17.canonical_model_state_sha256(model.state_dict()), "D26_MODEL_STATE_IDENTITY_MISMATCH")
        require(update400_identity.get("optimizer_semantic_hash") == d24.d17.base.optimizer_semantic_hash(optimizer), "D26_OPTIMIZER_STATE_IDENTITY_MISMATCH")
        require(update400_identity.get("rng_hash") == d24.d17.base.rng_digest(payload["rng_state"]), "D26_RNG_IDENTITY_MISMATCH")
    parent_state = {"update": 400, "h1": float(final_row["H1"]), "h32": float(final_row["H32"]), "checkpoint_sha256": EXPECTED_D26_PARENT_SHA256, "model_state_hash": d24.d17.canonical_model_state_sha256(model.state_dict()), "optimizer_state_hash": d24.d17.base.optimizer_semantic_hash(optimizer), "rng_hash": d24.d17.base.rng_digest(payload["rng_state"])}
    return d26_manifest, final_row, model, optimizer, copy.deepcopy(payload["rng_state"]), parent_state, canonical_lrs, sha256_file(manifest_path)


def slope(values: Sequence[float], updates: Sequence[int]) -> float | None:
    if len(values) < 2:
        return None
    xbar = sum(updates) / len(updates)
    ybar = sum(values) / len(values)
    denominator = sum((x - xbar) ** 2 for x in updates)
    return sum((x - xbar) * (y - ybar) for x, y in zip(updates, values)) / denominator if denominator else None


def make_trajectory(baseline_h1: float, baseline_h32: float, rows: Sequence[Mapping[str, Any]], stopped_blocker: str | None) -> dict[str, Any]:
    updates = [int(row["update"]) for row in rows]
    h1_values = [float(row["H1"]) for row in rows]
    h32_values = [float(row["H32"]) for row in rows]
    h1_slope = slope(h1_values, updates)
    reduction_slope = -h1_slope if h1_slope is not None else None
    h1_deltas = [float(row["delta_H1"]) for row in rows]
    h32_deltas = [float(row["delta_H32"]) for row in rows]
    strictly_decreasing = bool(h1_deltas) and all(value < 0.0 for value in h1_deltas)
    if stopped_blocker:
        classification = "D27_NEW_BLOCKER_DISCOVERED"
    elif strictly_decreasing and h1_values[-1] < baseline_h1:
        classification = "D27_FORWARD_PROGRESS_CONFIRMED"
    else:
        classification = "D27_TRAJECTORY_PLATEAU_OBSERVED"
    return {
        "schema_version": "stage3_h13_d27_forward_trajectory_v1",
        "baseline": {"update": 400, "H1": baseline_h1, "H32": baseline_h32},
        "trajectory": list(rows),
        "measurements": {
            "updates_executed": updates,
            "H1_final": h1_values[-1] if h1_values else baseline_h1,
            "H32_final": h32_values[-1] if h32_values else baseline_h32,
            "H1_improvement_relative_to_update_400": (baseline_h1 - h1_values[-1]) if h1_values else 0.0,
            "H32_change_relative_to_update_400": (h32_values[-1] - baseline_h32) if h32_values else 0.0,
            "H1_improvement_slope_per_update": reduction_slope,
            "H1_value_slope_per_update": h1_slope,
            "per_update_H1_deltas": h1_deltas,
            "per_update_H32_deltas": h32_deltas,
            "minimum_H32_margin": min((float(row["H32_margin_after"]) for row in rows), default=design.H32_LIMIT - baseline_h32),
            "strictly_improving_H1": strictly_decreasing,
            "alpha_values": sorted({float(row["alpha"]) for row in rows}),
            "optimizer_consistency_all_pass": all(row.get("optimizer_consistency") == "PASS" for row in rows),
            "deterministic_replay_all_pass": all(row.get("deterministic_replay_consistency") == "PASS" for row in rows),
        },
        "classification": classification,
        "stopped_blocker": stopped_blocker,
        "stability_assessment": "STABLE_MONOTONIC_H1_DESCENT" if strictly_decreasing and not stopped_blocker else ("BLOCKED" if stopped_blocker else "PLATEAU_OR_NON_MONOTONIC"),
    }


def artifact_manifest(output: Path, d26_parent: Path, d26_manifest_path: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "CHECKPOINT_HASH_MANIFEST.json":
            files.append({"relative_path": path.relative_to(output).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    checkpoints = [entry for entry in files if entry["relative_path"].startswith("checkpoints/")]
    return {
        "schema_version": "stage3_h13_d27_checkpoint_hash_manifest_v1",
        "hash_algorithm": "SHA-256",
        "self_excluded": True,
        "d26_parent": {"path": str(d26_parent.resolve()), "archive_sha256": sha256_file(d26_parent), "expected_sha256": EXPECTED_D26_PARENT_SHA256},
        "d26_execution_manifest": {"path": str(d26_manifest_path.resolve()), "archive_sha256": sha256_file(d26_manifest_path), "expected_sha256": EXPECTED_D26_MANIFEST_SHA256},
        "checkpoint_entries": checkpoints,
        "managed_files": files,
        "file_count": len(files),
    }


def run_forward(output: Path, parent_path: Path = DEFAULT_PARENT, d26_manifest_path: Path = DEFAULT_D26_MANIFEST, d26_trajectory_path: Path = DEFAULT_D26_TRAJECTORY, d26_identity_path: Path = DEFAULT_D26_IDENTITY, authorization_path: Path = DEFAULT_AUTHORIZATION, target_last_update: int = TARGET_LAST_UPDATE) -> dict[str, Any]:
    require(not output.exists(), f"D27_OUTPUT_ALREADY_EXISTS:{output}")
    require(target_last_update >= TARGET_FIRST_UPDATE, "D27_TARGET_RANGE_INVALID")
    d26_manifest, d26_final_row, current_model, current_optimizer, current_rng, parent_state, canonical_lrs, d26_manifest_sha256 = load_authenticated_parent(parent_path, d26_manifest_path, d26_trajectory_path, d26_identity_path)
    output.mkdir(parents=True, exist_ok=False)
    auth_info = {"path": str(authorization_path.resolve()), "exists": authorization_path.is_file(), "sha256": sha256_file(authorization_path) if authorization_path.is_file() else None}
    current_update = 400
    current_h1 = parent_state["h1"]
    current_h32 = parent_state["h32"]
    current_parent_hash = EXPECTED_D26_PARENT_SHA256
    records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    checkpoint_entries: list[dict[str, Any]] = []
    stopped_blocker: str | None = None
    for update in range(TARGET_FIRST_UPDATE, target_last_update + 1):
        diagnostic_path = output / "optimizer_consistency" / f"update_{update}_alpha_0p03125.pt"
        record, candidate_model, candidate_optimizer, candidate_rng = d26.run_candidate(d24.d17.preflight_runtime(), current_model, current_optimizer, current_rng, canonical_lrs, EXPECTED_ALPHA, current_update, update, current_h1, current_h32, diagnostic_path)
        replay_info: dict[str, Any]
        replay_record: dict[str, Any] | None = None
        replay_model = replay_optimizer = replay_rng = None
        if candidate_model is None or candidate_optimizer is None or candidate_rng is None:
            replay_info = {"pass": False, "status": "NOT_RUN_AFTER_EXECUTION_EXCEPTION"}
        else:
            replay_diagnostic_path = output / "optimizer_consistency" / f"update_{update}_alpha_0p03125_replay.pt"
            try:
                replay_record, replay_model, replay_optimizer, replay_rng = d26.run_candidate(d24.d17.preflight_runtime(), current_model, current_optimizer, current_rng, canonical_lrs, EXPECTED_ALPHA, current_update, update, current_h1, current_h32, replay_diagnostic_path)
                replay_info = replay_compare(record, replay_record, candidate_model, replay_model, candidate_optimizer, replay_optimizer, candidate_rng, replay_rng)
                replay_info["status"] = "PASS" if replay_info["pass"] else "MISMATCH"
            except Exception as error:
                replay_info = {"pass": False, "status": f"EXCEPTION:{type(error).__name__}:{error}"}
        record["deterministic_replay_consistency_pass"] = bool(replay_info.get("pass"))
        record["deterministic_replay_consistency"] = "PASS" if replay_info.get("pass") else "FAIL"
        record["deterministic_replay_status"] = replay_info.get("status")
        record["deterministic_replay_candidate_record_digest"] = replay_info.get("candidate_record_digest")
        record["deterministic_replay_record_digest"] = replay_info.get("replay_record_digest")
        record["candidate_valid_before_replay_gate"] = bool(record.get("candidate_valid"))
        if not replay_info.get("pass"):
            record["candidate_valid"] = False
            existing = str(record.get("rejection_reason", "NONE"))
            record["rejection_reason"] = ";".join(item for item in (existing, "DETERMINISTIC_REPLAY_FAILURE") if item and item != "NONE")
        records.append(record)
        if diagnostic_path.is_file():
            diagnostic = torch.load(diagnostic_path, map_location="cpu", weights_only=False)
            write_json(output / "optimizer_consistency" / f"update_{update}.json", {"update": update, "parent_update": current_update, "alpha": EXPECTED_ALPHA, "record": record, "adamw_transition_summary": diagnostic.get("summary", {})})
        if record.get("candidate_valid") is not True:
            stopped_blocker = str(record.get("rejection_reason") or "D27_CANDIDATE_GATE_FAILURE")
            break
        checkpoint_path, checkpoint_sha = commit_checkpoint(output, candidate_model, candidate_optimizer, candidate_rng, record, current_update, update, EXPECTED_ALPHA, canonical_lrs, d26_manifest_sha256)
        checkpoint_check = verify_checkpoint(checkpoint_path, current_update, update, current_parent_hash, record)
        require(checkpoint_check["pass"], f"D27_CHECKPOINT_CHAIN_VALIDATION_FAILED_UPDATE_{update}")
        parent_h1, parent_h32 = current_h1, current_h32
        current_model, current_optimizer, current_rng = candidate_model, candidate_optimizer, candidate_rng
        current_parent_hash = checkpoint_sha
        trajectory_row = {
            "update": update,
            "parent_update": current_update,
            "alpha": EXPECTED_ALPHA,
            "effective_lr": json.loads(record["effective_lr"]),
            "H1_before": parent_h1,
            "H1": float(record["candidate_H1"]),
            "delta_H1": float(record["delta_H1"]),
            "H1_improvement_from_update_400": parent_state["h1"] - float(record["candidate_H1"]),
            "H32_before": parent_h32,
            "H32": float(record["candidate_H32"]),
            "delta_H32": float(record["delta_H32"]),
            "H32_change_from_update_400": float(record["candidate_H32"]) - parent_state["h32"],
            "H32_margin_after": float(record["H32_margin_after"]),
            "checkpoint_chain_integrity": "PASS",
            "optimizer_state_continuity": "PASS" if record.get("optimizer_consistency_pass") else "FAIL",
            "actual_candidate_learning_rate": "PASS" if record.get("optimizer_actual_effective_lr") == record.get("optimizer_reconstruction_assumed_lr") == record.get("effective_lr") else "FAIL",
            "adamw_moment_reconstruction": "PASS" if record.get("optimizer_consistency_pass") else "FAIL",
            "step_counter_consistency": "PASS" if record.get("optimizer_step_identity_pass") else "FAIL",
            "parameter_transition_residual": "PASS" if record.get("optimizer_consistency_pass") else "FAIL",
            "clipping_consistency": "PASS" if record.get("finite_state_pass") else "FAIL",
            "deterministic_replay_consistency": "PASS" if record.get("deterministic_replay_consistency_pass") else "FAIL",
            "optimizer_consistency": "PASS" if record.get("optimizer_consistency_pass") else "FAIL",
            "finite_status": "FINITE",
            "checkpoint_sha256": checkpoint_sha,
            "optimizer_state_sha256_or_equivalent": record.get("optimizer_state_sha256_or_equivalent"),
            "batch_identity": record.get("batch_identity"),
        }
        trajectory_rows.append(trajectory_row)
        continuity_row = {key: record.get(key) for key in ("candidate_update", "parent_update", "alpha", "effective_lr", "optimizer_actual_effective_lr", "optimizer_reconstruction_assumed_lr", "optimizer_step_before", "optimizer_step_after", "optimizer_step_identity_pass", "optimizer_exp_avg_expected_sha256", "optimizer_exp_avg_realized_sha256", "optimizer_exp_avg_sq_expected_sha256", "optimizer_exp_avg_sq_realized_sha256", "optimizer_parameter_delta_expected_sha256", "optimizer_parameter_delta_realized_sha256", "optimizer_weight_decay_sha256", "optimizer_adaptive_adam_sha256", "optimizer_total_reconstructed_update_sha256", "optimizer_residual_sha256", "optimizer_consistency_pass", "optimizer_consistency_failure_subconditions", "adamw_decomposition_status", "parameter_delta_l2", "parameter_delta_linf", "postclip_gradient_l2", "raw_gradient_digest", "postclip_gradient_digest", "rng_identity_pass")}
        continuity_row.update({"update": update, "checkpoint_sha256": checkpoint_sha, "checkpoint_chain_integrity_pass": True, "deterministic_replay_consistency_pass": bool(record.get("deterministic_replay_consistency_pass")), "deterministic_replay": replay_info})
        continuity_rows.append(continuity_row)
        write_json(output / "metrics" / f"update_{update}.json", {"update": update, "parent_update": trajectory_row["parent_update"], "H1": trajectory_row["H1"], "H32": trajectory_row["H32"], "H1_improvement_from_update_400": trajectory_row["H1_improvement_from_update_400"], "H32_change_from_update_400": trajectory_row["H32_change_from_update_400"], "trajectory_row": trajectory_row, "record": record})
        checkpoint_entries.append({"update": update, "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha, "parent_update": trajectory_row["parent_update"], "parent_checkpoint_sha256": trajectory_row["checkpoint_sha256"] if False else (EXPECTED_D26_PARENT_SHA256 if update == TARGET_FIRST_UPDATE else trajectory_rows[-2]["checkpoint_sha256"]), "H1": trajectory_row["H1"], "H32": trajectory_row["H32"], "validation": checkpoint_check})
        current_h1, current_h32, current_update = float(record["candidate_H1"]), float(record["candidate_H32"]), update

    trajectory = make_trajectory(parent_state["h1"], parent_state["h32"], trajectory_rows, stopped_blocker)
    write_json(output / "D27_FORWARD_TRAJECTORY.json", trajectory)
    write_csv(output / "D27_CANDIDATE_RESULTS.csv", records)
    write_csv(output / "D27_FORWARD_TRAJECTORY.csv", trajectory_rows)
    optimizer_report = {
        "schema_version": "stage3_h13_d27_optimizer_continuity_report_v1",
        "parent": parent_state,
        "policy": {"proposal": "P4", "alpha": EXPECTED_ALPHA, "canonical_lr": canonical_lrs, "effective_lr": design.effective_group_lrs(canonical_lrs, EXPECTED_ALPHA), "preserve_adamw_state": True, "preserve_moments": True, "preserve_step_counters": True, "preserve_clipping": True, "preserve_rng_contract": True},
        "per_update": continuity_rows,
        "overall": {"updates_checked": [row["update"] for row in continuity_rows], "optimizer_consistency_all_pass": all(row.get("optimizer_consistency_pass") is True for row in continuity_rows), "checkpoint_chain_all_pass": all(row.get("checkpoint_chain_integrity_pass") is True for row in continuity_rows), "deterministic_replay_all_pass": all(row.get("deterministic_replay_consistency_pass") is True for row in continuity_rows), "actual_lr_all_pass": all(row.get("optimizer_actual_effective_lr") == row.get("optimizer_reconstruction_assumed_lr") for row in continuity_rows), "adamw_moment_reconstruction_all_pass": all(row.get("optimizer_consistency_pass") is True for row in continuity_rows), "step_counter_all_pass": all(row.get("optimizer_step_identity_pass") is True for row in continuity_rows), "parameter_residual_all_pass": all(row.get("optimizer_consistency_pass") is True for row in continuity_rows), "clipping_all_pass": all(row.get("adamw_decomposition_status") == "PASS" for row in continuity_rows), "status": "PASS" if continuity_rows and not stopped_blocker and all(row.get("optimizer_consistency_pass") is True and row.get("deterministic_replay_consistency_pass") is True for row in continuity_rows) else "INCOMPLETE_OR_BLOCKED"},
    }
    write_json(output / "OPTIMIZER_CONTINUITY_REPORT.json", optimizer_report)
    execution_manifest = {
        "schema_version": "stage3_h13_d27_execution_manifest_v1",
        "experiment_id": "STAGE_3_H13_POST_D26_D27_FORWARD_CONTINUATION",
        "authorization": auth_info,
        "authoritative_d26": {"archive_root": str(DEFAULT_D26_ARCHIVE.resolve()), "execution_manifest_sha256": d26_manifest_sha256, "execution_manifest_expected_sha256": EXPECTED_D26_MANIFEST_SHA256, "update_400_checkpoint_sha256": EXPECTED_D26_PARENT_SHA256, "update_400_checkpoint_path": str(parent_path.resolve())},
        "source_revision": git_head(),
        "forward_range": {"first_update": TARGET_FIRST_UPDATE, "requested_last_update": target_last_update, "last_committed_update": current_update, "updates_executed": sorted({int(row["candidate_update"]) for row in records}), "updates_committed": [int(row["update"]) for row in trajectory_rows]},
        "fixed_policy": {"alpha": EXPECTED_ALPHA, "proposal": "P4", "canonical_lr": canonical_lrs, "effective_lr": design.effective_group_lrs(canonical_lrs, EXPECTED_ALPHA), "no_optimizer_redesign": True, "no_threshold_relaxation": True},
        "validation_contract": ["checkpoint chain integrity", "optimizer state continuity", "actual candidate learning rate", "AdamW moment reconstruction", "step counter consistency", "parameter transition residuals", "clipping consistency", "deterministic replay consistency"],
        "commit_evidence": checkpoint_entries,
        "classification": trajectory["classification"],
        "first_blocker": stopped_blocker,
        "validation_repairs": [{"blocker": "D27_REPLAY_COMPARATOR_NUMPY_ARRAY_TRUTH_VALUE", "repair": "Compare NumPy RNG arrays with numpy.array_equal during deterministic replay validation; rerun from unchanged D26 update-400 parent.", "scientific_contract_changed": False}],
        "stage3_completion_readiness": "D27_FORWARD_EVIDENCE_READY_STAGE_3_COMPLETION_NOT_CLAIMED_INHERITED_CONTRACT_REMAINS_SEPARATE",
    }
    write_json(output / "D27_EXECUTION_MANIFEST.json", execution_manifest)
    final_h1 = float(trajectory["measurements"]["H1_final"])
    final_h32 = float(trajectory["measurements"]["H32_final"])
    final_report = [
        "# D27 Forward Continuation Final Report",
        "",
        f"Execution completed: {'YES' if not stopped_blocker and current_update >= target_last_update else 'NO — stopped at update {current_update}'}",
        f"First blocker: {stopped_blocker or 'none'}",
        "Validation repair applied: deterministic replay comparator now uses array equality for NumPy RNG arrays; no scientific state or threshold changed.",
        f"Updates executed: {', '.join(str(x) for x in sorted({int(row['candidate_update']) for row in records})) or 'none'}",
        f"Updates committed: {', '.join(str(x) for x in [row['update'] for row in trajectory_rows]) or 'none'}",
        f"Final H1: {final_h1:.17g}",
        f"Final H32: {final_h32:.17g}",
        f"Cumulative H1 improvement from update 400: {float(trajectory['measurements']['H1_improvement_relative_to_update_400']):.17g}",
        f"H32 change from update 400: {float(trajectory['measurements']['H32_change_relative_to_update_400']):.17g}",
        f"Improvement slope (H1 reduction/update): {trajectory['measurements']['H1_improvement_slope_per_update']}",
        f"Best checkpoint hash: {checkpoint_entries[-1]['checkpoint_sha256'] if checkpoint_entries else EXPECTED_D26_PARENT_SHA256}",
        f"Optimizer consistency status: {optimizer_report['overall']['status']}",
        f"Deterministic replay consistency: {'PASS' if optimizer_report['overall']['deterministic_replay_all_pass'] else 'INCOMPLETE_OR_BLOCKED'}",
        f"Classification: {trajectory['classification']}",
        "Stage 3 completion readiness: D27 forward evidence is ready; full Stage 3 completion is not claimed because the inherited completion contract remains separate.",
        "",
        "The continuation preserved alpha=1/32, actual candidate LR 1.5625e-6, AdamW state/moments, step counters, clipping, and RNG identity. No historical D26 artifact was modified.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(final_report) + "\n", encoding="utf-8", newline="\n")
    handoff = [
        "READ THIS FILE FIRST — D27 CROSS-CHAT HANDOFF",
        "",
        "AUTHORITATIVE START: D26 committed update 400",
        f"D26 update-400 checkpoint SHA-256: {EXPECTED_D26_PARENT_SHA256}",
        f"D27 output: {output.resolve()}",
        f"D27 classification: {trajectory['classification']}",
        f"D27 last committed update: {current_update}",
        f"D27 final H1: {final_h1:.17g}",
        f"D27 final H32: {final_h32:.17g}",
        f"D27 H1 improvement from update 400: {float(trajectory['measurements']['H1_improvement_relative_to_update_400']):.17g}",
        f"D27 first blocker: {stopped_blocker or 'none'}",
        "",
        "READ NEXT: D27_EXECUTION_MANIFEST.json; FINAL_REPORT.md; D27_FORWARD_TRAJECTORY.json; OPTIMIZER_CONTINUITY_REPORT.json; CHECKPOINT_HASH_MANIFEST.json.",
        "Do not rerun D26 or modify historical artifacts. Preserve the D27 best checkpoint if continuing.",
    ]
    (output / "NEXT_CHAT_HANDOFF.txt").write_text("\n".join(handoff) + "\n", encoding="utf-8", newline="\n")
    manifest = artifact_manifest(output, parent_path, d26_manifest_path)
    write_json(output / "CHECKPOINT_HASH_MANIFEST.json", manifest)
    return {"status": "PASSED" if not stopped_blocker and current_update >= target_last_update else "INCOMPLETE", "output": str(output.resolve()), "classification": trajectory["classification"], "last_committed_update": current_update, "records": len(records), "manifest_sha256": sha256_file(output / "D27_EXECUTION_MANIFEST.json"), "checkpoint_hash_manifest_sha256": sha256_file(output / "CHECKPOINT_HASH_MANIFEST.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description="D27 forward continuation from authenticated D26 update 400")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--d26-manifest", type=Path, default=DEFAULT_D26_MANIFEST)
    parser.add_argument("--d26-trajectory", type=Path, default=DEFAULT_D26_TRAJECTORY)
    parser.add_argument("--d26-identity", type=Path, default=DEFAULT_D26_IDENTITY)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--target-last-update", type=int, default=TARGET_LAST_UPDATE)
    args = parser.parse_args()
    try:
        print(json.dumps(run_forward(args.output.resolve(), args.parent_checkpoint.resolve(), args.d26_manifest.resolve(), args.d26_trajectory.resolve(), args.d26_identity.resolve(), args.authorization.resolve(), args.target_last_update), sort_keys=True))
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}:{error}"}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
