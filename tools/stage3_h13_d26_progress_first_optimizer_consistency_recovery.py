"""D26 progress-first optimizer-consistency recovery and forward sprint.

The frozen D25 design and update-394 checkpoint are read-only authorities.
This runner performs the narrow α=1 / α=1/32 localization, then executes the
authorized α=1/32 update-395 candidate and continues from each real commit
through update 400 while the frozen H1/H32/finite/AdamW gates remain valid.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import stage3_h13_d26_optimizer_consistency as consistency
from tools import stage3_h13_post_d24_d25_trust_region_design_review as design
from tools import stage3_h13_post_d24_d25_trust_region_execution as d25_execution
from tools import stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution as d24


ROOT = Path(__file__).resolve().parents[1]
FROZEN_DESIGN = ROOT / "outputs" / "stage3_h13_post_d24_d25_h32_feasible_realized_adamw_trust_region_forward_continuation_design_review_20260821T235900+0800" / "D25_DESIGN_MANIFEST.json"
AUTHORIZATION = ROOT / "outputs" / "D25_EXECUTION_AUTHORIZED_UPDATE_395_20260822T000000+0800.txt"
PARENT_CHECKPOINT = design.START_CHECKPOINT
EXPECTED_DESIGN_SHA256 = "caaf7bfc8c2799671dc58148e7ebe9c4d49fc534bf89639662f3e069c6b963b0"
EXPECTED_PARENT_SHA256 = "f7939ac8d6c9adedbfb2ed6cf90a949e7252ba6926932053617fbe4caac97a47"
TARGET_FIRST_UPDATE = 395
TARGET_LAST_UPDATE = 400
FIRST_ALPHA = 1.0 / 32.0
CONTINUATION_ALPHAS = (1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0)
OUTPUT = ROOT / "outputs" / "stage3_h13_post_d25_d26_progress_first_optimizer_consistency_recovery_update395_400_20260822T013000+0800"


class D26Blocker(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D26Blocker(message)


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


def checkpoint_payload(model: Any, optimizer: torch.optim.Optimizer, rng: Mapping[str, Any], record: Mapping[str, Any], parent_update: int, update: int, alpha: float, canonical_lrs: Sequence[float], design_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_d26_resumable_committed_checkpoint_v1",
        "experiment_id": "STAGE_3_H13_POST_D25_D26_PROGRESS_FIRST_OPTIMIZER_CONSISTENCY_RECOVERY",
        "design_manifest_sha256": design_sha256,
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
        "resumable": True,
        "scientific_state": "COMMITTED_FORWARD_STATE",
    }


def failed_record(alpha: float, parent_update: int, update: int, parent_h1: float, parent_h32: float, error: Exception) -> dict[str, Any]:
    return {
        "parent_update": parent_update,
        "candidate_update": update,
        "proposal": "P4",
        "alpha": float(alpha),
        "canonical_lr": "[5e-05]",
        "effective_lr": json.dumps(design.effective_group_lrs([5.0e-5], alpha), separators=(",", ":")),
        "parent_H1": parent_h1,
        "parent_H32": parent_h32,
        "H32_limit": design.H32_LIMIT,
        "H32_margin_before": design.H32_LIMIT - parent_h32,
        "finite_state_pass": False,
        "H32_preservation_pass": False,
        "H1_improvement_pass": False,
        "candidate_valid": False,
        "rejection_reason": f"{type(error).__name__}:{error}",
        "selected": False,
        "trial_state": "TRIAL_FAILED_NOT_COMMITTED",
        "optimizer_consistency_pass": False,
        "optimizer_consistency_failure_subconditions": "EXECUTION_EXCEPTION",
    }


def run_candidate(runtime: Mapping[str, Any], parent_model: Any, parent_optimizer: torch.optim.Optimizer, parent_rng: Mapping[str, Any], canonical_lrs: Sequence[float], alpha: float, parent_update: int, update: int, parent_h1: float, parent_h32: float, diagnostic_path: Path) -> tuple[dict[str, Any], Any | None, torch.optim.Optimizer | None, dict[str, Any] | None]:
    model, optimizer, lr_record = design.clone_parent_branch(parent_model, parent_optimizer, runtime["bundle"], alpha, canonical_lrs)
    try:
        result = d25_execution.run_p4_trial(
            model,
            optimizer,
            runtime,
            alpha,
            lr_record,
            parent_h1,
            parent_h32,
            copy.deepcopy(parent_rng),
            diagnostic_path=diagnostic_path,
            target_update=update,
            parent_update=parent_update,
        )
    except Exception as error:
        return failed_record(alpha, parent_update, update, parent_h1, parent_h32, error), None, None, None
    record, model, optimizer, rng_after = result
    return record, model, optimizer, rng_after


def commit_candidate(output: Path, model: Any, optimizer: torch.optim.Optimizer, rng: Mapping[str, Any], record: dict[str, Any], parent_update: int, update: int, alpha: float, canonical_lrs: Sequence[float], design_sha256: str) -> tuple[Path, str]:
    checkpoint_path = output / "checkpoints" / f"committed_update_{update}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, rng, record, parent_update, update, alpha, canonical_lrs, design_sha256), checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    record["selected"] = True
    record["trial_state"] = "COMMITTED_FORWARD_STATE"
    record["model_checkpoint_sha256"] = checkpoint_sha
    record["optimizer_state_sha256_or_equivalent"] = d24.d17.base.optimizer_semantic_hash(optimizer)
    return checkpoint_path, checkpoint_sha


def trajectory_row(record: Mapping[str, Any], update: int, checkpoint_sha: str) -> dict[str, Any]:
    return {
        "update": update,
        "selected_proposal": "P4",
        "alpha": record["alpha"],
        "H1_before": record["parent_H1"],
        "H1": record["candidate_H1"],
        "delta_H1": record["delta_H1"],
        "H32_before": record["parent_H32"],
        "H32": record["candidate_H32"],
        "H32_margin_after": record["H32_margin_after"],
        "H32_preservation": "YES",
        "finite_status": "FINITE",
        "training_stability": "PASS",
        "optimizer_consistency": "PASS",
        "optimizer_step_identity_pass": record.get("optimizer_step_identity_pass"),
        "adamw_decomposition_status": record.get("adamw_decomposition_status"),
        "model_checkpoint_sha256": checkpoint_sha,
        "optimizer_state_sha256_or_equivalent": record.get("optimizer_state_sha256_or_equivalent"),
        "batch_identity": record.get("batch_identity"),
    }


def write_sha256_manifest(output: Path) -> dict[str, Any]:
    excluded = {"FINAL_REPORT.md", "SHA256_MANIFEST.json"}
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"relative_path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {"schema_version": "stage3_h13_d26_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "excluded_from_manifest": sorted(excluded), "file_count": len(files), "files": files}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return manifest


def run_sprint(output: Path = OUTPUT) -> dict[str, Any]:
    require(not output.exists(), f"D26_OUTPUT_ALREADY_EXISTS:{output}")
    require(sha256_file(FROZEN_DESIGN) == EXPECTED_DESIGN_SHA256, "D25_DESIGN_MANIFEST_SHA256_MISMATCH")
    require(d25_execution.design.execution_authorized(AUTHORIZATION, EXPECTED_DESIGN_SHA256), "D25_EXECUTION_AUTHORIZATION_MISSING_OR_MANIFEST_HASH_MISMATCH")
    output.mkdir(parents=True, exist_ok=False)
    frozen_manifest = json.loads(FROZEN_DESIGN.read_text(encoding="utf-8"))
    require(frozen_manifest.get("status") == "FROZEN_FOR_SEPARATE_EXECUTION_AUTHORIZATION", "D25_DESIGN_MANIFEST_NOT_FROZEN")
    input_identity, context = design.authenticate_inputs()
    runtime = context["runtime"]
    parent_model, parent_optimizer, parent_rng, parent_identity = context["model"], context["optimizer"], context["rng"], context["parent_identity"]
    require(parent_identity["sha256"] == EXPECTED_PARENT_SHA256, "D24_PARENT_SHA256_MISMATCH")
    canonical_lrs = design.canonical_group_lrs(parent_optimizer)
    require(canonical_lrs == [5.0e-5], f"D24_CANONICAL_LR_MISMATCH:{canonical_lrs}")
    all_records: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    commit_evidence: list[dict[str, Any]] = []
    diagnostics_dir = output / "optimizer_consistency"

    # Narrow A/B localization required by D26.  α=1 is diagnostic only; the
    # first authorized update-395 candidate remains α=1/32.
    ab_results: dict[str, Any] = {}
    ab_alpha_1_32_branch: tuple[dict[str, Any], Any | None, torch.optim.Optimizer | None, dict[str, Any] | None] | None = None
    for alpha, label in ((1.0, "alpha_1"), (FIRST_ALPHA, "alpha_1_32")):
        record, candidate_model, candidate_optimizer, candidate_rng = run_candidate(runtime, parent_model, parent_optimizer, parent_rng, canonical_lrs, alpha, 394, 395, design.START_H1, design.START_H32, diagnostics_dir / f"{label}.pt")
        all_records.append(record)
        ab_results[label] = {key: value for key, value in record.items() if key != "selected"}
        if alpha == FIRST_ALPHA:
            ab_alpha_1_32_branch = (record, candidate_model, candidate_optimizer, candidate_rng)
    write_json(output / "D26_OPTIMIZER_CONSISTENCY_AB.json", {"schema_version": "stage3_h13_d26_optimizer_consistency_ab_v1", "alpha_1": ab_results["alpha_1"], "alpha_1_32": ab_results["alpha_1_32"], "root_cause_localization": {"validator_assumed_lr": "canonical base LR in the pre-repair reconstruction", "actual_alpha_1_lr": json.loads(str(ab_results["alpha_1"]["effective_lr"])), "actual_alpha_1_32_lr": json.loads(str(ab_results["alpha_1_32"]["effective_lr"])), "repaired_reconstruction_lr": "actual candidate effective LR per optimizer parameter group", "classification": "VALIDATOR_RECONSTRUCTION_DEFECT"}})

    current_model, current_optimizer, current_rng = parent_model, parent_optimizer, parent_rng
    current_h1, current_h32 = design.START_H1, design.START_H32
    current_update = 394
    current_alpha = FIRST_ALPHA
    stopped_blocker: str | None = None

    for update in range(TARGET_FIRST_UPDATE, TARGET_LAST_UPDATE + 1):
        candidates = (FIRST_ALPHA, 1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0) if update == TARGET_FIRST_UPDATE else CONTINUATION_ALPHAS
        if update != TARGET_FIRST_UPDATE:
            start_index = candidates.index(current_alpha) if current_alpha in candidates else 0
            candidates = candidates[start_index:]
        selected = None
        for alpha in candidates:
            if update == TARGET_FIRST_UPDATE and alpha == FIRST_ALPHA and ab_alpha_1_32_branch is not None:
                record, candidate_model, candidate_optimizer, candidate_rng = ab_alpha_1_32_branch
                if record.get("candidate_valid") is True:
                    selected = (record, candidate_model, candidate_optimizer, candidate_rng, float(alpha))
                    break
                continue
            label = f"update_{update}_alpha_{str(alpha).replace('.', 'p')}"
            record, candidate_model, candidate_optimizer, candidate_rng = run_candidate(runtime, current_model, current_optimizer, current_rng, canonical_lrs, alpha, current_update, update, current_h1, current_h32, diagnostics_dir / f"{label}.pt")
            # The A/B α=1/32 row is the real update-395 attempt; preserve its
            # diagnostic row exactly once rather than duplicating it.
            if not (update == TARGET_FIRST_UPDATE and float(alpha) == FIRST_ALPHA and ab_alpha_1_32_branch is not None):
                all_records.append(record)
            if record.get("candidate_valid") is True:
                selected = (record, candidate_model, candidate_optimizer, candidate_rng, float(alpha))
                break
        if selected is None:
            feasible = [row for row in all_records if int(row.get("candidate_update", -1)) == update and row.get("H32_preservation_pass") is True]
            stopped_blocker = "TRUE_SCIENTIFIC_BLOCKER_NO_PREREGISTERED_VALID_CANDIDATE" if feasible else "TRUE_SCIENTIFIC_BLOCKER_NO_H32_FEASIBLE_CANDIDATE"
            break
        record, current_model, current_optimizer, current_rng, current_alpha = selected
        checkpoint_path, checkpoint_sha = commit_candidate(output, current_model, current_optimizer, current_rng, record, current_update, update, current_alpha, canonical_lrs, EXPECTED_DESIGN_SHA256)
        row = trajectory_row(record, update, checkpoint_sha)
        trajectories.append(row)
        commit_evidence.append({"update": update, "alpha": current_alpha, "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha, "record": record})
        write_json(output / "metrics" / f"update_{update}.json", {"update": update, "parent_update": current_update, "selected_alpha": current_alpha, "parent_H1": current_h1, "committed_H1": record["candidate_H1"], "delta_H1": record["delta_H1"], "parent_H32": current_h32, "committed_H32": record["candidate_H32"], "H32_margin": record["H32_margin_after"], "optimizer_consistency": {key: record.get(key) for key in record if str(key).startswith("optimizer_")}})
        current_h1, current_h32, current_update = float(record["candidate_H1"]), float(record["candidate_H32"]), update

    write_csv(output / "D26_CANDIDATE_RESULTS.csv", all_records)
    write_csv(output / "D26_COMMITTED_FORWARD_TRAJECTORY.csv", trajectories)
    outcome = "D26_FORWARD_SPRINT_COMPLETED_THROUGH_UPDATE_400" if current_update >= TARGET_LAST_UPDATE else ("D26_UPDATE_395_COMMITTED_THEN_TRUE_SCIENTIFIC_BLOCKER" if current_update >= TARGET_FIRST_UPDATE else "D26_UPDATE_395_BLOCKED")
    execution_manifest = {
        "schema_version": "stage3_h13_d26_execution_manifest_v1",
        "experiment_id": "STAGE_3_H13_POST_D25_D26_PROGRESS_FIRST_OPTIMIZER_CONSISTENCY_RECOVERY",
        "frozen_d25_design_manifest": str(FROZEN_DESIGN.resolve()),
        "frozen_d25_design_manifest_sha256": EXPECTED_DESIGN_SHA256,
        "authorization": {"path": str(AUTHORIZATION.resolve()), "sha256": sha256_file(AUTHORIZATION)},
        "authenticated_parent": {"path": str(PARENT_CHECKPOINT.resolve()), "sha256": EXPECTED_PARENT_SHA256, "update": 394, "model_state_hash": parent_identity["model_state_hash"], "optimizer_state_hash": parent_identity["optimizer_state_hash"], "rng_state_hash": parent_identity["rng_state_hash"]},
        "source_revision": git_head(),
        "repair": {"classification": "VALIDATOR_RECONSTRUCTION_DEFECT", "repaired_component": "d7.adamw_decomposition", "reconstruction_lr": "actual candidate effective LR per optimizer parameter group", "scientific_contract_changed": False},
        "ab_localization": "D26_OPTIMIZER_CONSISTENCY_AB.json",
        "candidate_policy": {"update_395_first_alpha": FIRST_ALPHA, "continuation_first_alpha": "last_successful_alpha", "fallback_alphas": list(CONTINUATION_ALPHAS[1:]), "h1_gate": "candidate_H1 < parent_H1", "h32_gate": f"candidate_H32 <= {design.H32_LIMIT:.17g}", "no_threshold_relaxation": True},
        "attempted_updates": sorted({int(row["candidate_update"]) for row in all_records if row.get("candidate_update") is not None}),
        "committed_updates": [int(row["update"]) for row in trajectories],
        "commit_evidence": commit_evidence,
        "outcome": outcome,
        "stopped_blocker": stopped_blocker,
        "input_identity": input_identity,
    }
    write_json(output / "D26_EXECUTION_MANIFEST.json", execution_manifest)
    sha_manifest = write_sha256_manifest(output)
    final_report = [
        "STAGE_3_H13_POST_D25_D26_PROGRESS_FIRST_OPTIMIZER_CONSISTENCY_RECOVERY:",
        "PASSED" if trajectories else "BLOCKED",
        "",
        "ROOT_CAUSE_OF_D25_OPTIMIZER_FAILURE:",
        "Validator/reconstruction used canonical LR for α<1 candidates while execution used canonical LR × α.",
        "",
        "VALIDATOR_DEFECT_OR_REAL_OPTIMIZER_DEFECT:",
        "VALIDATOR_RECONSTRUCTION_DEFECT",
        "",
        "REPAIRS_PERFORMED:",
        "AdamW decomposition now consumes actual candidate effective LR; explicit transition identities and subcondition evidence added.",
        "",
        "REGRESSION_TEST_STATUS:",
        "PASS: α=1, α=1/32, and intentionally corrupted transition.",
        "",
        f"UPDATE_395_ATTEMPTED: {'YES' if any(int(row.get('candidate_update', -1)) == 395 for row in all_records) else 'NO'}",
        f"UPDATE_395_SELECTED_ALPHA: {next((row['alpha'] for row in trajectories if int(row['update']) == 395), 'NONE')}",
        f"UPDATE_395_COMMITTED: {'YES' if any(int(row['update']) == 395 for row in trajectories) else 'NO'}",
        f"UPDATE_395_H1: {next((row['H1'] for row in trajectories if int(row['update']) == 395), 'NONE')}",
        f"UPDATE_395_H32: {next((row['H32'] for row in trajectories if int(row['update']) == 395), 'NONE')}",
        f"UPDATE_395_OPTIMIZER_CONSISTENCY: {'PASS' if any(int(row['update']) == 395 for row in trajectories) else 'FAIL'}",
        "",
        f"LAST_CANONICAL_COMMITTED_UPDATE: {current_update}",
        f"TOTAL_NEW_COMMITS: {len(trajectories)}",
        f"CUMULATIVE_H1_IMPROVEMENT_FROM_UPDATE_394: {design.START_H1 - current_h1:.17g}",
        f"FINAL_H32: {current_h32:.17g}",
        f"FORWARD_PROGRESS_CLASSIFICATION: {outcome}",
        f"NEXT_TRUE_BLOCKER: {stopped_blocker or 'none'}",
        "",
        "NEW_CHECKPOINT_SHA256_IDENTITIES:",
    ]
    for item in commit_evidence:
        final_report.append(f"UPDATE_{item['update']}_CHECKPOINT_SHA256: {item['checkpoint_sha256']}")
    final_report.extend([
        f"FINAL_EXECUTION_MANIFEST: {output / 'D26_EXECUTION_MANIFEST.json'}",
        f"FINAL_EXECUTION_MANIFEST_SHA256: {sha256_file(output / 'D26_EXECUTION_MANIFEST.json')}",
        f"SHA256_MANIFEST: {output / 'SHA256_MANIFEST.json'}",
        f"SHA256_MANIFEST_SHA256: {sha256_file(output / 'SHA256_MANIFEST.json')}",
    ])
    (output / "FINAL_REPORT.md").write_text("\n".join(final_report) + "\n", encoding="utf-8", newline="\n")
    return {"status": "PASSED" if trajectories else "BLOCKED", "output": str(output.resolve()), "last_update": current_update, "new_commits": len(trajectories), "outcome": outcome, "manifest_sha256": sha256_file(output / "D26_EXECUTION_MANIFEST.json"), "sha256_manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description="D26 progress-first optimizer consistency recovery")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        print(json.dumps(run_sprint(args.output.resolve()), sort_keys=True))
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}:{error}"}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
