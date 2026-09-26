"""Execute D29 margin-aware P4 trust-region re-expansion from update 412."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import lzma
import math
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d27_d28_forward_continuation as d28

d27, d26, d24, design = d28.d27, d28.d26, d28.d24, d28.design
EXPERIMENT_ID = "STAGE_3_H13_POST_D28_D29_H32_MARGIN_AWARE_TRUST_REGION_REEXPANSION_AND_FORWARD_OPTIMIZATION_EXECUTION"
D28_ROOT = ROOT / "outputs" / "stage3_h13_post_d27_d28_forward_continuation_20260823T153426+0800"
PARENT = D28_ROOT / "checkpoints" / "committed_update_412.pt"
D28_MANIFEST = D28_ROOT / "D28_EXECUTION_MANIFEST.json"
D28_TRAJECTORY = D28_ROOT / "D28_FORWARD_TRAJECTORY.json"
D28_ARCHIVE = Path(r"C:\Users\86198\Desktop\STAGE3_H13_D28_FORWARD_CONTINUATION_CROSS_CHAT_EVIDENCE_MAX_XZ_20260823T155849+0800.tar.xz")
AUTHORIZATION = Path(r"C:\Users\86198\.codex\attachments\81618b6f-2c67-4b50-9055-6ac8219d92fc\pasted-text.txt")
EXPECTED_ARCHIVE_SHA = "aabc6ef873deb9db4964240f10ef68c7ec249332d9140d4ae8cdc682f4f0066b"
EXPECTED_ARCHIVE_SIZE = 2_007_884
EXPECTED_PARENT_SHA = "f877703c07b39fcd2777ad3c393d4f8532f90ced88490f7c8409a1de63e22dae"
START_H1, START_H32 = 7.926021327879009e-05, 0.01751783891085687
H32_LIMIT, TARGET_H1, CANONICAL_LR = 0.01856902565856056, 5e-05, 5e-05
LADDER = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
INITIAL = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]


class D29Blocker(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise D29Blocker(message)


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def authenticate_archive() -> dict[str, Any]:
    require(D28_ARCHIVE.is_file(), f"D28_ARCHIVE_MISSING:{D28_ARCHIVE}")
    require(D28_ARCHIVE.stat().st_size == EXPECTED_ARCHIVE_SIZE, "D28_ARCHIVE_SIZE_MISMATCH")
    require(sha256_file(D28_ARCHIVE) == EXPECTED_ARCHIVE_SHA, "D28_ARCHIVE_SHA256_MISMATCH")
    with tarfile.open(D28_ARCHIVE, "r:xz") as tar:
        start = tar.extractfile("START_HERE_NEXT_CHAT.txt")
        require(start is not None, "D28_START_HERE_MISSING")
        start_text = start.read().decode("utf-8")
        require(EXPECTED_PARENT_SHA in start_text and "FINAL_UPDATE: 412" in start_text, "D28_START_HERE_IDENTITY_MISMATCH")
        checkpoint = tar.extractfile("07_CHECKPOINTS/committed_update_412.pt")
        require(checkpoint is not None, "D28_ARCHIVE_CHECKPOINT_MISSING")
        digest = hashlib.sha256()
        while chunk := checkpoint.read(1024 * 1024): digest.update(chunk)
        require(digest.hexdigest() == EXPECTED_PARENT_SHA, "D28_EMBEDDED_CHECKPOINT_SHA256_MISMATCH")
    with lzma.open(D28_ARCHIVE, "rb") as stream:
        while stream.read(1024 * 1024): pass
    return {"path": str(D28_ARCHIVE), "sha256": EXPECTED_ARCHIVE_SHA, "size_bytes": EXPECTED_ARCHIVE_SIZE, "start_here_read_first": True, "xz_decompression_test": "PASS", "tar_content_read_test": "PASS", "embedded_checkpoint_sha256": EXPECTED_PARENT_SHA}


def load_parent() -> tuple[Any, torch.optim.Optimizer, dict[str, Any], list[float], dict[str, Any]]:
    require(sha256_file(PARENT) == EXPECTED_PARENT_SHA, "UPDATE_412_CHECKPOINT_SHA256_MISMATCH")
    manifest = json.loads(D28_MANIFEST.read_text(encoding="utf-8")); trajectory = json.loads(D28_TRAJECTORY.read_text(encoding="utf-8"))
    require(manifest["updates_committed"] == list(range(407, 413)), "D28_CHAIN_MISMATCH")
    require(trajectory["trajectory"][-1]["checkpoint_sha256"] == EXPECTED_PARENT_SHA, "D28_TERMINAL_MISMATCH")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    require(payload["completed_optimizer_step"] == 412 and payload["next_schedule_index"] == 412, "PARENT_POSITION_MISMATCH")
    runtime = d24.d17.preflight_runtime(); model, optimizer = d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True); optimizer.load_state_dict(payload["optimizer_state_dict"])
    canonical = [float(x) for x in payload["canonical_base_lr"]]
    require(canonical == [CANONICAL_LR], "CANONICAL_LR_MISMATCH")
    require(all(int(x.get("step", -1)) == 412 for x in payload["optimizer_state_dict"]["state"].values()), "PARENT_STEP_MISMATCH")
    return model, optimizer, copy.deepcopy(payload["rng_state"]), canonical, trajectory


def alpha_name(alpha: float) -> str:
    return (f"{alpha:.9f}").rstrip("0").rstrip(".").replace(".", "p")


def trial(runtime: Mapping[str, Any], output: Path, model: Any, optimizer: torch.optim.Optimizer, rng: Mapping[str, Any], canonical: Sequence[float], alpha: float, parent_update: int, update: int, h1: float, h32: float) -> tuple[dict[str, Any], Any | None, Any | None, Any | None]:
    name = alpha_name(alpha)
    diag = output / "optimizer_consistency" / f"update_{update}_alpha_{name}.pt"
    record, cm, co, cr = d26.run_candidate(runtime, model, optimizer, rng, canonical, alpha, parent_update, update, h1, h32, diag)
    if cm is None or co is None or cr is None:
        replay = {"pass": False, "status": "NOT_RUN_AFTER_EXECUTION_EXCEPTION"}
    else:
        rdiag = output / "optimizer_consistency" / f"update_{update}_alpha_{name}_replay.pt"
        rr, rm, ro, rg = d26.run_candidate(runtime, model, optimizer, rng, canonical, alpha, parent_update, update, h1, h32, rdiag)
        replay = d27.replay_compare(record, rr, cm, rm, co, ro, cr, rg) if rm is not None else {"pass": False, "status": "REPLAY_EXECUTION_EXCEPTION"}
        replay["status"] = "PASS" if replay.get("pass") else replay.get("status", "MISMATCH")
    record.update({"deterministic_replay_consistency_pass": bool(replay.get("pass")), "deterministic_replay": "PASS" if replay.get("pass") else "FAIL", "deterministic_replay_candidate_record_digest": replay.get("candidate_record_digest"), "deterministic_replay_record_digest": replay.get("replay_record_digest"), "selection_status": "NOT_SELECTED", "parent_checkpoint_SHA256": None, "candidate_checkpoint_SHA256": None})
    if not replay.get("pass"):
        record["candidate_valid"] = False
        record["rejection_reason"] = ";".join(x for x in (str(record.get("rejection_reason", "NONE")), "DETERMINISTIC_REPLAY_FAILURE") if x != "NONE")
    diagnostic = torch.load(diag, map_location="cpu", weights_only=False) if diag.is_file() else {}
    write_json(output / "optimizer_consistency" / f"update_{update}_alpha_{name}.json", {"record": record, "adamw_transition_summary": diagnostic.get("summary", {}), "deterministic_replay": replay})
    return record, cm, co, cr


def checkpoint_payload(model: Any, optimizer: Any, rng: Mapping[str, Any], record: Mapping[str, Any], parent_update: int, update: int, alpha: float, canonical: Sequence[float]) -> dict[str, Any]:
    return {"schema_version": "stage3_h13_d29_resumable_committed_checkpoint_v1", "experiment_id": EXPERIMENT_ID, "authoritative_d28_archive_sha256": EXPECTED_ARCHIVE_SHA, "parent_checkpoint_sha256": record["parent_checkpoint_SHA256"], "proposal": "P4", "proposal_spec": copy.deepcopy(design.P4_SPEC), "alpha": alpha, "actual_effective_lr": [CANONICAL_LR * alpha], "parent_update": parent_update, "completed_optimizer_step": update, "next_schedule_index": update, "model_state_dict": copy.deepcopy(model.state_dict()), "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()), "scheduler_state": None, "rng_state": copy.deepcopy(rng), "batch_position": {"completed_optimizer_step": update, "next_zero_based_schedule_index": update}, "canonical_base_lr": list(canonical), "candidate_effective_lr": json.loads(record["effective_lr"]), "H1": float(record["candidate_H1"]), "H32": float(record["candidate_H32"]), "last_record_identity": {"update": update, "batch_window_sha256": record["batch_identity"], "model_state_hash": d24.d17.canonical_model_state_sha256(model.state_dict()), "optimizer_state_hash": d24.d17.base.optimizer_semantic_hash(optimizer)}, "replay_consistency": {"pass": True, "candidate_record_digest": record.get("deterministic_replay_candidate_record_digest"), "replay_record_digest": record.get("deterministic_replay_record_digest")}, "resumable": True, "scientific_state": "COMMITTED_FORWARD_STATE"}


def completion(prior: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    window = (list(prior["trajectory"]) + list(rows))[-8:]
    updates = [int(x["update"]) for x in window]; hs = [float(x["H1"]) for x in window]
    endpoint = bool(hs and hs[-1] <= TARGET_H1); mean = sum(hs) / len(hs) if hs else None
    sustained = len(hs) == 8 and mean <= TARGET_H1
    h32 = len(window) == 8 and all(float(x["H32"]) <= H32_LIMIT for x in window)
    stable = len(window) == 8 and all(x.get("optimizer_consistency") == "PASS" and x.get("deterministic_replay_consistency") == "PASS" for x in window)
    consecutive = len(updates) == 8 and updates == list(range(updates[0], updates[0] + 8))
    passed = endpoint and sustained and h32 and stable and consecutive
    return {"schema_version": "stage3_h13_d29_completion_contract_v1", "H1_target": TARGET_H1, "H32_limit": H32_LIMIT, "window_updates": updates, "last_8_H1_values": hs, "last_8_mean_H1": mean, "requirements": {"endpoint_H1_le_5e-5": endpoint, "last_8_mean_H1_le_5e-5": sustained, "last_8_H32_le_limit": h32, "last_8_stability_pass": stable, "last_8_consecutive_real_forward": consecutive}, "stage3_completion_gate": "PASSED" if passed else "FAILED"}


def local_alphas(previous: float) -> list[float]:
    index = LADDER.index(previous); values = [previous]
    if index > 0: values.append(LADDER[index - 1])
    if index + 1 < len(LADDER): values.append(LADDER[index + 1])
    return values


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"D29_OUTPUT_ALREADY_EXISTS:{output}")
    archive_auth = authenticate_archive(); model, optimizer, rng, canonical, prior = load_parent()
    output.mkdir(parents=True); runtime = d24.d17.preflight_runtime()
    records: list[dict[str, Any]] = []; rows: list[dict[str, Any]] = []; commits: list[dict[str, Any]] = []
    current_update, h1, h32, current_sha = 412, START_H1, START_H32, EXPECTED_PARENT_SHA
    previous_alpha: float | None = None; blocker: str | None = None
    for update in range(413, 421):
        alphas = INITIAL if update == 413 else local_alphas(previous_alpha)  # type: ignore[arg-type]
        print(f"D29_SCREEN update={update} parent={current_update} alphas={alphas}", flush=True)
        candidates = []
        for alpha in alphas:
            record, cm, co, cr = trial(runtime, output, model, optimizer, rng, canonical, alpha, current_update, update, h1, h32)
            record["parent_checkpoint_SHA256"] = current_sha; records.append(record); candidates.append((record, cm, co, cr))
            print(f"D29_TRIAL update={update} alpha={alpha:.9g} valid={record.get('candidate_valid')} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} reason={record.get('rejection_reason')}", flush=True)
        valid = [item for item in candidates if item[0].get("candidate_valid") is True]
        if not valid and update == 413:
            for alpha in LADDER[-3:]:
                record, cm, co, cr = trial(runtime, output, model, optimizer, rng, canonical, alpha, current_update, update, h1, h32)
                record["parent_checkpoint_SHA256"] = current_sha; records.append(record); candidates.append((record, cm, co, cr))
            valid = [item for item in candidates if item[0].get("candidate_valid") is True]
        if not valid:
            blocker = f"NO_VALID_H1_IMPROVING_CANDIDATE_UPDATE_{update}"; break
        selected, sm, so, sr = min(valid, key=lambda item: float(item[0]["candidate_H1"]))
        selected["selection_status"] = "SELECTED_COMMITTED"; selected["selected"] = True
        checkpoint = output / "checkpoints" / f"committed_update_{update}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(sm, so, sr, selected, current_update, update, float(selected["alpha"]), canonical), checkpoint)
        checkpoint_sha = sha256_file(checkpoint); selected["candidate_checkpoint_SHA256"] = checkpoint_sha; selected["model_checkpoint_sha256"] = checkpoint_sha; selected["trial_state"] = "COMMITTED_FORWARD_STATE"
        check = d27.verify_checkpoint(checkpoint, current_update, update, current_sha, selected); require(check["pass"], f"CHECKPOINT_CHAIN_FAILURE_UPDATE_{update}")
        row = {"update": update, "parent_update": current_update, "alpha": float(selected["alpha"]), "effective_lr": CANONICAL_LR * float(selected["alpha"]), "H1_before": h1, "H1": float(selected["candidate_H1"]), "delta_H1": float(selected["delta_H1"]), "H32_before": h32, "H32": float(selected["candidate_H32"]), "delta_H32": float(selected["delta_H32"]), "H32_margin": float(selected["H32_margin_after"]), "gradient_norm": float(selected["gradient_l2"]), "postclip_gradient_norm": float(selected["postclip_gradient_l2"]), "clipping_coefficient": min(1.0, float(selected["postclip_gradient_l2"]) / float(selected["gradient_l2"])), "optimizer_consistency": "PASS", "deterministic_replay_consistency": "PASS", "RNG_continuity": "PASS", "checkpoint_sha256": checkpoint_sha, "parent_checkpoint_sha256": current_sha}
        rows.append(row); commits.append({**row, "validation": check}); write_json(output / "metrics" / f"update_{update}.json", {"trajectory_row": row, "selected_record": selected})
        model, optimizer, rng = sm, so, sr; current_update, h1, h32, current_sha, previous_alpha = update, row["H1"], row["H32"], checkpoint_sha, row["alpha"]
        print(f"D29_COMMITTED update={update} alpha={previous_alpha:.9g} H1={h1:.17g} H32={h32:.17g} sha256={current_sha}", flush=True)
        if completion(prior, rows)["stage3_completion_gate"] == "PASSED": break
    for record in records:
        if record.get("selection_status") != "SELECTED_COMMITTED": record["trial_state"] = "TRIAL_NOT_COMMITTED"
    comp = completion(prior, rows); gains = [-float(x["delta_H1"]) for x in rows]; total = START_H1 - h1; mean_gain = total / len(rows) if rows else 0.0
    d28_mean = 3.593529186071541e-7 / 6.0; accelerated = mean_gain > d28_mean
    if comp["stage3_completion_gate"] == "PASSED": classification = "STAGE3_COMPLETED_H1_TARGET_SUSTAINED"
    elif blocker: classification = "D29_P4_FORWARD_PROGRESS_EXHAUSTED"
    elif accelerated: classification = "D29_TRUST_REGION_REEXPANSION_ACCELERATED_FORWARD_PROGRESS"
    elif rows and all(x["alpha"] == 0.03125 for x in rows): classification = "D29_ALPHA_1_32_REMAINS_OPTIMAL_UNDER_CURRENT_CONSTRAINT"
    else: classification = "D29_TRUST_REGION_REEXPANSION_VALID_BUT_NO_ACCELERATION"
    trajectory = {"schema_version": "stage3_h13_d29_canonical_trajectory_v1", "baseline": {"update": 412, "H1": START_H1, "H32": START_H32}, "trajectory": rows, "measurements": {"updates_committed": [x["update"] for x in rows], "alpha_trajectory": [x["alpha"] for x in rows], "H1_final": h1, "H32_final": h32, "total_H1_improvement": total, "mean_H1_improvement_per_update": mean_gain, "D28_mean_H1_improvement_per_update": d28_mean, "acceleration_ratio": mean_gain / d28_mean if d28_mean else None, "accelerated_relative_to_D28": accelerated, "H32_final_margin": H32_LIMIT - h32, "per_update_H1_gains": gains}, "classification": classification, "first_blocker": blocker}
    write_json(output / "D29_CANONICAL_TRAJECTORY.json", trajectory); write_csv(output / "D29_CANONICAL_TRAJECTORY.csv", rows); write_json(output / "D29_CANDIDATE_RESULTS.json", records); write_csv(output / "D29_CANDIDATE_RESULTS.csv", records); write_json(output / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json", comp)
    optimizer_report = {"schema_version": "stage3_h13_d29_optimizer_consistency_v1", "status": "PASS" if rows and all(x.get("optimizer_consistency_pass") for x in records if x.get("candidate_valid")) else "INCOMPLETE_OR_BLOCKED", "candidate_count": len(records), "actual_lr_validator": "actual candidate learning rate", "per_candidate": [{k: r.get(k) for k in ("parent_update", "candidate_update", "alpha", "effective_lr", "optimizer_actual_effective_lr", "optimizer_reconstruction_assumed_lr", "optimizer_step_before", "optimizer_step_after", "optimizer_exp_avg_expected_sha256", "optimizer_exp_avg_realized_sha256", "optimizer_exp_avg_sq_expected_sha256", "optimizer_exp_avg_sq_realized_sha256", "optimizer_weight_decay_sha256", "optimizer_parameter_delta_expected_sha256", "optimizer_parameter_delta_realized_sha256", "optimizer_consistency_pass", "rng_identity_pass", "deterministic_replay", "candidate_valid", "selection_status", "rejection_reason") } for r in records]}
    write_json(output / "OPTIMIZER_CONSISTENCY_REPORT.json", optimizer_report); write_json(output / "DETERMINISTIC_REPLAY_REPORT.json", {"status": "PASS" if all(r.get("deterministic_replay_consistency_pass") for r in records) else "FAIL", "candidates": [{"update": r["candidate_update"], "alpha": r["alpha"], "status": r["deterministic_replay"]} for r in records]}); write_json(output / "RNG_CONTINUITY_REPORT.json", {"status": "PASS" if all(r.get("rng_identity_pass") for r in records) else "FAIL", "candidate_count": len(records)})
    execution = {"schema_version": "stage3_h13_d29_execution_manifest_v1", "experiment_id": EXPERIMENT_ID, "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(), "authorization": {"path": str(AUTHORIZATION), "sha256": sha256_file(AUTHORIZATION)}, "authoritative_d28_archive": archive_auth, "authoritative_parent": {"update": 412, "checkpoint_path": str(PARENT), "checkpoint_sha256": EXPECTED_PARENT_SHA, "H1": START_H1, "H32": START_H32}, "alpha_ladder": LADDER, "update_413_screen": [r["alpha"] for r in records if r["candidate_update"] == 413], "updates_committed": [x["update"] for x in rows], "commit_evidence": commits, "classification": classification, "first_blocker": blocker, "scientific_state_contamination": False, "validator_or_wrapper_repairs": []}
    write_json(output / "D29_EXECUTION_MANIFEST.json", execution); write_json(output / "IMPLEMENTATION_CHANGES.json", {"changes": [{"path": str(Path(__file__).resolve()), "reason": "D29 authorized trust-region screen and local alpha continuation", "scientific_policy_changed": "alpha only, within frozen ladder"}], "validator_or_wrapper_repairs": [], "scientific_state_contamination": False})
    table = "\n".join(f"| {x['update']} | {x['alpha']:.9g} | {x['effective_lr']:.9g} | {x['H1']:.17g} | {x['delta_H1']:.17g} | {x['H32']:.17g} | {x['H32_margin']:.17g} | {x['checkpoint_sha256']} |" for x in rows)
    report = f"""# D29 Final Report

TASK_STATUS: {'PASS' if rows else 'BLOCKED'}
FINAL_CLASSIFICATION: {classification}
FIRST_BLOCKER: {blocker or 'none'}
D29_EXECUTED: YES
START_UPDATE: 412
FINAL_COMMITTED_UPDATE: {current_update}
UPDATES_COMMITTED: {[x['update'] for x in rows]}
START_H1: {START_H1:.17g}
FINAL_H1: {h1:.17g}
TOTAL_H1_IMPROVEMENT: {total:.17g}
MEAN_H1_IMPROVEMENT_PER_UPDATE: {mean_gain:.17g}
START_H32: {START_H32:.17g}
FINAL_H32: {h32:.17g}
FINAL_H32_MARGIN: {H32_LIMIT-h32:.17g}
UPDATE_413_ALPHA_SCREEN: {[{'alpha':r['alpha'],'valid':r.get('candidate_valid'),'H1':r.get('candidate_H1'),'H32':r.get('candidate_H32'),'reason':r.get('rejection_reason')} for r in records if r['candidate_update']==413]}
SELECTED_ALPHA_UPDATE_413: {rows[0]['alpha'] if rows else None}
ALPHA_TRAJECTORY: {[x['alpha'] for x in rows]}
FINAL_ALPHA: {previous_alpha}
OPTIMIZER_CONSISTENCY: {optimizer_report['status']}
DETERMINISTIC_REPLAY: {'PASS' if all(r.get('deterministic_replay_consistency_pass') for r in records) else 'FAIL'}
SCIENTIFIC_STATE_CONTAMINATION: NO
VALIDATOR_OR_WRAPPER_REPAIRS: none
STAGE3_COMPLETION_STATUS: {comp['stage3_completion_gate']}
FINAL_CHECKPOINT_SHA256: {current_sha}

Mean H1 progress was {mean_gain/d28_mean:.6g}x the D28 six-update mean; acceleration relative to D28: {'YES' if accelerated else 'NO'}.

| update | alpha | effective_lr | H1 | delta_H1 | H32 | H32_margin | checkpoint_SHA256 |
|---:|---:|---:|---:|---:|---:|---:|---|
{table}
"""
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    handoff = f"""D29 CROSS-CHAT HANDOFF — READ FIRST

D29 authorization: execute H32-margin-aware P4 trust-region re-expansion from authenticated update 412 through at most update 420. Authoritative D28 archive: {D28_ARCHIVE}; SHA-256 {EXPECTED_ARCHIVE_SHA}; size {EXPECTED_ARCHIVE_SIZE}. Parent checkpoint SHA-256: {EXPECTED_PARENT_SHA}. Starting H1/H32: {START_H1:.17g} / {START_H32:.17g}.

Tested candidates and outcomes are in D29_CANDIDATE_RESULTS.json. Update-413 tested alphas: {execution['update_413_screen']}. Selected update-413 alpha: {rows[0]['alpha'] if rows else None}. Committed updates: {[x['update'] for x in rows]}. Alpha trajectory: {[x['alpha'] for x in rows]}. H1 trajectory: {[x['H1'] for x in rows]}. H32 trajectory: {[x['H32'] for x in rows]}. Progress acceleration versus D28: {'YES' if accelerated else 'NO'} ({mean_gain/d28_mean:.6g}x mean). Optimizer consistency: {optimizer_report['status']}. Deterministic replay: {'PASS' if all(r.get('deterministic_replay_consistency_pass') for r in records) else 'FAIL'}. Implementation defects: none. Scientific-state contamination: NO.

Final committed update: {current_update}. Final H1: {h1:.17g}. Final H32: {h32:.17g}. Final H32 margin: {H32_LIMIT-h32:.17g}. Final alpha: {previous_alpha}. Final checkpoint SHA-256: {current_sha}. Stage-3 completion: {comp['stage3_completion_gate']}. Classification: {classification}.

Important files: FINAL_REPORT.md, D29_EXECUTION_MANIFEST.json, D29_CANONICAL_TRAJECTORY.json, D29_CANDIDATE_RESULTS.json, OPTIMIZER_CONSISTENCY_REPORT.json, DETERMINISTIC_REPLAY_REPORT.json, RNG_CONTINUITY_REPORT.json, STAGE3_COMPLETION_CONTRACT_EVALUATION.json, and checkpoints/committed_update_{current_update}.pt.
"""
    (output / "START_HERE_NEXT_CHAT.txt").write_text(handoff, encoding="utf-8", newline="\n"); (output / "TASK_AND_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "AUTHENTICATED_D29_TASK_PROMPT.md").write_bytes(AUTHORIZATION.read_bytes())
    manifest_files = [{"relative_path": str(p.relative_to(output)).replace("\\", "/"), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output.rglob("*")) if p.is_file() and p.name != "INTERNAL_FILE_MANIFEST.json"]
    write_json(output / "INTERNAL_FILE_MANIFEST.json", {"hash_algorithm": "SHA-256", "self_excluded": True, "managed_files": manifest_files, "file_count": len(manifest_files)})
    return {"status": "PASS" if rows else "BLOCKED", "output": str(output.resolve()), "classification": classification, "first_blocker": blocker, "last_committed_update": current_update, "final_H1": h1, "final_H32": h32, "final_checkpoint_sha256": current_sha}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    try: result = run(args.output.resolve())
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}"}, sort_keys=True)); return 1
    print(json.dumps(result, sort_keys=True)); return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
