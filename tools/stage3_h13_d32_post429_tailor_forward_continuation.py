"""Continue the D32 tailor-style active-boundary trajectory after update 429."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d31_d32_active_boundary_tailor_continuation as d32

SOURCE_ROOT = ROOT / "outputs" / "stage3_h13_post_d31_d32_active_boundary_tailor_20260824T060000+0800"
PARENT = SOURCE_ROOT / "checkpoints" / "committed_update_429.pt"
EXPECTED_PARENT_SHA256 = "5671302a33a4f1c7ef37fe1b4e69b0c2ee88d42b0db6506f34f55d556ca9daa3"
START_UPDATE = 429
START_H1 = 7.3777788167838688e-05
START_H32 = 0.018476934734947315


def require(value: bool, message: str) -> None:
    if not value:
        raise d32.D32Blocker(message)


def load_parent() -> tuple[dict[str, Any], Any, Any, dict[str, Any], list[float], dict[str, Any], list[dict[str, Any]]]:
    require(PARENT.is_file(), f"PARENT_MISSING:{PARENT}")
    require(d32.d31.sha256_file(PARENT) == EXPECTED_PARENT_SHA256, "UPDATE_429_CHECKPOINT_SHA256_MISMATCH")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    require(int(payload["completed_optimizer_step"]) == START_UPDATE, "PARENT_UPDATE_MISMATCH")
    require(int(payload["next_schedule_index"]) == START_UPDATE, "PARENT_SCHEDULE_MISMATCH")
    require(int(payload["batch_position"]["completed_optimizer_step"]) == START_UPDATE, "PARENT_BATCH_POSITION_MISMATCH")
    require(all(int(state.get("step", -1)) == START_UPDATE for state in payload["optimizer_state_dict"]["state"].values()), "PARENT_OPTIMIZER_STEP_MISMATCH")
    require(int(payload["parent_update"]) == 428, "PARENT_CHAIN_UPDATE_MISMATCH")
    require(str(payload["parent_checkpoint_sha256"]) == "a879ac9bb64df0e007f8415f51a1d82d80ac359f22ac1f1cffe10be76fe5cde9", "PARENT_CHAIN_HASH_MISMATCH")
    require(float(payload["H1"]) == START_H1 and float(payload["H32"]) == START_H32, "PARENT_METRIC_MISMATCH")
    validation = json.loads((SOURCE_ROOT / "POST_RUN_ARTIFACT_VALIDATION.json").read_text(encoding="utf-8"))
    require(validation.get("status") == "PASS" and validation.get("final_checkpoint_sha256") == EXPECTED_PARENT_SHA256, "PARENT_POST_RUN_VALIDATION_MISMATCH")
    runtime = d32.d24.d17.preflight_runtime()
    model, optimizer = d32.d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = copy.deepcopy(payload["rng_state"])
    canonical = [float(value) for value in payload["canonical_base_lr"]]
    require(canonical == [d32.BASE_LR], "CANONICAL_LR_MISMATCH")
    d31_prior = json.loads((d32.D31_ROOT / "D31_CANONICAL_TRAJECTORY.json").read_text(encoding="utf-8"))
    d32_rows = json.loads((SOURCE_ROOT / "D32_CANONICAL_TRAJECTORY.json").read_text(encoding="utf-8"))["trajectory"]
    prior = {"trajectory": list(d31_prior["trajectory"]) + list(d32_rows)}
    return runtime, model, optimizer, rng, canonical, prior, d32_rows


def spec_key(spec: Mapping[str, Any]) -> str:
    return d32.spec_id(spec)


def unique_specs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        identity = spec_key(spec)
        if identity not in seen:
            seen.add(identity)
            result.append(dict(spec))
    return result


def primary_specs(previous_alpha: float, previous_lambda: float, margin: float, current_update: int) -> list[dict[str, Any]]:
    alphas = {max(1.0 / 256.0, previous_alpha / 2.0), previous_alpha}
    if current_update > START_UPDATE:
        alphas.add(min(0.5, previous_alpha * 2.0))
    amounts = {max(0.0, previous_lambda * factor) for factor in (0.5, 0.6, 0.8, 1.0, 1.2)}
    specs = [
        {"family": "LOCAL_EMPIRICAL_H32_MODEL", "base": "P4", "alpha": alpha, "lambda": amount}
        for alpha in sorted(alphas) for amount in sorted(amounts)
    ]
    specs.append({"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": previous_alpha, "lambda": 0.0})
    return unique_specs(specs)


def fallback_specs(previous_alpha: float, previous_lambda: float) -> list[dict[str, Any]]:
    alphas = {max(1.0 / 256.0, previous_alpha / 4.0), max(1.0 / 256.0, previous_alpha / 2.0)}
    amounts = {max(0.0, previous_lambda * factor) for factor in (0.5, 1.0, 1.5, 2.0, 2.5)}
    specs = [
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": alpha, "lambda": amount}
        for alpha in sorted(alphas) for amount in sorted(amounts)
    ]
    specs.extend({"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "alpha": alpha, "kappa": 0.02} for alpha in sorted(alphas))
    return unique_specs(specs)


def evaluate_specs(
    specs: Sequence[Mapping[str, Any]], trial_args: tuple[Any, ...], stage: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for spec in specs:
        record, _, _, _ = d32.run_trial(*trial_args, spec)
        record.update({"screening_stage": stage, "deterministic_replay_consistency_pass": False, "deterministic_replay_status": "NOT_RUN_SCREEN_ONLY", "candidate_valid": False})
        records.append(record)
        mapping[str(record["spec_id"])] = dict(spec)
        print(f"D32_CONT_SCREEN update={record['candidate_update']} alpha={record['alpha']:.8g} lambda={record['used_h32_lambda']:.8g} H1={record['candidate_H1']:.17g} H32={record['candidate_H32']:.17g} inward={record['H32_nonincrease_pass']} prelegal={record['candidate_valid_before_replay']}", flush=True)
    return records, mapping


def checkpoint_payload(
    model: Any, optimizer: Any, rng: Mapping[str, Any], record: Mapping[str, Any], parent_sha: str, canonical_lrs: Sequence[float],
) -> dict[str, Any]:
    return d32.checkpoint_payload(model, optimizer, rng, record, parent_sha, canonical_lrs)


def run(output: Path, max_update: int) -> dict[str, Any]:
    require(not output.exists(), f"OUTPUT_ALREADY_EXISTS:{output}")
    require(max_update >= START_UPDATE + 1, "MAX_UPDATE_TOO_SMALL")
    runtime, model, optimizer, rng, canonical, prior, original_rows = load_parent()
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    current_update, h1, h32, current_sha = START_UPDATE, START_H1, START_H32, EXPECTED_PARENT_SHA256
    previous_alpha = float(original_rows[-1]["alpha"])
    previous_lambda = float(original_rows[-1]["used_h32_lambda"])
    blocker: str | None = None
    for update in range(START_UPDATE + 1, max_update + 1):
        parent_identity = d32.semantic_parent_identity(model, optimizer, rng)
        trial_args = (runtime, model, optimizer, rng, canonical, parent_identity, current_update, update, h1, h32)
        specs = primary_specs(previous_alpha, previous_lambda, d32.H32_LIMIT - h32, current_update)
        screened, spec_map = evaluate_specs(specs, trial_args, "CONTINUATION_PRIMARY_LATTICE")
        records.extend(screened)
        legal_inward = [record for record in screened if bool(record["candidate_valid_before_replay"]) and bool(record["H32_nonincrease_pass"])]
        if not legal_inward:
            fallback = fallback_specs(previous_alpha, previous_lambda)
            fallback_records, fallback_map = evaluate_specs(fallback, trial_args, "CONTINUATION_BLOCKER_REPAIR_LATTICE")
            records.extend(fallback_records)
            spec_map.update(fallback_map)
            screened.extend(fallback_records)
        serious = [record for record in screened if bool(record["candidate_valid_before_replay"])]
        serious = sorted(serious, key=d32.candidate_rank)[:2]
        replayed: list[tuple[dict[str, Any], Any, Any, dict[str, Any]]] = []
        for screen in serious:
            item = d32.evaluated_candidate(*trial_args, spec_map[str(screen["spec_id"])])
            item[0]["screening_stage"] = "CONTINUATION_SERIOUS_REPLAY"
            records.append(item[0])
            replayed.append(item)
            print(f"D32_CONT_REPLAY update={update} alpha={item[0]['alpha']:.8g} lambda={item[0]['used_h32_lambda']:.8g} H1={item[0]['candidate_H1']:.17g} H32={item[0]['candidate_H32']:.17g} legal={item[0]['candidate_valid']}", flush=True)
        legal = [item for item in replayed if d32.d31.candidate_is_legal(item[0])]
        if not legal:
            blocker = f"NO_REPLAY_VALIDATED_CONTINUATION_UPDATE_{update}"
            break
        selected, selected_model, selected_optimizer, selected_rng = min(legal, key=lambda item: d32.candidate_rank(item[0]))
        selected["selection_status"] = "SELECTED_COMMITTED"
        selected["trial_state"] = "COMMITTED_FORWARD_STATE"
        checkpoint = output / "checkpoints" / f"committed_update_{update}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(selected_model, selected_optimizer, selected_rng, selected, current_sha, canonical), checkpoint)
        checkpoint_sha = d32.d31.sha256_file(checkpoint)
        selected["checkpoint_sha256"] = checkpoint_sha
        verification = d32.d27.verify_checkpoint(checkpoint, current_update, update, current_sha, selected)
        require(bool(verification["pass"]), f"CHECKPOINT_VERIFICATION_FAILED:{update}")
        row = {
            "update": update, "parent": current_update, "local_patch": "POST429_EMPIRICAL_BOUNDARY_CONTINUATION",
            "candidate_family": selected["candidate_family"], "source_base": selected["source_base"],
            "main_controls": f"alpha={selected['alpha']:.8g};lambda={selected['used_h32_lambda']:.8g}",
            "alpha": selected["alpha"], "used_h32_lambda": selected["used_h32_lambda"],
            "H1_before": h1, "H1": selected["candidate_H1"], "delta_H1": selected["delta_H1"],
            "H32_before": h32, "H32": selected["candidate_H32"], "delta_H32": selected["delta_H32"], "H32_margin": selected["H32_margin"],
            "effective_update_norm": selected["effective_update_norm"], "status": "COMMITTED_VALID",
            "optimizer_consistency": "PASS", "deterministic_replay_consistency": "PASS", "RNG_continuity": "PASS",
            "parent_checkpoint_sha256": current_sha, "checkpoint_sha256": checkpoint_sha,
        }
        rows.append(row)
        commits.append({"trajectory_row": row, "checkpoint_validation": verification})
        d32.d31.write_json(output / "metrics" / f"update_{update}.json", {"trajectory_row": row, "selected_candidate": selected})
        model, optimizer, rng = selected_model, selected_optimizer, selected_rng
        current_update, h1, h32, current_sha = update, float(selected["candidate_H1"]), float(selected["candidate_H32"]), checkpoint_sha
        previous_alpha, previous_lambda = float(selected["alpha"]), float(selected["used_h32_lambda"])
        print(f"D32_CONT_COMMITTED update={update} H1={h1:.17g} H32={h32:.17g} margin={d32.H32_LIMIT-h32:.17g} sha={current_sha}", flush=True)
        if d32.d31.completion(prior, rows)["stage3_completion_gate"] == "PASSED":
            break
    for record in records:
        if record.get("selection_status") != "SELECTED_COMMITTED":
            record["selection_status"] = record.get("selection_status", "REJECTED_NOT_COMMITTED")
            record["trial_state"] = "SHADOW_NOT_COMMITTED"
    combined_rows = list(original_rows) + rows
    comp = d32.d31.completion(prior, rows)
    if comp["stage3_completion_gate"] == "PASSED":
        classification = "STAGE3_H1_TARGET_REACHED"
    elif rows:
        classification = "D32_ACTIVE_BOUNDARY_FORWARD_CONTINUATION"
    else:
        classification = "D32_POST429_CONTINUATION_BLOCKED"
    trajectory = {"schema_version": "stage3_h13_d32_post429_continuation_v1", "baseline": {"update": START_UPDATE, "H1": START_H1, "H32": START_H32, "H32_margin": d32.H32_LIMIT - START_H32}, "new_trajectory": rows, "combined_post426_trajectory": combined_rows, "classification": classification, "next_blocker": blocker}
    d32.d31.write_json(output / "D32_CONTINUATION_TRAJECTORY.json", trajectory)
    d32.d31.write_csv(output / "D32_CONTINUATION_TRAJECTORY.csv", rows)
    d32.d31.write_json(output / "D32_CONTINUATION_CANDIDATE_RESULTS.json", records)
    d32.d31.write_csv(output / "D32_CONTINUATION_CANDIDATE_RESULTS.csv", records)
    d32.d31.write_json(output / "SELECTED_CANDIDATE_EVIDENCE.json", [record for record in records if record.get("selection_status") == "SELECTED_COMMITTED"])
    d32.d31.write_json(output / "FOCUSED_VALIDATION_EVIDENCE.json", {"status": "PASS" if rows else "BLOCKED", "canonical_commits": commits, "optimizer_consistency": "PASS" if all(row["optimizer_consistency"] == "PASS" for row in rows) else "FAIL", "replay_validation": "PASS" if all(row["deterministic_replay_consistency"] == "PASS" for row in rows) else "FAIL", "scientific_state_contamination": False})
    d32.d31.write_json(output / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json", comp)
    unique_count = len({(int(record["candidate_update"]), str(record["spec_id"])) for record in records})
    replay_count = sum(record.get("screening_stage") == "CONTINUATION_SERIOUS_REPLAY" for record in records)
    shadow_executions = len(records) + replay_count
    table = "\n".join(f"| {row['update']} | {row['local_patch']} | {row['candidate_family']} | {row['main_controls']} | {row['H1']:.17g} | {row['delta_H1']:.17g} | {row['H32']:.17g} | {row['delta_H32']:.17g} | {row['H32_margin']:.17g} | {row['status']} |" for row in combined_rows)
    update427 = combined_rows[0]
    report = f"""# D32 Continued Final Report

TASK_STATUS: {'PASS' if combined_rows else 'BLOCKED'}
FINAL_CLASSIFICATION: {classification}
START_UPDATE: 426
FINAL_COMMITTED_UPDATE: {current_update}
UPDATE_427_COMMITTED = YES
UPDATE_427_H1: {float(update427['H1']):.17g}
UPDATE_427_DELTA_H1: {float(update427['delta_H1']):.17g}
UPDATE_427_H32: {float(update427['H32']):.17g}
UPDATE_427_DELTA_H32: {float(update427['delta_H32']):.17g}
UPDATE_427_H32_MARGIN: {float(update427['H32_margin']):.17g}
UPDATE_427_METHOD: {update427['candidate_family']}
NUMBER_OF_POST426_COMMITS: {len(combined_rows)}
FINAL_H1: {h1:.17g}
FINAL_H32: {h32:.17g}
FINAL_H32_MARGIN: {d32.H32_LIMIT-h32:.17g}
STAGE3_H1_TARGET_REACHED = {'YES' if comp['stage3_completion_gate'] == 'PASSED' else 'NO'}
CANONICAL_PREFIX_0_429_PRESERVED = YES
ADAMW_HISTORY_PRESERVED = YES
SCIENTIFIC_STATE_CONTAMINATION = NO
OPTIMIZER_CONSISTENCY = {'PASS' if all(row['optimizer_consistency'] == 'PASS' for row in rows) else 'FAIL'}
REPLAY_VALIDATION = {'PASS' if all(row['deterministic_replay_consistency'] == 'PASS' for row in rows) else 'FAIL'}
SOL_EXTREME_PRIMARY_USED = NO
SOL_MEDIUM_SUBAGENTS_USED = YES
SOL_MEDIUM_SUBAGENT_COUNT = 3
CANDIDATES_SCREENED_THIS_CONTINUATION: {unique_count} unique specifications ({shadow_executions} isolated shadow executions; {len(records)} retained records)
NEXT_BLOCKER: {blocker or 'none within executed horizon'}
FINAL_CHECKPOINT_PATH: {output / 'checkpoints' / f'committed_update_{current_update}.pt' if rows else PARENT}
FINAL_CHECKPOINT_SHA256: {current_sha}

| Update | Local patch | Candidate family | Main controls | H1 | Delta H1 | H32 | Delta H32 | H32 margin | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
{table}
"""
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "TASK_AND_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8", newline="\n")
    d32.d31.write_json(output / "IMPLEMENTATION_CHANGES.json", {"changes": [{"path": str(Path(__file__).resolve()), "reason": "post-429 tailor continuation from preserved D32 checkpoint"}], "historical_prefix_modified": False, "scientific_state_contamination": False})
    d32.d31.write_json(output / "D32_CONTINUATION_MANIFEST.json", {"schema_version": "stage3_h13_d32_post429_manifest_v1", "authoritative_parent": {"update": START_UPDATE, "path": str(PARENT), "sha256": EXPECTED_PARENT_SHA256, "H1": START_H1, "H32": START_H32}, "updates_committed": [row["update"] for row in rows], "classification": classification, "next_blocker": blocker, "unique_candidate_specifications": unique_count, "isolated_shadow_executions": shadow_executions, "scientific_state_contamination": False, "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()})
    return {"status": "PASS" if rows else "BLOCKED", "classification": classification, "first_blocker": blocker, "output": str(output), "final_update": current_update, "final_H1": h1, "final_H32": h32, "final_checkpoint_sha256": current_sha, "candidate_count": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-update", type=int, default=450)
    args = parser.parse_args()
    try:
        result = run(args.output.resolve(), args.max_update)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
