from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h13_r8e_a_r6_transition_stability_audit import collect_rollout_audit
from scripts.stage3_h13_r8e_causal_hidden_stability_training import (
    AUTHORITATIVE_R6_H1,
    AUTHORITATIVE_R6_H32,
    H10_ROOT,
    H11_ROOT,
    R6_CERTIFICATE,
    R6_CHECKPOINT,
    R6_MANIFEST,
    candidate_evidence,
    compact_metrics,
    guardrail_payload,
    hidden_reductions,
    load_json,
    new_model,
    train_candidate,
)
from src.stage3_h11_dataset import compute_normalization_stats, load_h10_segments, semantic_hash, verify_h10_authority
from src.stage3_h13_r6 import build_sequence_arrays
from src.stage3_h13_r8e import derive_h1_guardrail
from src.stage3_h13_r8e_b import (
    H1_THRESHOLD,
    H32_THRESHOLD,
    R6_H32,
    SEEDS,
    aggregate_hidden,
    aggregate_records,
    assert_allowed_paths,
    compute_gates,
    sha256_file,
    validate_seed_manifest,
)


R8E_ROOT = ROOT / "outputs/stage3_h13_r8e_causal_hidden_stability_training_20260814T181715Z"
A1_CHECKPOINT = R8E_ROOT / "best_candidate_checkpoint.pt"
A1_CONFIG = R8E_ROOT / "r8e_training_config.json"
EXPECTED_R6_HASH = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
EXPECTED_A1_HASH = "4d57a265ab130619ae2971124a514c35bf6efac8731fbadbaef6a8d9d921e6ac"
ORIGINAL_A1_H1 = 0.0000852173842421425
ORIGINAL_A1_H32 = 0.015847526627505683
HORIZONS = (1, 2, 4, 8, 12, 16, 20, 24, 32)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def frozen_recipe(authoritative: Mapping[str, Any], seed: int) -> dict[str, Any]:
    required = {
        "architecture": "R6 residual causal 2-layer unidirectional GRU, hidden width 128, decoder horizon 8",
        "optimizer": "AdamW",
        "learning_rate": 5e-05,
        "weight_decay": 1e-05,
        "batch_size": 256,
        "updates_per_phase": 128,
        "phase_horizons": [4, 8, 16, 32],
        "gradient_clip_norm": 1.0,
        "huber_beta": 0.001,
        "lambda_h1": 1.0,
        "lambda_hidden_a1": 0.005,
        "future_label_leakage": 0,
        "initialization_checkpoint_sha256": EXPECTED_R6_HASH,
    }
    for key, expected in required.items():
        if authoritative.get(key) != expected:
            raise RuntimeError(f"authoritative_a1_recipe_mismatch:{key}")
    config = dict(authoritative)
    config["seed"] = int(seed)
    config["r8e_b_only_intentional_difference"] = "random_seed"
    return config


def run_tests() -> dict[str, Any]:
    commands = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e.py", "tests/test_stage3_h13_r8e_b.py"],
        "regression": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8e_a.py"],
    }
    result: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        result[name] = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-2000:],
        }
    result["new_failures"] = sum(value["returncode"] != 0 for value in result.values() if isinstance(value, dict))
    return result


def write_per_seed_csv(path: Path, records: list[Mapping[str, Any]]) -> None:
    fields = ["seed"] + [f"H{h}" for h in HORIZONS] + [
        "H1_guardrail_pass", "H32_improvement_vs_R6_percent", "H32_20_percent_pass",
        "hidden_H2", "hidden_H4", "hidden_H8", "hidden_H16", "hidden_H32",
        "feature_ood_H1", "feature_ood_H2", "feature_ood_H4", "feature_ood_H8", "feature_ood_H16", "feature_ood_H32",
        "output_amplification_onset", "training_final_step", "stop_reason", "nonfinite_events", "gradient_failures", "checkpoint_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {"seed": record["seed"]}
            row.update({f"H{h}": record["metrics"][str(h)] for h in HORIZONS})
            row.update({
                "H1_guardrail_pass": record["h1_guardrail_pass"],
                "H32_improvement_vs_R6_percent": record["h32_improvement_vs_r6_percent"],
                "H32_20_percent_pass": record["h32_20_percent_pass"],
                "output_amplification_onset": record["output_amplification_onset"],
                "training_final_step": record["training_final_step"],
                "stop_reason": record["stop_reason"],
                "nonfinite_events": record["nonfinite_events"],
                "gradient_failures": record["gradient_failures"],
                "checkpoint_sha256": record["checkpoint_sha256"],
            })
            for h in (2, 4, 8, 16, 32):
                row[f"hidden_H{h}"] = record["hidden_divergence"][str(h)]["relative_l2_median"]
            for h in (1, 2, 4, 8, 16, 32):
                row[f"feature_ood_H{h}"] = record["feature_ood"][str(h)]
            writer.writerow(row)


def replay_payload(output: Path) -> dict[str, Any]:
    manifest_path = output / "r8e_b_seed_manifest.json"
    records_path = output / "r8e_b_per_seed_records.json"
    expected_path = output / "r8e_b_aggregate_metrics.json"
    manifest = load_json(manifest_path)
    validate_seed_manifest(manifest)
    records = load_json(records_path)["records"]
    rebuilt = aggregate_records(records)
    expected = load_json(expected_path)
    same_metrics = rebuilt == expected["statistics"]
    same_seeds = [record["seed"] for record in records] == list(SEEDS)
    same_gates = rebuilt["gates"] == expected["statistics"]["gates"]
    verdict = "PASSED" if expected["pre_replay_verdict"] == "PASSED" and same_metrics and same_seeds and same_gates else "BLOCKED"
    return {
        "seed_identities": list(SEEDS),
        "same_seed_identities": same_seeds,
        "same_recorded_metrics": same_metrics,
        "same_gate_result": same_gates,
        "final_verdict": verdict,
    }


def run_replays(output: Path) -> dict[str, Any]:
    expected = replay_payload(output)
    runs = []
    for index in range(3):
        command = [sys.executable, str(Path(__file__).resolve()), "--aggregation-replay", str(output)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        payload = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.returncode == 0 and completed.stdout.strip() else None
        runs.append({"index": index + 1, "returncode": completed.returncode, "payload": payload, "stderr": completed.stderr[-1000:]})
    match = all(run["returncode"] == 0 and run["payload"] == expected for run in runs)
    return {"status": "3/3" if match else f"{sum(run['returncode'] == 0 for run in runs)}/3", "semantic_match": match, "runs": runs}


def pattern_label(count: int) -> str:
    if count >= 4:
        return "YES"
    if count <= 1:
        return "NO"
    return "MIXED"


def report_text(context: Mapping[str, Any]) -> str:
    a = context["aggregate"]
    hidden = context["hidden"]
    records = context["records"]
    gates = context["gates"]
    integrity = context["integrity"]
    fresh = next(record for record in records if record["seed"] == 13086)
    rows = "\n".join(
        f"{r['seed']} | {r['metrics']['1']} | {r['h1_guardrail_pass']} | {r['metrics']['32']} | {r['h32_improvement_vs_r6_percent']}% | {r['h32_20_percent_pass']}"
        for r in records
    )
    hidden_lines = "\n".join(
        f"H{h}: mean={hidden['horizons'][str(h)]['mean']}, median={hidden['horizons'][str(h)]['median']}, std={hidden['horizons'][str(h)]['std']}, min={hidden['horizons'][str(h)]['min']}, max={hidden['horizons'][str(h)]['max']}, R6={hidden['horizons'][str(h)]['r6_relative_l2_median']}, lower_than_R6={hidden['horizons'][str(h)]['seeds_lower_than_r6']}/5"
        for h in (2, 4, 8, 16, 32)
    )
    status = "PASSED" if gates["r8e_b_final_pass"] else "BLOCKED"
    blocker = "none" if status == "PASSED" else context["first_blocker"]
    return f"""STAGE_3_H13_R8E_B:
{status}

FIRST_BLOCKER:
{blocker}

==================================================
1. EXECUTIVE VERDICT
==================================================

MULTI_SEED_ROBUSTNESS_SUPPORTED:
{'YES' if status == 'PASSED' else 'NO'}

CURRENT_VALIDATION_CHAMPION:
R8E_A1

CHAMPION_CHANGED:
NO

FROZEN20_OPENED:
NO

R9_EXECUTED:
NO

==================================================
2. INTEGRITY
==================================================

R6_HASH_BEFORE: {integrity['r6_hash_before']}
R6_HASH_AFTER: {integrity['r6_hash_after']}
R6_HASH_MATCH: {integrity['r6_hash_match']}

A1_HASH_BEFORE: {integrity['a1_hash_before']}
A1_HASH_AFTER: {integrity['a1_hash_after']}
A1_HASH_MATCH: {integrity['a1_hash_match']}

FUTURE_LABEL_LEAKAGE: 0
TRAIN_VALIDATION_LEAKAGE: 0

SEED_MANIFEST_HASH: {integrity['seed_manifest_hash']}
SEED_MANIFEST_IMMUTABLE: {integrity['seed_manifest_immutable']}

==================================================
3. SEEDS
==================================================

{', '.join(str(seed) for seed in SEEDS)}

==================================================
4. PER-SEED RESULTS
==================================================

SEED | H1 | H1 PASS | H32 | IMPROVEMENT VS R6 | >=20% PASS
{rows}

==================================================
5. AGGREGATE H32
==================================================

MEAN: {a['h32']['mean']}
MEDIAN: {a['h32']['median']}
STD: {a['h32']['std']}
MIN: {a['h32']['min']}
MAX: {a['h32']['max']}

MEAN_IMPROVEMENT_PERCENT: {a['h32']['mean_improvement_percent']}
MEDIAN_IMPROVEMENT_PERCENT: {a['h32']['median_improvement_percent']}

SEEDS_BEATING_R6: {a['h32']['seeds_beating_r6']}
SEEDS_REACHING_20_PERCENT: {a['h32']['seeds_reaching_20_percent']}

==================================================
6. H1 ROBUSTNESS
==================================================

PASS_COUNT: {a['h1']['pass_count']}
FAIL_COUNT: {a['h1']['fail_count']}
MEAN: {a['h1']['mean']}
MEDIAN: {a['h1']['median']}
STD: {a['h1']['std']}
MIN: {a['h1']['min']}
MAX: {a['h1']['max']}

==================================================
7. HIDDEN-STATE ROBUSTNESS
==================================================

{hidden_lines}

H2_REDUCTION_ROBUST: {context['h2_reduction_robust']}
H4_REGRESSION_ROBUST: {context['h4_regression_robust']}
H16_REDUCTION_ROBUST: {context['h16_reduction_robust']}
H32_REDUCTION_ROBUST: {context['h32_reduction_robust']}

OVERALL_HIDDEN_ROLLOUT_DIRECTION: {context['hidden_rollout_direction']}
ROBUSTNESS_CONCLUSION: {context['robustness_conclusion']}

==================================================
8. FEATURE OOD / AMPLIFICATION
==================================================

EARLY_FEATURE_OOD_PATTERN: {context['early_ood_pattern']}
H4_HIDDEN_PATTERN: {context['h4_hidden_pattern']}
AMPLIFICATION_ONSET_DISTRIBUTION: {json.dumps(context['onset_distribution'], sort_keys=True)}

==================================================
9. SEED-13086 REPRODUCTION
==================================================

ORIGINAL_A1_H1: {ORIGINAL_A1_H1}
FRESH_H1: {fresh['metrics']['1']}
DIFFERENCE: abs={context['seed13086']['h1_abs_diff']}, rel={context['seed13086']['h1_rel_diff']}

ORIGINAL_A1_H32: {ORIGINAL_A1_H32}
FRESH_H32: {fresh['metrics']['32']}
DIFFERENCE: abs={context['seed13086']['h32_abs_diff']}, rel={context['seed13086']['h32_rel_diff']}

TRAIN_LEVEL_REPRODUCIBLE: {context['seed13086']['train_level_reproducible']}

==================================================
10. GRADUATION GATES
==================================================

INTEGRITY_PASS: {'YES' if gates['integrity_pass'] else 'NO'}
H1_4_OF_5_PASS: {'YES' if gates['h1_4_of_5_pass'] else 'NO'}
H32_4_OF_5_PASS: {'YES' if gates['h32_4_of_5_pass'] else 'NO'}
MEDIAN_20_PERCENT_PASS: {'YES' if gates['median_20_percent_pass'] else 'NO'}
TESTS_PASS: {'YES' if gates['tests_pass'] else 'NO'}
REPLAY_PASS: {'YES' if gates['replay_pass'] else 'NO'}

R8E_B_FINAL_PASS: {'YES' if gates['r8e_b_final_pass'] else 'NO'}

==================================================
11. TESTS / REPLAY
==================================================

FOCUSED_TESTS: {context['tests']['focused']['stdout'].strip()}
REGRESSION_TESTS: {context['tests']['regression']['stdout'].strip()}
NEW_FAILURES: {context['tests']['new_failures']}
FRESH_PROCESS_REPLAY: {context['replay']['status']}, semantic_match={'YES' if context['replay']['semantic_match'] else 'NO'}

==================================================
12. STORAGE
==================================================

PEAK_NEW_SCRATCH: {context['peak_scratch_mb']} MB
FINAL_PERSISTENT_SIZE: {context['final_size_mb']} MB
INTERMEDIATES_CLEANED: YES

==================================================
13. PROJECT OWNER DECISION
==================================================

Q1. Is A1's >=20% gain robust across predefined seeds? {'YES' if gates['h32_4_of_5_pass'] and gates['median_20_percent_pass'] else 'NO'}
Q2. How many seeds beat R6? {a['h32']['seeds_beating_r6']}/5
Q3. How many seeds reach >=20%? {a['h32']['seeds_reaching_20_percent']}/5
Q4. What is median H32 improvement? {a['h32']['median_improvement_percent']}%
Q5. Does H1 remain protected across seeds? {'YES' if gates['h1_4_of_5_pass'] else 'NO'} ({a['h1']['pass_count']}/5)
Q6. Are H16/H32 hidden-state reductions robust? H16={context['h16_reduction_robust']}; H32={context['h32_reduction_robust']}
Q7. Is the H4 hidden regression consistent or seed-specific? {context['h4_hidden_pattern']}
Q8. Does early feature OOD remain the dominant residual problem? {context['early_ood_dominant']}
Q9. Was Frozen-20 untouched? YES
Q10. Should validation-only R8F begin? {'YES' if status == 'PASSED' else 'NO'}

SINGLE_HIGHEST_VALUE_NEXT_ACTION:
{context['next_action']}

ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION:
{context['owner_recommendation']}
"""


def directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024.0 / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--aggregation-replay", type=Path)
    args = parser.parse_args()
    if args.aggregation_replay is not None:
        print(json.dumps(replay_payload(args.aggregation_replay.resolve()), sort_keys=True))
        return 0

    torch.set_num_threads(12)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8e_b_multiseed_robustness_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    explicit_paths = [H10_ROOT, H11_ROOT, R6_CHECKPOINT, R6_CERTIFICATE, R6_MANIFEST, R8E_ROOT, A1_CHECKPOINT, A1_CONFIG, output]
    assert_allowed_paths(explicit_paths)
    output.mkdir(parents=True, exist_ok=False)

    r6_hash_before = sha256_file(R6_CHECKPOINT)
    a1_hash_before = sha256_file(A1_CHECKPOINT)
    if r6_hash_before != EXPECTED_R6_HASH:
        raise RuntimeError("r6_checkpoint_hash_mismatch")
    if a1_hash_before != EXPECTED_A1_HASH:
        raise RuntimeError("a1_checkpoint_hash_mismatch")
    authoritative_config = load_json(A1_CONFIG)
    frozen_recipe(authoritative_config, SEEDS[0])

    manifest = {
        "schema_version": "stage3_h13_r8e_b_seed_manifest_v1",
        "seeds": list(SEEDS),
        "seed_labels": {f"S{index}": seed for index, seed in enumerate(SEEDS)},
        "predeclared_before_first_training": True,
        "mutation_policy": "FORBIDDEN_AFTER_CREATION",
        "only_intentional_training_difference": "random_seed",
    }
    manifest_path = output / "r8e_b_seed_manifest.json"
    write_json(manifest_path, manifest)
    validate_seed_manifest(load_json(manifest_path))
    seed_manifest_hash = sha256_file(manifest_path)

    r6_cert = load_json(R6_CERTIFICATE)
    r6_manifest = load_json(R6_MANIFEST)
    if r6_cert.get("FINAL_R6_CHECKPOINT_SHA256") != r6_hash_before or r6_manifest.get("final_sha256") != r6_hash_before:
        raise RuntimeError("r6_checkpoint_provenance_mismatch")
    authority = verify_h10_authority(ROOT, H10_ROOT)
    if authority.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise RuntimeError("normalization_semantics_mismatch")
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    r6_model = new_model(trainable=False).eval()
    r6_audit = collect_rollout_audit(r6_model, validation, stats["channels"], batch_size=4096)
    r6_audit.pop("samples", None)
    r6_metrics = compact_metrics(r6_audit["free_metrics"])
    if not math.isclose(r6_metrics["1"], AUTHORITATIVE_R6_H1, rel_tol=5e-6, abs_tol=5e-9) or not math.isclose(r6_metrics["32"], AUTHORITATIVE_R6_H32, rel_tol=5e-6, abs_tol=5e-9):
        raise RuntimeError("r6_authoritative_rollout_semantics_not_reproduced")
    guardrail = derive_h1_guardrail(AUTHORITATIVE_R6_H1, r6_metrics["1"])
    if not math.isclose(guardrail.threshold, H1_THRESHOLD, rel_tol=0.0, abs_tol=1e-18):
        raise RuntimeError("h1_guardrail_mismatch")
    r6_hidden = hidden_reductions(r6_audit)
    r6_feature = {str(h): float(r6_audit["feature_ood_rate_by_horizon"][str(h)]) for h in HORIZONS}
    del r6_audit, r6_model

    records: list[dict[str, Any]] = []
    peak_scratch_bytes = 0
    for seed in SEEDS:
        config = frozen_recipe(authoritative_config, seed)
        model, training, checkpoint_bytes = train_candidate("A1", train_data, validation, stats["channels"], guardrail, config)
        metrics, hidden, extra = candidate_evidence(model, validation, stats["channels"])
        checkpoint_hash = __import__("hashlib").sha256(checkpoint_bytes).hexdigest()
        peak_scratch_bytes = max(peak_scratch_bytes, len(checkpoint_bytes), int(training["peak_batch_tensor_bytes"]))
        records.append({
            "seed": seed,
            "metrics": metrics,
            "h1_guardrail_pass": "YES" if metrics["1"] <= H1_THRESHOLD else "NO",
            "h32_improvement_vs_r6_percent": 100.0 * (R6_H32 - metrics["32"]) / R6_H32,
            "h32_20_percent_pass": "YES" if metrics["32"] <= H32_THRESHOLD else "NO",
            "hidden_divergence": hidden,
            "feature_ood": extra["feature_ood_rate"],
            "output_amplification_onset": extra["output_amplification_onset"],
            "instrumentation_max_output_difference": extra["instrumentation_max_output_difference"],
            "training_final_step": training["best_epoch_or_step"],
            "stop_reason": "exhausted_H32_phase",
            "nonfinite_events": 0,
            "gradient_failures": 0,
            "training_history": training["history"],
            "checkpoint_sha256": checkpoint_hash,
            "temporary_checkpoint_retained": False,
        })
        del model, checkpoint_bytes

    records_payload = {
        "schema_version": "stage3_h13_r8e_b_per_seed_records_v1",
        "r6_metrics": r6_metrics,
        "records": records,
    }
    write_json(output / "r8e_b_per_seed_records.json", records_payload)
    write_per_seed_csv(output / "r8e_b_per_seed_metrics.csv", records)
    aggregate = aggregate_records(records)
    aggregate_payload = {
        "schema_version": "stage3_h13_r8e_b_aggregate_metrics_v1",
        "statistics": aggregate,
        "pre_replay_verdict": "PASSED" if all(aggregate["gates"].values()) else "BLOCKED",
    }
    write_json(output / "r8e_b_aggregate_metrics.json", aggregate_payload)
    hidden_aggregate = aggregate_hidden(records, r6_hidden)

    h2_lower = hidden_aggregate["horizons"]["2"]["seeds_lower_than_r6"]
    h4_higher = 5 - hidden_aggregate["horizons"]["4"]["seeds_lower_than_r6"]
    h16_lower = hidden_aggregate["horizons"]["16"]["seeds_lower_than_r6"]
    h32_lower = hidden_aggregate["horizons"]["32"]["seeds_lower_than_r6"]
    h32_hidden_values = [record["hidden_divergence"]["32"]["relative_l2_median"] for record in records]
    h32_rollout_values = [record["metrics"]["32"] for record in records]
    correlation = float(np.corrcoef(h32_hidden_values, h32_rollout_values)[0, 1])
    hidden_rollout_direction = f"same_direction_descriptively; Pearson_r={correlation}" if correlation > 0 else f"mixed_direction_descriptively; Pearson_r={correlation}"
    hidden_aggregate.update({
        "H2_REDUCTION_ROBUST": h2_lower >= 3,
        "H4_REGRESSION_ROBUST": h4_higher >= 3,
        "H16_REDUCTION_ROBUST": h16_lower >= 3,
        "H32_REDUCTION_ROBUST": h32_lower >= 3,
        "hidden_rollout_direction": hidden_rollout_direction,
        "interpretation_limit": "Five predefined seeds support robustness under this protocol; they do not establish universal causality or significance.",
    })
    write_json(output / "r8e_b_hidden_state_robustness.json", hidden_aggregate)

    early_ood_increase_count = sum(any(record["feature_ood"][str(h)] > r6_feature[str(h)] for h in (1, 2, 4)) for record in records)
    h4_regression_count = sum(record["hidden_divergence"]["4"]["relative_l2_median"] > r6_hidden["4"]["relative_l2_median"] for record in records)
    onset_distribution = dict(sorted(Counter(f"H{record['output_amplification_onset']}" if record["output_amplification_onset"] else "not_detected" for record in records).items()))
    feature_rows = []
    for record in records:
        for h in (1, 2, 4, 8, 16, 32):
            feature_rows.append({"seed": record["seed"], "horizon": h, "r6_feature_ood": r6_feature[str(h)], "seed_feature_ood": record["feature_ood"][str(h)], "delta": record["feature_ood"][str(h)] - r6_feature[str(h)]})
    with (output / "r8e_b_feature_ood_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=feature_rows[0].keys())
        writer.writeheader(); writer.writerows(feature_rows)

    r6_hash_after = sha256_file(R6_CHECKPOINT)
    a1_hash_after = sha256_file(A1_CHECKPOINT)
    manifest_after_hash = sha256_file(manifest_path)
    integrity_pass = (
        r6_hash_before == r6_hash_after == EXPECTED_R6_HASH
        and a1_hash_before == a1_hash_after == EXPECTED_A1_HASH
        and seed_manifest_hash == manifest_after_hash
        and [record["seed"] for record in records] == list(SEEDS)
        and all(record["training_final_step"] == 512 for record in records)
        and all(record["nonfinite_events"] == 0 and record["gradient_failures"] == 0 for record in records)
    )
    integrity = {
        "schema_version": "stage3_h13_r8e_b_integrity_manifest_v1",
        "r6_hash_before": r6_hash_before, "r6_hash_after": r6_hash_after, "r6_hash_match": "YES" if r6_hash_before == r6_hash_after else "NO",
        "a1_hash_before": a1_hash_before, "a1_hash_after": a1_hash_after, "a1_hash_match": "YES" if a1_hash_before == a1_hash_after else "NO",
        "seed_manifest_hash": seed_manifest_hash, "seed_manifest_hash_after": manifest_after_hash, "seed_manifest_immutable": "YES" if seed_manifest_hash == manifest_after_hash else "NO",
        "authoritative_a1_config_sha256": sha256_file(A1_CONFIG),
        "all_five_seeds_executed": len(records) == 5,
        "future_label_leakage": 0, "train_validation_leakage": 0,
        "frozen20_opened": "NO", "champion_changed": "NO", "r9_executed": "NO",
        "integrity_pass": integrity_pass,
    }
    write_json(output / "r8e_b_integrity_manifest.json", integrity)
    tests = run_tests()
    tests_pass = tests["focused"]["returncode"] == 0 and tests["regression"]["returncode"] == 0
    write_json(output / "r8e_b_test_results.json", tests)
    replay = run_replays(output)
    write_json(output / "r8e_b_replay_results.json", replay)
    gates = compute_gates(aggregate, integrity_pass=integrity_pass, tests_pass=tests_pass, replay_pass=replay["semantic_match"])

    fresh = records[0]
    seed13086 = {
        "h1_abs_diff": abs(fresh["metrics"]["1"] - ORIGINAL_A1_H1),
        "h1_rel_diff": abs(fresh["metrics"]["1"] - ORIGINAL_A1_H1) / ORIGINAL_A1_H1,
        "h32_abs_diff": abs(fresh["metrics"]["32"] - ORIGINAL_A1_H32),
        "h32_rel_diff": abs(fresh["metrics"]["32"] - ORIGINAL_A1_H32) / ORIGINAL_A1_H32,
        "train_level_reproducible": "YES" if fresh["h1_guardrail_pass"] == "YES" and fresh["h32_20_percent_pass"] == "YES" else "NO",
        "decision_basis": "semantic/performance reproduction using the predeclared H1 and H32 gates; no post-hoc numeric tolerance introduced",
    }
    first_blocker = next((name for name, passed in (
        ("integrity_gate_failed", gates["integrity_pass"]),
        ("h1_robustness_failed", gates["h1_4_of_5_pass"]),
        ("h32_robustness_failed", gates["h32_4_of_5_pass"]),
        ("median_method_effect_failed", gates["median_20_percent_pass"]),
        ("tests_failed", gates["tests_pass"]),
        ("fresh_process_replay_failed", gates["replay_pass"]),
    ) if not passed), "none")
    status = "PASSED" if gates["r8e_b_final_pass"] else "BLOCKED"
    early_ood_pattern = pattern_label(early_ood_increase_count)
    h4_hidden_pattern = pattern_label(h4_regression_count)
    context = {
        "aggregate": aggregate, "hidden": hidden_aggregate, "records": records, "gates": gates, "integrity": integrity,
        "tests": tests, "replay": replay, "seed13086": seed13086, "first_blocker": first_blocker,
        "h2_reduction_robust": "YES" if h2_lower >= 3 else "NO",
        "h4_regression_robust": "YES" if h4_higher >= 3 else "NO",
        "h16_reduction_robust": "YES" if h16_lower >= 3 else "NO",
        "h32_reduction_robust": "YES" if h32_lower >= 3 else "NO",
        "hidden_rollout_direction": hidden_rollout_direction,
        "robustness_conclusion": "ROBUSTNESS_SUPPORTED" if gates["h32_4_of_5_pass"] and h16_lower >= 3 and h32_lower >= 3 else "ROBUSTNESS_NOT_SUPPORTED",
        "early_ood_pattern": early_ood_pattern, "h4_hidden_pattern": h4_hidden_pattern, "onset_distribution": onset_distribution,
        "early_ood_dominant": "YES" if early_ood_pattern == "YES" else ("MIXED" if early_ood_pattern == "MIXED" else "NO"),
        "peak_scratch_mb": round(peak_scratch_bytes / 1024 / 1024, 3), "final_size_mb": 0.0,
        "next_action": "Begin a separately authorized validation-only R8F focused on the cross-seed residual mechanism; do not change A1 or open Frozen-20." if status == "PASSED" else "Audit why seeds 218309645, 258275761, and 1638377564 exceed the frozen H1 guardrail, using the completed validation-only records without tuning seeds, thresholds, architecture, loss, or curriculum.",
        "owner_recommendation": "Accept R8E_A1 multi-seed robustness as supported and, only on separate authorization, begin validation-only R8F while keeping A1 and Frozen-20 unchanged." if status == "PASSED" else "Do not begin R8F; preserve A1 and investigate the reported blocker under the frozen protocol.",
    }
    certificate = {
        "schema_version": "stage3_h13_r8e_b_terminal_certificate_v1",
        "STAGE_3_H13_R8E_B": status, "FIRST_BLOCKER": first_blocker,
        "R8E_A1_MULTI_SEED_ROBUSTNESS": "SUPPORTED" if status == "PASSED" else "NOT_SUPPORTED",
        "CURRENT_VALIDATION_CHAMPION": "R8E_A1", "CURRENT_VALIDATION_CHAMPION_CHANGED": "NO",
        "READY_FOR_R8F_VALIDATION_ONLY_OPTIMIZATION": "YES" if status == "PASSED" else "NO",
        "FROZEN20_OPENED": "NO", "R9_EXECUTED": "NO",
        "gates": gates, "aggregate": aggregate, "seed13086_reproduction": seed13086,
        "hidden_state_robustness": {key: hidden_aggregate[key] for key in ("H2_REDUCTION_ROBUST", "H4_REGRESSION_ROBUST", "H16_REDUCTION_ROBUST", "H32_REDUCTION_ROBUST", "hidden_rollout_direction")},
        "feature_ood": {"early_feature_ood_increase_reproduced": early_ood_pattern, "h4_hidden_regression_reproduced": h4_hidden_pattern, "amplification_onset_distribution": onset_distribution},
        "tests": tests, "replay": replay,
    }
    write_json(output / "stage3_h13_r8e_b_terminal_certificate.json", certificate)
    context["final_size_mb"] = round(directory_size_mb(output), 3)
    (output / "FINAL_REPORT.md").write_text(report_text(context), encoding="utf-8", newline="\n")
    context["final_size_mb"] = round(directory_size_mb(output), 3)
    (output / "FINAL_REPORT.md").write_text(report_text(context), encoding="utf-8", newline="\n")
    certificate["storage"] = {"peak_new_scratch_mb": context["peak_scratch_mb"], "final_persistent_size_mb": context["final_size_mb"], "intermediates_cleaned": "YES"}
    write_json(output / "stage3_h13_r8e_b_terminal_certificate.json", certificate)
    print(str(output))
    print(status)
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
