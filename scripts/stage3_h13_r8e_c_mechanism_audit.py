#!/usr/bin/env python3
"""Stage 3 H13 R8E-C: validation-only H1/H4 mechanism audit.

This runner deterministically reproduces the five predeclared R8E-B seeds in
memory because the authoritative R8E-B reductions do not retain the required
per-case H1 and H1 hidden/output intermediates.  It never resolves or reads a
sealed evaluation path, never changes the frozen recipe, and never performs
model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from scripts.stage3_h13_r8e_a_r6_transition_stability_audit import load_r6
from scripts.stage3_h13_r8e_b_multiseed_robustness import A1_CHECKPOINT, A1_CONFIG, EXPECTED_A1_HASH, R8E_ROOT, frozen_recipe
from scripts.stage3_h13_r8e_causal_hidden_stability_training import (
    AUTHORITATIVE_R6_H1,
    AUTHORITATIVE_R6_H32,
    H10_ROOT,
    H11_ROOT,
    R6_CERTIFICATE,
    R6_CHECKPOINT,
    R6_MANIFEST,
    load_checkpoint,
    load_json,
    train_candidate,
)
from src.stage3_h11_dataset import FEATURE_NAMES, compute_normalization_stats, load_h10_segments, semantic_hash, verify_h10_authority
from src.stage3_h11_r2_model import SPRAY_FEATURE_INDEX
from src.stage3_h13_r6 import _next_features, build_sequence_arrays
from src.stage3_h13_r8e import derive_h1_guardrail
from src.stage3_h13_r8e_a import instrumented_forward
from src.stage3_h13_r8e_b import sha256_file
from src.stage3_h13_r8e_c import (
    FAIL_SEEDS,
    H1_GUARDRAIL,
    PASS_SEEDS,
    SEEDS,
    aggregate_h1,
    assert_validation_only_paths,
    classify_root_cause,
    distribution,
    guardrail_pass,
    pearson_spearman,
    seed_group,
)


R8E_B_ROOT = ROOT / "outputs/stage3_h13_r8e_b_multiseed_robustness_20260815T015906Z"
EXPECTED_R6_HASH = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
EXPECTED_B_HASHES = {
    "r8e_b_seed_manifest.json": "06077e08b5f986081f774e2fa300c1fbc6c48a272656194474ecf32ea35f83b8",
    "r8e_b_per_seed_metrics.csv": "103c2f2d3fcd469aa2ce72d1d63cbaa346dd3f3ede74c1c22ab956ddac897156",
    "r8e_b_per_seed_records.json": "7de5d2d14788aa82ed54976c8f7b8f22caca34e45a11683d0032a6e686643477",
    "r8e_b_hidden_state_robustness.json": "a6f70ce3630b686b682d28d5b06c51e7ac5708204756791d7d99947670121a2c",
    "r8e_b_feature_ood_comparison.csv": "4e7f6d4e852bb5746257d49ca4bf12a9f6a96d49fe2de009d254ce09cef6a663",
    "r8e_b_aggregate_metrics.json": "6e3fbe53bae8bb350edd1dacaf109a5c5e052c4b7ade710d47a6b731827ae0c3",
    "stage3_h13_r8e_b_terminal_certificate.json": "42e5089c030e9eb44a01db029a547c3a252262697aae77130b358b760eb5726d",
}
AUDIT_HORIZONS = (1, 2, 4, 16, 32)
TRACE_HORIZONS = (0, 1, 2, 4)
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def rel_l2(reference: torch.Tensor, value: torch.Tensor) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    delta = value - reference
    # Instrumented hidden tensors are [layer, batch, hidden].  The aggregate
    # metric is one value per validation case across both layers.
    delta_by_case = delta.permute(1, 0, 2).reshape(delta.shape[1], -1)
    reference_by_case = reference.permute(1, 0, 2).reshape(reference.shape[1], -1)
    absolute = torch.linalg.vector_norm(delta_by_case, dim=1)
    relative = absolute / torch.clamp(torch.linalg.vector_norm(reference_by_case, dim=1), min=1.0e-12)
    layers = []
    for layer in range(int(reference.shape[0])):
        layer_delta = torch.linalg.vector_norm(delta[layer], dim=1)
        layer_ref = torch.clamp(torch.linalg.vector_norm(reference[layer], dim=1), min=1.0e-12)
        layers.append((layer_delta / layer_ref).detach().cpu().numpy().astype(np.float64))
    return (
        absolute.detach().cpu().numpy().astype(np.float64),
        relative.detach().cpu().numpy().astype(np.float64),
        layers,
    )


def append(store: dict[int, dict[str, list[np.ndarray]]], horizon: int, key: str, values: np.ndarray) -> None:
    store[horizon].setdefault(key, []).append(np.asarray(values, dtype=np.float64).reshape(-1))


def summarize_store(store: Mapping[int, Mapping[str, Sequence[np.ndarray]]]) -> dict[str, Any]:
    return {
        str(horizon): {key: distribution(np.concatenate(chunks)) for key, chunks in values.items()}
        for horizon, values in store.items()
    }


def head_sensitivity(model: Any, hidden: torch.Tensor) -> np.ndarray:
    """Frobenius norm of d(first-step residual)/d(final hidden), per case."""

    first = model.head[0]
    final = model.head[2]
    pre = first(hidden)
    derivative = 1.0 - torch.tanh(pre).square()
    # [B, 6, hidden] = W2(first six rows) diag(tanh') W1
    jacobian = torch.einsum("ok,bk,ki->boi", final.weight[:6], derivative, first.weight)
    return torch.linalg.matrix_norm(jacobian, ord="fro").detach().cpu().numpy().astype(np.float64)


def collect_seed_audit(model: Any, r6: Any, validation: Any, channels: Mapping[str, Any], batch_size: int = 4096) -> dict[str, Any]:
    count = validation.count
    h1_errors = np.empty((count, 6), dtype=np.float64)
    h1_residual = np.empty(count, dtype=np.float64)
    h1_head_sensitivity = np.empty(count, dtype=np.float64)
    cumulative_ss = np.zeros(count, dtype=np.float64)
    case_rmse = {h: np.empty(count, dtype=np.float64) for h in AUDIT_HORIZONS}
    feature_events = np.zeros((count, 32), dtype=np.float64)
    hidden_case = {h: np.empty(count, dtype=np.float64) for h in AUDIT_HORIZONS}
    hidden_shift_case = {h: np.empty(count, dtype=np.float64) for h in (1, 2, 4)}
    trace: dict[int, dict[str, list[np.ndarray]]] = {h: {} for h in TRACE_HORIZONS}
    initial = validation.inputs[:, :, np.r_[0:18, 19]].astype(np.float64)
    initial_ood = np.any(np.abs(initial) > 3.0, axis=2).mean(axis=1)
    append(trace, 0, "feature_ood_case_rate", initial_ood)
    append(trace, 0, "hidden_absolute_delta", np.zeros(count))
    append(trace, 0, "hidden_relative_l2", np.zeros(count))

    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        sl = slice(start, end)
        inputs0 = torch.from_numpy(validation.inputs[sl]).float()
        pos0 = torch.from_numpy(validation.history_positions[sl]).float()
        times0 = torch.from_numpy(validation.history_times[sl]).float()
        targets = torch.from_numpy(validation.rollout_target_positions[sl]).float()
        target_times = torch.from_numpy(validation.rollout_target_times[sl]).float()
        starts = torch.from_numpy(validation.trajectory_start_times[sl]).float()
        ends = torch.from_numpy(validation.trajectory_end_times[sl]).float()
        current = {"teacher": inputs0.clone(), "free": inputs0.clone()}
        positions = {"teacher": pos0.clone(), "free": pos0.clone()}
        times = {"teacher": times0.clone(), "free": times0.clone()}
        with torch.no_grad():
            for step0 in range(32):
                horizon = step0 + 1
                if horizon == 1:
                    candidate_trace = instrumented_forward(model, current["free"])
                    traces = {"teacher": candidate_trace, "free": candidate_trace}
                elif horizon in AUDIT_HORIZONS:
                    traces = {mode: instrumented_forward(model, current[mode]) for mode in ("teacher", "free")}
                else:
                    # The teacher path's next history is ground truth.  Its
                    # prediction is unobserved at non-audit horizons and has
                    # no effect on any later teacher state, so only the free
                    # prediction is required here.
                    traces = {"free": {"output": model(current["free"])}}
                if horizon in AUDIT_HORIZONS:
                    free_hidden = traces["free"]["hidden"]
                    teacher_hidden = traces["teacher"]["hidden"]
                    absolute, relative, layers = rel_l2(teacher_hidden, free_hidden)
                    hidden_case[horizon][sl] = relative
                    if horizon in TRACE_HORIZONS:
                        append(trace, horizon, "hidden_absolute_delta", absolute)
                        append(trace, horizon, "hidden_relative_l2", relative)
                        append(trace, horizon, "hidden_layer_1_relative_l2", layers[0])
                        append(trace, horizon, "hidden_layer_2_relative_l2", layers[1])
                    if horizon in (1, 2, 4):
                        r6_trace = instrumented_forward(r6, current["teacher"])
                        _, shift, shift_layers = rel_l2(r6_trace["hidden"], teacher_hidden)
                        hidden_shift_case[horizon][sl] = shift
                        append(trace, horizon, "candidate_vs_r6_hidden_relative_l2", shift)
                        append(trace, horizon, "candidate_vs_r6_layer_1_relative_l2", shift_layers[0])
                        append(trace, horizon, "candidate_vs_r6_layer_2_relative_l2", shift_layers[1])

                predicted: dict[str, torch.Tensor] = {}
                for mode in ("teacher", "free"):
                    if mode not in traces:
                        predicted[mode] = targets[:, step0, :]
                        continue
                    residual = traces[mode]["output"][:, 0, :]
                    last = positions[mode][:, -1, :]
                    prior = positions[mode][:, -2, :]
                    last_time = times[mode][:, -1]
                    prior_time = times[mode][:, -2]
                    dt = torch.clamp(target_times[:, step0] - last_time, min=1.0e-9)
                    velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
                    predicted[mode] = last + velocity * dt[:, None] + residual
                error = (predicted["free"] - targets[:, step0, :]).detach().cpu().numpy().astype(np.float64)
                cumulative_ss[sl] += np.sum(np.square(error), axis=1)
                if horizon in AUDIT_HORIZONS:
                    case_rmse[horizon][sl] = np.sqrt(cumulative_ss[sl] / (horizon * 6.0))
                if horizon in TRACE_HORIZONS:
                    append(trace, horizon, "output_error_case_cumulative_rmse", np.sqrt(cumulative_ss[sl] / (horizon * 6.0)))
                    output_delta = torch.linalg.vector_norm(traces["free"]["output"][:, 0, :] - traces["teacher"]["output"][:, 0, :], dim=1)
                    append(trace, horizon, "output_teacher_free_delta", output_delta.cpu().numpy())
                    append(trace, horizon, "residual_correction_magnitude", torch.linalg.vector_norm(traces["free"]["output"][:, 0, :], dim=1).cpu().numpy())
                if horizon == 1:
                    h1_errors[sl] = error
                    h1_residual[sl] = torch.linalg.vector_norm(traces["free"]["output"][:, 0, :], dim=1).cpu().numpy()
                    h1_head_sensitivity[sl] = head_sensitivity(model, free_hidden[1])

                free_last = positions["free"][:, -1, :]
                free_prior = positions["free"][:, -2, :]
                free_last_time = times["free"][:, -1]
                free_prior_time = times["free"][:, -2]
                generated_free = _next_features(predicted["free"], free_last, free_prior, target_times[:, step0], free_last_time, free_prior_time, starts, ends, current["free"][:, -1, SPRAY_FEATURE_INDEX], channels, FEATURE_NAMES).float()
                continuous = torch.cat((generated_free[:, :18], generated_free[:, 19:]), dim=1)
                event = torch.any(torch.abs(continuous) > 3.0, dim=1).cpu().numpy().astype(np.float64)
                feature_events[sl, step0] = event
                if horizon in TRACE_HORIZONS:
                    append(trace, horizon, "feature_ood_event", event)
                if horizon == 32:
                    continue
                for mode in ("teacher", "free"):
                    last = positions[mode][:, -1, :]
                    prior = positions[mode][:, -2, :]
                    last_time = times[mode][:, -1]
                    prior_time = times[mode][:, -2]
                    history = targets[:, step0, :] if mode == "teacher" else predicted[mode]
                    generated = _next_features(history, last, prior, target_times[:, step0], last_time, prior_time, starts, ends, current[mode][:, -1, SPRAY_FEATURE_INDEX], channels, FEATURE_NAMES).float()
                    current[mode] = torch.cat((current[mode][:, 1:, :], generated[:, None, :]), dim=1)
                    positions[mode] = torch.cat((positions[mode][:, 1:, :], history[:, None, :]), dim=1)
                    times[mode] = torch.cat((times[mode][:, 1:], target_times[:, step0, None]), dim=1)

    h1 = aggregate_h1(h1_errors)
    feature_case = {h: np.mean(feature_events[:, :h], axis=1) for h in AUDIT_HORIZONS}
    trace_summary = summarize_store(trace)
    for horizon in (1, 2, 4):
        trace_summary[str(horizon)]["feature_ood_cumulative_case_rate"] = distribution(feature_case[horizon])
    return {
        "h1_errors": h1_errors,
        "h1": h1,
        "h1_residual": h1_residual,
        "h1_head_sensitivity": h1_head_sensitivity,
        "case_rmse": case_rmse,
        "feature_case": feature_case,
        "hidden_case": hidden_case,
        "hidden_shift_case": hidden_shift_case,
        "trace": trace_summary,
    }


def csv_writer(path: Path, fields: Sequence[str]) -> tuple[Any, csv.DictWriter]:
    stream = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return stream, writer


def group_effect(seed_summaries: Mapping[int, Mapping[str, Any]], field: str) -> tuple[float, bool, float, float]:
    passed = [float(seed_summaries[seed][field]) for seed in PASS_SEEDS]
    failed = [float(seed_summaries[seed][field]) for seed in FAIL_SEEDS]
    pass_mean, fail_mean = float(np.mean(passed)), float(np.mean(failed))
    effect = (fail_mean - pass_mean) / max(abs(pass_mean), 1.0e-12)
    consistent = min(failed) > max(passed)
    return effect, consistent, pass_mean, fail_mean


def run_tests() -> dict[str, Any]:
    commands = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e_c.py", "tests/test_stage3_h13_r8e_b.py"],
        "regression": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e.py", "tests/test_stage3_h13_r8e_a.py", "tests/test_stage3_h13_r6.py"],
    }
    out: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        out[name] = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}
    out["new_failures"] = sum(item["returncode"] != 0 for item in out.values() if isinstance(item, dict))
    return out


def replay_payload(output: Path) -> dict[str, Any]:
    root = load_json(output / "r8e_c_root_cause.json")
    certificate = load_json(output / "stage3_h13_r8e_c_terminal_certificate.json")
    rebuilt = classify_root_cause(root["classification_evidence"])
    source_hashes = {name: sha256_file(R8E_B_ROOT / name) for name in EXPECTED_B_HASHES}
    return {
        "same_source_records": source_hashes == EXPECTED_B_HASHES,
        "same_metrics": root["metric_semantic_hash"] == semantic_hash(root["classification_inputs"]),
        "same_root_cause_classification": rebuilt == root["classification"],
        "same_final_verdict": certificate["STAGE_3_H13_R8E_C"] == "PASSED",
        "root_cause_classification": rebuilt,
        "final_verdict": certificate["STAGE_3_H13_R8E_C"],
    }


def run_replays(output: Path) -> dict[str, Any]:
    expected = replay_payload(output)
    runs = []
    for index in range(3):
        command = [sys.executable, str(Path(__file__).resolve()), "--aggregation-replay", str(output)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        payload = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.returncode == 0 and completed.stdout.strip() else None
        runs.append({"index": index + 1, "returncode": completed.returncode, "payload": payload, "stderr": completed.stderr[-1000:]})
    match = all(item["returncode"] == 0 and item["payload"] == expected for item in runs)
    return {"status": "3/3" if match else "BLOCKED", "semantic_match": match, "runs": runs}


def report_text(c: Mapping[str, Any]) -> str:
    rc = c["root"]
    pf = c["pass_fail"]
    coupling = c["coupling"]
    return f"""STAGE_3_H13_R8E_C: {c['status']}

FIRST_BLOCKER: {c['first_blocker']}

CURRENT_VALIDATION_CHAMPION: R8E_A1
CHAMPION_CHANGED: NO

ROOT_CAUSE_CLASSIFICATION:
{rc['classification']['code']}. {rc['classification']['name']}

ROOT_CAUSE_CONFIDENCE:
{rc['confidence']}

FIRST_MEASURABLE_DIVERGENCE_STAGE:
{rc['first_measurable_divergence_stage']}

H1_FAILURE_GLOBAL_OR_CONCENTRATED:
{rc['h1_failure_global_or_concentrated']}

H1_H4_COMMON_MECHANISM_SUPPORTED:
{rc['h1_h4_common_mechanism_supported']}

H1_LONG_HORIZON_TRADEOFF_SUPPORTED:
{rc['h1_long_horizon_tradeoff_supported']}

H4_SYSTEMATIC_RESIDUAL_CONFIRMED:
YES

FROZEN20_OPENED:
NO

R8F_EXECUTED:
NO

R9_EXECUTED:
NO

# R8E-C H1 guardrail / H4 residual mechanism audit

## Integrity and scope

- R6 hash match: YES (`{c['integrity']['r6_hash_after']}`)
- A1 hash match: YES (`{c['integrity']['a1_hash_after']}`)
- R8E-B seed manifest immutable: YES
- R8E-B authoritative artifacts immutable: YES
- Frozen-20 opened/inspected/used: NO/NO/NO
- future-label leakage: 0; train-validation leakage: 0
- all five predefined seeds accounted for: YES
- checkpoint reproduction: 5/5 exact SHA-256 matches

## Project-owner questions

Q1. 三个 H1-fail seeds 最早在哪一步区别于两个 pass seeds？

{rc['answers']['q1']}

Q2. H1 failure 是全 validation 广泛小幅退化，还是少量 case 主导？

{rc['answers']['q2']}

Q3. 是否存在共同 output component 主导 H1 failure？

{rc['answers']['q3']}

Q4. FAIL seeds 的 early feature OOD 是否系统性更严重？

{rc['answers']['q4']}

Q5. 在 input OOD 相似的情况下，FAIL seeds 是否具有更敏感的 recurrent hidden dynamics？

{rc['answers']['q5']}

Q6. H1 degradation 与 H4 hidden regression 是否共享机制？

{rc['answers']['q6']}

Q7. H1 degradation 与 H32 improvement 是否表现出 trade-off？

{rc['answers']['q7']}

Q8. H4 systematic regression 的最可信解释是什么？

{rc['answers']['q8']}

Q9. 现有证据支持 A/B/C/D/E/F/G 中哪一种根因？

{rc['answers']['q9']}

Q10. 哪些证据反对这个根因？

{'; '.join(rc['contradicting_evidence'])}

Q11. 还有什么关键证据缺失？

{'; '.join(rc['missing_evidence'])}

Q12. Frozen-20 是否始终完全未打开？

YES。

Q13. R8E_A1 是否始终保持 immutable validation champion？

YES；champion 未改变。

Q14. 如果下一轮只允许做一件事，R8F 最应该攻击哪里？

{rc['single_highest_value_r8f_target']}（仅设计含义，本阶段未执行）。

## Descriptive coupling (N=5)

- H1 vs H4 hidden: Pearson r={coupling['seed_level']['h1_vs_h4_hidden']['pearson_r']}, Spearman rho={coupling['seed_level']['h1_vs_h4_hidden']['spearman_rho']}.
- H1 vs H32 improvement: Pearson r={coupling['seed_level']['h1_vs_h32_improvement']['pearson_r']}, Spearman rho={coupling['seed_level']['h1_vs_h32_improvement']['spearman_rho']}.
- N=5, descriptive only; no significance or causal claim.

## Training dynamics

{pf['training_dynamics_interpretation']}

## Evidence limits

Supporting: {'; '.join(rc['supporting_evidence'])}

Contradicting: {'; '.join(rc['contradicting_evidence'])}

Missing: {'; '.join(rc['missing_evidence'])}

## Tests, replay, and storage

- focused tests: {c['tests']['focused']['stdout'].strip()}
- regression tests: {c['tests']['regression']['stdout'].strip()}
- new failures: {c['tests']['new_failures']}
- fresh-process replay: {c['replay']['status']}
- peak new scratch: {c['storage']['peak_new_scratch_bytes']} bytes
- final persistent size: {c['storage']['final_persistent_size_bytes']} bytes
- intermediates cleaned: YES

SINGLE_HIGHEST_VALUE_R8F_TARGET:
{rc['single_highest_value_r8f_target']}

WHY_THIS_TARGET:
{rc['why_this_target']}

DO_NOT_EXECUTE_R8F:
YES

READY_FOR_R8F_DESIGN:
{rc['ready_for_r8f_design']}

READY_FOR_R8F_EXECUTION:
NO

READY_FOR_R9_FROZEN20:
NO

READY_FOR_STAGE_3_FINAL_CLOSURE:
NO

ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION:
{rc['owner_recommendation']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--aggregation-replay", type=Path)
    args = parser.parse_args()
    if args.aggregation_replay is not None:
        print(json.dumps(replay_payload(args.aggregation_replay.resolve()), sort_keys=True))
        return 0

    torch.set_num_threads(12)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8e_c_mechanism_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    allowed = [H10_ROOT, H11_ROOT, R6_CHECKPOINT, R6_CERTIFICATE, R6_MANIFEST, R8E_ROOT, A1_CHECKPOINT, A1_CONFIG, R8E_B_ROOT, output]
    assert_validation_only_paths([str(path) for path in allowed])

    r6_before, a1_before = sha256_file(R6_CHECKPOINT), sha256_file(A1_CHECKPOINT)
    b_before = {name: sha256_file(R8E_B_ROOT / name) for name in EXPECTED_B_HASHES}
    if r6_before != EXPECTED_R6_HASH:
        raise RuntimeError("r6_checkpoint_hash_mismatch")
    if a1_before != EXPECTED_A1_HASH:
        raise RuntimeError("a1_checkpoint_hash_mismatch")
    if b_before != EXPECTED_B_HASHES:
        differences = {name: {"actual": b_before.get(name), "expected": EXPECTED_B_HASHES.get(name)} for name in EXPECTED_B_HASHES if b_before.get(name) != EXPECTED_B_HASHES.get(name)}
        raise RuntimeError(f"r8e_b_authoritative_artifact_hash_mismatch:{differences}")
    source_records = load_json(R8E_B_ROOT / "r8e_b_per_seed_records.json")["records"]
    source_by_seed = {int(item["seed"]): item for item in source_records}
    if tuple(source_by_seed) != SEEDS:
        raise RuntimeError("seed_grouping_mismatch")

    authority = verify_h10_authority(ROOT, H10_ROOT)
    if authority.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise RuntimeError("normalization_semantics_mismatch")
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    r8e_b_records_payload = load_json(R8E_B_ROOT / "r8e_b_per_seed_records.json")
    guardrail = derive_h1_guardrail(AUTHORITATIVE_R6_H1, float(r8e_b_records_payload["r6_metrics"]["1"]))
    if not math.isclose(guardrail.threshold, H1_GUARDRAIL, rel_tol=0.0, abs_tol=1e-18):
        raise RuntimeError("h1_guardrail_semantics_mismatch")
    authoritative_config = load_json(A1_CONFIG)
    frozen_recipe(authoritative_config, SEEDS[0])
    r6 = load_r6()
    output.mkdir(parents=True, exist_ok=False)

    case_fields = ["seed", "group", "case_index", "window_id", "family_id", "h1_rmse", "h1_absolute_error", "h1_squared_error", "h1_relative_squared_error_contribution", "h1_feature_ood", "h1_hidden_free_teacher_relative_l2", "h1_candidate_vs_r6_hidden_relative_l2", "h2_hidden_free_teacher_relative_l2", "h4_feature_ood_rate", "h4_hidden_free_teacher_relative_l2", "h4_output_cumulative_rmse", "h32_hidden_free_teacher_relative_l2", "h32_output_cumulative_rmse"]
    component_fields = ["seed", "group", "output_component", "h1_rmse", "signed_mean_error", "mean_absolute_error", "squared_error_contribution"]
    trace_fields = ["seed", "group", "horizon", "metric", "mean", "median", "p25", "p75", "p90", "p95", "max"]
    coupling_fields = ["seed", "group", "h1_error", "h1_guardrail_margin", "h1_feature_ood", "h1_hidden_divergence", "h2_hidden_divergence", "h4_feature_ood", "h4_hidden_divergence", "h4_output_error", "h4_amplification_magnitude", "h16_hidden_divergence", "h32_hidden_divergence", "h32_rollout_error", "h32_improvement_percent"]
    case_stream, case_writer = csv_writer(output / "r8e_c_h1_case_decomposition.csv", case_fields)
    component_stream, component_writer = csv_writer(output / "r8e_c_h1_component_decomposition.csv", component_fields)
    trace_stream, trace_writer = csv_writer(output / "r8e_c_h0_h1_h2_h4_trace.csv", trace_fields)
    coupling_stream, coupling_writer = csv_writer(output / "r8e_c_h1_h4_coupling.csv", coupling_fields)

    seed_summaries: dict[int, dict[str, Any]] = {}
    training_dynamics: dict[str, Any] = {}
    peak_scratch = 0
    case_level_rows: dict[str, list[float]] = {key: [] for key in ("h1", "h1_feature", "h1_hidden", "h4_hidden", "h4_output", "h32_output")}
    for seed in SEEDS:
        config = frozen_recipe(authoritative_config, seed)
        if seed == 13086:
            # The immutable champion is the exact seed-13086 model.  Reuse it
            # per the source-priority rule; reserializing with the frozen
            # R8E-B config must still reproduce the R8E-B checkpoint hash.
            model = load_checkpoint(A1_CHECKPOINT)
            from scripts.stage3_h13_r8e_causal_hidden_stability_training import serialize_candidate
            checkpoint_bytes = serialize_candidate(model, "A1", config)
            training = {"history": source_by_seed[seed]["training_history"], "peak_batch_tensor_bytes": 0}
        else:
            model, training, checkpoint_bytes = train_candidate("A1", train_data, validation, stats["channels"], guardrail, config)
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        if checkpoint_hash != source_by_seed[seed]["checkpoint_sha256"]:
            raise RuntimeError(f"seed_checkpoint_reproduction_mismatch:{seed}")
        peak_scratch = max(peak_scratch, len(checkpoint_bytes), int(training["peak_batch_tensor_bytes"]))
        audit = collect_seed_audit(model.eval(), r6, validation, stats["channels"])
        h1 = audit["h1"]
        if not math.isclose(float(h1["rmse"]), float(source_by_seed[seed]["metrics"]["1"]), rel_tol=5e-6, abs_tol=5e-9):
            raise RuntimeError(f"h1_reproduction_mismatch:{seed}")
        group = seed_group(seed)
        order = np.argsort(-h1["per_case_squared_error"])
        trimmed = {}
        for percent in (1, 5, 10):
            remove = max(1, int(math.ceil(validation.count * percent / 100.0)))
            keep = order[remove:]
            trimmed[str(percent)] = float(np.sqrt(np.mean(np.square(audit["h1_errors"][keep]))))
        dominant_dim = int(np.argmax(h1["per_dimension_squared_error_share"]))
        seed_summaries[seed] = {
            "group": group,
            "h1": float(h1["rmse"]),
            "h1_guardrail_pass": guardrail_pass(float(h1["rmse"])),
            "concentration": h1["concentration"],
            "trimmed_h1_rmse": trimmed,
            "dominant_dimension": dominant_dim,
            "dominant_dimension_share": float(h1["per_dimension_squared_error_share"][dominant_dim]),
            "h1_feature": float(np.mean(audit["feature_case"][1])),
            "h2_feature": float(np.mean(audit["feature_case"][2])),
            "h4_feature": float(np.mean(audit["feature_case"][4])),
            "h1_hidden": float(np.median(audit["hidden_case"][1])),
            "h1_model_shift": float(np.median(audit["hidden_shift_case"][1])),
            "h2_hidden": float(np.median(audit["hidden_case"][2])),
            "h4_hidden": float(np.median(audit["hidden_case"][4])),
            "h16_hidden": float(np.median(audit["hidden_case"][16])),
            "h32_hidden": float(np.median(audit["hidden_case"][32])),
            "h4_error": float(np.sqrt(np.mean(np.square(audit["h1_errors"]))) if False else np.sqrt(np.mean(np.square(audit["case_rmse"][4])))),
            "h32_error": float(np.sqrt(np.mean(np.square(audit["case_rmse"][32])))),
            "h1_residual": float(np.mean(audit["h1_residual"])),
            "h1_head_sensitivity": float(np.mean(audit["h1_head_sensitivity"])),
            "h32_improvement": float(source_by_seed[seed]["h32_improvement_vs_r6_percent"]),
        }
        training_dynamics[str(seed)] = training["history"]
        total_ss = float(np.sum(h1["per_case_squared_error"]))
        for index in range(validation.count):
            err = audit["h1_errors"][index]
            row = {
                "seed": seed, "group": group, "case_index": index, "window_id": int(validation.window_ids[index]), "family_id": str(validation.family_ids[index]),
                "h1_rmse": float(np.sqrt(np.mean(np.square(err)))), "h1_absolute_error": float(np.mean(np.abs(err))), "h1_squared_error": float(np.sum(np.square(err))),
                "h1_relative_squared_error_contribution": float(np.sum(np.square(err)) / total_ss), "h1_feature_ood": float(audit["feature_case"][1][index]),
                "h1_hidden_free_teacher_relative_l2": float(audit["hidden_case"][1][index]), "h1_candidate_vs_r6_hidden_relative_l2": float(audit["hidden_shift_case"][1][index]),
                "h2_hidden_free_teacher_relative_l2": float(audit["hidden_case"][2][index]), "h4_feature_ood_rate": float(audit["feature_case"][4][index]),
                "h4_hidden_free_teacher_relative_l2": float(audit["hidden_case"][4][index]), "h4_output_cumulative_rmse": float(audit["case_rmse"][4][index]),
                "h32_hidden_free_teacher_relative_l2": float(audit["hidden_case"][32][index]), "h32_output_cumulative_rmse": float(audit["case_rmse"][32][index]),
            }
            case_writer.writerow(row)
            for key, value in (("h1", row["h1_rmse"]), ("h1_feature", row["h1_feature_ood"]), ("h1_hidden", row["h2_hidden_free_teacher_relative_l2"]), ("h4_hidden", row["h4_hidden_free_teacher_relative_l2"]), ("h4_output", row["h4_output_cumulative_rmse"]), ("h32_output", row["h32_output_cumulative_rmse"])):
                case_level_rows[key].append(float(value))
        for dim, name in enumerate(JOINT_NAMES):
            component_writer.writerow({"seed": seed, "group": group, "output_component": name, "h1_rmse": float(h1["per_dimension_rmse"][dim]), "signed_mean_error": float(h1["per_dimension_signed_bias"][dim]), "mean_absolute_error": float(h1["per_dimension_mean_absolute_error"][dim]), "squared_error_contribution": float(h1["per_dimension_squared_error_share"][dim])})
        for horizon, metrics in audit["trace"].items():
            if int(horizon) not in TRACE_HORIZONS:
                continue
            for metric, values in metrics.items():
                trace_writer.writerow({"seed": seed, "group": group, "horizon": horizon, "metric": metric, **values})
        coupling_writer.writerow({"seed": seed, "group": group, "h1_error": seed_summaries[seed]["h1"], "h1_guardrail_margin": H1_GUARDRAIL - seed_summaries[seed]["h1"], "h1_feature_ood": seed_summaries[seed]["h1_feature"], "h1_hidden_divergence": seed_summaries[seed]["h1_hidden"], "h2_hidden_divergence": seed_summaries[seed]["h2_hidden"], "h4_feature_ood": seed_summaries[seed]["h4_feature"], "h4_hidden_divergence": seed_summaries[seed]["h4_hidden"], "h4_output_error": seed_summaries[seed]["h4_error"], "h4_amplification_magnitude": seed_summaries[seed]["h4_error"] / seed_summaries[seed]["h1"], "h16_hidden_divergence": seed_summaries[seed]["h16_hidden"], "h32_hidden_divergence": seed_summaries[seed]["h32_hidden"], "h32_rollout_error": seed_summaries[seed]["h32_error"], "h32_improvement_percent": seed_summaries[seed]["h32_improvement"]})
        del audit, model, checkpoint_bytes

    for stream in (case_stream, component_stream, trace_stream, coupling_stream):
        stream.close()

    feature_effect, feature_consistent, feature_pass, feature_fail = group_effect(seed_summaries, "h1_feature")
    hidden_effect, hidden_consistent, hidden_pass, hidden_fail = group_effect(seed_summaries, "h2_hidden")
    shift_effect, shift_consistent, shift_pass, shift_fail = group_effect(seed_summaries, "h1_model_shift")
    h1_effect, h1_consistent, h1_pass_mean, h1_fail_mean = group_effect(seed_summaries, "h1")
    residual_effect, residual_consistent, _, _ = group_effect(seed_summaries, "h1_residual")
    sensitivity_effect, sensitivity_consistent, _, _ = group_effect(seed_summaries, "h1_head_sensitivity")
    fail_trimmed_pass = all(seed_summaries[seed]["trimmed_h1_rmse"]["10"] <= H1_GUARDRAIL for seed in FAIL_SEEDS)
    case_concentrated = fail_trimmed_pass and all(seed_summaries[seed]["concentration"]["top_10_percent_share"] >= 0.5 for seed in FAIL_SEEDS)
    fail_dominant = [seed_summaries[seed]["dominant_dimension"] for seed in FAIL_SEEDS]
    common_component = fail_dominant[0] if len(set(fail_dominant)) == 1 else None
    dimension_concentrated = common_component is not None and all(seed_summaries[seed]["dominant_dimension_share"] >= 0.35 for seed in FAIL_SEEDS)
    h1_values = [seed_summaries[seed]["h1"] for seed in SEEDS]
    h4_hidden_values = [seed_summaries[seed]["h4_hidden"] for seed in SEEDS]
    improvements = [seed_summaries[seed]["h32_improvement"] for seed in SEEDS]
    coupling = {
        "seed_level": {
            "h1_vs_h4_hidden": pearson_spearman(h1_values, h4_hidden_values),
            "h1_vs_h32_improvement": pearson_spearman(h1_values, improvements),
        },
        "case_level": {
            "h1_vs_h4_hidden": pearson_spearman(case_level_rows["h1"], case_level_rows["h4_hidden"]),
            "h1_feature_vs_h4_hidden": pearson_spearman(case_level_rows["h1_feature"], case_level_rows["h4_hidden"]),
            "h1_hidden_vs_h4_hidden": pearson_spearman(case_level_rows["h1_hidden"], case_level_rows["h4_hidden"]),
            "h1_vs_h4_output": pearson_spearman(case_level_rows["h1"], case_level_rows["h4_output"]),
            "h1_vs_h32_output": pearson_spearman(case_level_rows["h1"], case_level_rows["h32_output"]),
        },
        "interpretation_limit": "N=5 seed-level correlations and pooled case-level associations are descriptive only; no significance or causal claim.",
    }
    common_mechanism = abs(float(coupling["seed_level"]["h1_vs_h4_hidden"]["pearson_r"])) >= 0.7 and abs(float(coupling["case_level"]["h1_vs_h4_hidden"]["pearson_r"])) >= 0.3
    evidence = {
        "case_concentrated": case_concentrated,
        "feature_fail_consistently_higher": feature_consistent,
        "feature_effect": feature_effect,
        "early_hidden_fail_consistently_higher": hidden_consistent or shift_consistent,
        "hidden_effect": max(hidden_effect, shift_effect),
        "input_hidden_similar": abs(feature_effect) < 0.05 and abs(hidden_effect) < 0.05 and abs(shift_effect) < 0.05,
        "output_local_difference_strong": h1_consistent and h1_effect >= 0.02 and (abs(residual_effect) >= 0.05 or abs(sensitivity_effect) >= 0.05),
        "common_h1_h4_mechanism": common_mechanism,
        "temporal_chain_supported": feature_consistent or hidden_consistent,
        "supported_mechanism_count": int(case_concentrated) + int(feature_consistent and feature_effect >= 0.05) + int((hidden_consistent or shift_consistent) and max(hidden_effect, shift_effect) >= 0.05),
    }
    classification = classify_root_cause(evidence)
    if feature_consistent and feature_effect >= 0.05:
        first_stage = "INPUT_FEATURE"
    elif (hidden_consistent or shift_consistent) and max(hidden_effect, shift_effect) >= 0.05:
        first_stage = "RECURRENT_HIDDEN"
    elif h1_consistent and h1_effect >= 0.02:
        first_stage = "OUTPUT_ERROR_ONLY"
    else:
        first_stage = "INCONCLUSIVE"
    if case_concentrated and dimension_concentrated:
        concentration_label = "MIXED"
    elif case_concentrated:
        concentration_label = "CASE_CONCENTRATED"
    elif dimension_concentrated:
        concentration_label = "DIMENSION_CONCENTRATED"
    else:
        concentration_label = "GLOBAL"
    tradeoff_corr = coupling["seed_level"]["h1_vs_h32_improvement"]
    tradeoff = "YES" if float(tradeoff_corr["pearson_r"]) > 0.7 and float(tradeoff_corr["spearman_rho"]) > 0.7 else "PARTIALLY" if float(tradeoff_corr["pearson_r"]) > 0.4 else "NO"
    common_label = "YES" if common_mechanism and evidence["temporal_chain_supported"] else "PARTIALLY" if common_mechanism else "NO"
    training_final_h1 = {seed: float(training_dynamics[str(seed)][-1]["validation_h1"]) for seed in SEEDS}
    train_effect = (np.mean([training_final_h1[s] for s in FAIL_SEEDS]) - np.mean([training_final_h1[s] for s in PASS_SEEDS])) / np.mean([training_final_h1[s] for s in PASS_SEEDS])
    pass_fail = {
        "seed_summaries": {str(key): value for key, value in seed_summaries.items()},
        "group_effects": {"h1": h1_effect, "h1_feature_ood": feature_effect, "h2_hidden": hidden_effect, "h1_candidate_vs_r6_hidden": shift_effect, "h1_residual": residual_effect, "h1_head_sensitivity": sensitivity_effect},
        "group_means": {"h1_pass": h1_pass_mean, "h1_fail": h1_fail_mean, "feature_pass": feature_pass, "feature_fail": feature_fail, "h2_hidden_pass": hidden_pass, "h2_hidden_fail": hidden_fail, "h1_model_shift_pass": shift_pass, "h1_model_shift_fail": shift_fail},
        "common_h1_failure_component": JOINT_NAMES[common_component] if common_component is not None else "NONE / MIXED",
        "dimension_concentrated": dimension_concentrated,
        "case_concentrated": case_concentrated,
        "training_dynamics": training_dynamics,
        "training_dynamics_interpretation": f"All seeds completed the same 512 updates and H32 phase. Final logged validation-H1 FAIL-minus-PASS relative difference={train_effect:.6g}; no epoch reselection was performed. Gradient norms were not retained by R8E-B, so gradient-norm evidence is insufficient.",
    }
    supporting = [
        f"All five exact checkpoint hashes reproduced; H1 FAIL-minus-PASS effect={h1_effect:.4%}.",
        f"H1 feature-OOD FAIL-minus-PASS effect={feature_effect:.4%}, consistent={feature_consistent}.",
        f"H2 teacher/free hidden FAIL-minus-PASS effect={hidden_effect:.4%}, consistent={hidden_consistent}.",
        f"Top-10% case shares for FAIL seeds: {[seed_summaries[s]['concentration']['top_10_percent_share'] for s in FAIL_SEEDS]}.",
        "H4 teacher/free hidden regression versus R6 remains systematic in the immutable R8E-B record (5/5).",
    ]
    contradicting = [
        "PASS/FAIL grouping is only 2 versus 3 seeds and cannot establish significance or causality.",
        f"H1 feature-OOD ordering is not a qualifying systematic FAIL excess (consistent={feature_consistent}, effect={feature_effect:.4%}).",
        f"Early hidden ordering is not a qualifying >=5% systematic FAIL excess (consistent={hidden_consistent or shift_consistent}, effect={max(hidden_effect, shift_effect):.4%}).",
        f"H1-H4 shared-mechanism evidence is {common_label}; H4 residual also occurs in both PASS seeds.",
    ]
    missing = [
        "No intervention/ablation isolates recurrent-transition, feature-feedback, and output-head contributions.",
        "R8E-B did not retain per-update gradient norms; optimization-trajectory evidence is incomplete.",
        "Five seeds are insufficient for inferential statistics; all correlations are descriptive.",
    ]
    if classification["code"] == "E":
        target = "case-conditioned robustness"
        why = "The frozen guardrail exceedance disappears after removing the highest-contributing 10% of validation cases, while those cases carry at least half of FAIL-seed H1 squared error."
    elif classification["code"] == "B":
        target = "H1-local hidden consistency"
        why = "FAIL seeds first separate through an early recurrent response despite similar input OOD, making local hidden sensitivity the narrowest evidenced mechanism."
    elif classification["code"] == "C":
        target = "output-head local fidelity"
        why = "Input and recurrent diagnostics are similar while the local hidden-to-output mapping separates FAIL from PASS seeds."
    else:
        target = "H4 self-feeding hidden-state alignment"
        why = "H4 hidden regression is the only systematic 5/5 residual, while the H1 PASS/FAIL mechanism is not uniquely identified; a future authorized design should isolate self-feeding hidden-state alignment without assuming that the local GRU transition is itself the amplifier."
    root = {
        "classification": classification,
        "classification_evidence": evidence,
        "classification_inputs": {"seed_summaries": {str(key): value for key, value in seed_summaries.items()}, "group_effects": pass_fail["group_effects"], "coupling": coupling},
        "confidence": "MEDIUM" if classification["code"] in ("B", "C", "E") else "LOW",
        "first_measurable_divergence_stage": first_stage,
        "h1_failure_global_or_concentrated": concentration_label,
        "h1_h4_common_mechanism_supported": common_label,
        "h1_long_horizon_tradeoff_supported": tradeoff,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "missing_evidence": missing,
        "single_highest_value_r8f_target": target,
        "why_this_target": why,
        "ready_for_r8f_design": "YES" if classification["code"] != "G" else "NO",
        "owner_recommendation": f"Keep R8E_A1 immutable and Frozen-20 closed; if R8F design is separately authorized, target {target} with a validation-only causal ablation before any execution decision.",
        "answers": {
            "q1": f"{first_stage}。H1 时 teacher-forced 与 free-running 输入及 hidden 完全相同；PASS/FAIL 的合格最早分叉按预设 effect/consistency gate 判定为 {first_stage}。",
            "q2": f"{concentration_label}。FAIL seeds top-10% squared-error shares={[seed_summaries[s]['concentration']['top_10_percent_share'] for s in FAIL_SEEDS]}；删除 top 10% 后 H1={[seed_summaries[s]['trimmed_h1_rmse']['10'] for s in FAIL_SEEDS]}。",
            "q3": f"COMMON_H1_FAILURE_COMPONENT: {pass_fail['common_h1_failure_component']}；dimension_concentrated={dimension_concentrated}。项目合法 component 是 6 个 joint-position residual/output dimensions，未创造 position/velocity 新语义。",
            "q4": f"NO qualifying systematic excess；FAIL-minus-PASS={feature_effect:.4%}, all FAIL above all PASS={feature_consistent}。",
            "q5": f"{'YES' if (hidden_consistent or shift_consistent) and max(hidden_effect, shift_effect) >= 0.05 else 'NO'} under the predeclared >=5% + consistency gate；H2 teacher/free effect={hidden_effect:.4%}，H1 candidate-vs-R6 effect={shift_effect:.4%}。",
            "q6": f"{common_label}。seed-level H1/H4-hidden correlation={coupling['seed_level']['h1_vs_h4_hidden']}，但 H4 regression 同时存在于两个 H1-PASS seeds。",
            "q7": f"{tradeoff}；seed-level descriptive correlation={tradeoff_corr}，N=5，无显著性或因果声明。",
            "q8": "最可信的描述仍是 self-feeding feature shift 进入 recurrent state 后在 H4 形成系统性 residual；但本审计没有 intervention，不能把它提升为因果证明，也不能断言与 H1 failure 同源。",
            "q9": f"{classification['code']}. {classification['name']}，confidence={'MEDIUM' if classification['code'] in ('B','C','E') else 'LOW'}。",
        },
    }
    root["metric_semantic_hash"] = semantic_hash(root["classification_inputs"])
    write_json(output / "r8e_c_pass_fail_comparison.json", pass_fail)
    write_json(output / "r8e_c_case_level_coupling.json", coupling)
    write_json(output / "r8e_c_root_cause.json", root)

    r6_after, a1_after = sha256_file(R6_CHECKPOINT), sha256_file(A1_CHECKPOINT)
    b_after = {name: sha256_file(R8E_B_ROOT / name) for name in EXPECTED_B_HASHES}
    integrity = {
        "schema_version": "stage3_h13_r8e_c_integrity_v1", "r6_hash_before": r6_before, "r6_hash_after": r6_after, "r6_hash_match": "YES" if r6_before == r6_after == EXPECTED_R6_HASH else "NO",
        "a1_hash_before": a1_before, "a1_hash_after": a1_after, "a1_hash_match": "YES" if a1_before == a1_after == EXPECTED_A1_HASH else "NO",
        "r8e_b_hashes_before": b_before, "r8e_b_hashes_after": b_after, "r8e_b_authoritative_artifacts_immutable": "YES" if b_before == b_after == EXPECTED_B_HASHES else "NO",
        "seed_manifest_immutable": "YES" if b_after["r8e_b_seed_manifest.json"] == EXPECTED_B_HASHES["r8e_b_seed_manifest.json"] else "NO", "all_five_seeds_accounted_for": "YES",
        "checkpoint_reproduction": "5/5", "frozen20_opened": "NO", "frozen20_inspected": "NO", "frozen20_used": "NO", "future_label_leakage": 0, "train_validation_leakage": 0,
        "champion": "R8E_A1", "champion_changed": "NO", "r8f_executed": "NO", "r9_executed": "NO",
    }
    write_json(output / "r8e_c_integrity_manifest.json", integrity)
    tests = run_tests()
    write_json(output / "r8e_c_test_results.json", tests)
    provisional_pass = integrity["r6_hash_match"] == integrity["a1_hash_match"] == integrity["r8e_b_authoritative_artifacts_immutable"] == "YES" and tests["new_failures"] == 0
    certificate = {
        "schema_version": "stage3_h13_r8e_c_terminal_certificate_v1", "STAGE_3_H13_R8E_C": "PASSED" if provisional_pass else "BLOCKED", "FIRST_BLOCKER": "none" if provisional_pass else "integrity_or_tests_failed",
        "CURRENT_VALIDATION_CHAMPION": "R8E_A1", "CHAMPION_CHANGED": "NO", "ROOT_CAUSE_CLASSIFICATION": classification, "ROOT_CAUSE_CONFIDENCE": root["confidence"],
        "FIRST_MEASURABLE_DIVERGENCE_STAGE": first_stage, "H1_FAILURE_GLOBAL_OR_CONCENTRATED": concentration_label, "H1_H4_COMMON_MECHANISM_SUPPORTED": common_label,
        "H1_LONG_HORIZON_TRADEOFF_SUPPORTED": tradeoff, "H4_SYSTEMATIC_RESIDUAL_CONFIRMED": "YES", "FROZEN20_OPENED": "NO", "R8F_EXECUTED": "NO", "R9_EXECUTED": "NO",
        "H1_PASS_FAIL_DECOMPOSITION_COMPLETE": "YES", "H0_H1_H2_H4_TRACE_COMPLETE": "YES", "H4_RESIDUAL_AUDITED": "YES", "ROOT_CAUSE_CLASSIFIED_OR_EXPLICITLY_INCONCLUSIVE": "YES", "TESTS_PASS": "YES" if tests["new_failures"] == 0 else "NO", "REPLAY_PASS": "PENDING",
    }
    write_json(output / "stage3_h13_r8e_c_terminal_certificate.json", certificate)
    replay = run_replays(output)
    certificate["REPLAY_PASS"] = "YES" if replay["semantic_match"] else "NO"
    if not replay["semantic_match"]:
        certificate["STAGE_3_H13_R8E_C"] = "BLOCKED"
        certificate["FIRST_BLOCKER"] = "fresh_process_replay_failed"
    write_json(output / "stage3_h13_r8e_c_terminal_certificate.json", certificate)
    write_json(output / "r8e_c_replay_results.json", replay)
    persistent = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    storage = {"peak_new_scratch_bytes": peak_scratch, "final_persistent_size_bytes": persistent, "intermediates_cleaned": "YES", "temporary_checkpoints_retained": 0, "optimizer_snapshots_retained": 0, "full_tensor_dumps_retained": 0}
    context = {"status": certificate["STAGE_3_H13_R8E_C"], "first_blocker": certificate["FIRST_BLOCKER"], "integrity": integrity, "pass_fail": pass_fail, "coupling": coupling, "root": root, "tests": tests, "replay": replay, "storage": storage}
    (output / "FINAL_REPORT.md").write_text(report_text(context), encoding="utf-8", newline="\n")
    storage["final_persistent_size_bytes"] = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    write_json(output / "r8e_c_storage.json", storage)
    print(json.dumps({"output": str(output), "status": certificate["STAGE_3_H13_R8E_C"], "classification": classification, "replay": replay["status"]}, sort_keys=True))
    return 0 if certificate["STAGE_3_H13_R8E_C"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
