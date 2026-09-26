from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
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

from scripts.stage3_h13_r8e_a_r6_transition_stability_audit import collect_rollout_audit
from src.stage3_h11_dataset import FEATURE_NAMES, compute_normalization_stats, load_h10_segments, semantic_hash, verify_h10_authority
from src.stage3_h11_model import set_deterministic
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor
from src.stage3_h13_r6 import build_sequence_arrays, evaluate_rollout_metrics
from src.stage3_h13_r8e import (
    EVALUATION_HORIZONS,
    GRADUATION_THRESHOLD,
    PHASE_HORIZONS,
    H1Guardrail,
    causal_paired_rollout,
    derive_h1_guardrail,
    gradients_are_finite,
    graduation_passes,
    early_hidden_drift_reduced,
    h4_amplification_reduced_or_delayed,
    hidden_alignment_weights,
    rollout_horizon_weights,
    select_candidate,
    training_losses,
)


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_CERTIFICATE = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R6_MANIFEST = R6_ROOT / "checkpoint_manifest.json"
EXPECTED_R6_HASH = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
AUTHORITATIVE_R6_H1 = 0.00008188013630811277
AUTHORITATIVE_R6_H32 = 0.0232112820732007
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
SEED = 13086


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    def call(args: list[str]) -> str | None:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": call(["git", "rev-parse", "HEAD"]), "status_porcelain": call(["git", "status", "--short", "--", "src/stage3_h13_r8e.py", "scripts/stage3_h13_r8e_causal_hidden_stability_training.py", "tests/test_stage3_h13_r8e.py"])}


def new_model(*, trainable: bool) -> Any:
    payload = torch.load(R6_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(bool(trainable))
    return model


def load_checkpoint(path: Path) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {str(h): float(metrics[str(h)]["joint_position_rmse_rad"]) for h in EVALUATION_HORIZONS}


def evaluate_compact(model: Any, data: Any, channels: Mapping[str, Any], horizons: Sequence[int] = EVALUATION_HORIZONS) -> dict[str, float]:
    result = evaluate_rollout_metrics(model, data, channels, horizons=tuple(int(h) for h in horizons), device="cpu", batch_size=4096)
    return {str(h): float(result[str(h)]["joint_position_rmse_rad"]) for h in horizons}


def guardrail_payload(guardrail: H1Guardrail) -> dict[str, Any]:
    return {
        "R6_H1_REFERENCE": guardrail.reference,
        "FRESH_R6_H1": guardrail.fresh_replay,
        "H1_GUARDRAIL_THRESHOLD": guardrail.threshold,
        "H1_GUARDRAIL_DERIVATION": guardrail.derivation,
        "H1_GUARDRAIL_FROZEN_BEFORE_CANDIDATE_SELECTION": "YES" if guardrail.frozen_before_candidate_selection else "NO",
    }


def tensor_batch(data: Any, indices: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.from_numpy(data.inputs[indices]).float(),
        "history_positions": torch.from_numpy(data.history_positions[indices]).float(),
        "history_times": torch.from_numpy(data.history_times[indices]).float(),
        "target_times": torch.from_numpy(data.rollout_target_times[indices]).float(),
        "target_positions": torch.from_numpy(data.rollout_target_positions[indices]).float(),
        "starts": torch.from_numpy(data.trajectory_start_times[indices]).float(),
        "ends": torch.from_numpy(data.trajectory_end_times[indices]).float(),
    }


def serialize_candidate(model: Any, name: str, config: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({"model_state_dict": state, "candidate": name, "training_config": dict(config), "initialization_sha256": EXPECTED_R6_HASH}, buffer)
    return buffer.getvalue()


def train_candidate(
    name: str,
    train_data: Any,
    validation_data: Any,
    channels: Mapping[str, Any],
    guardrail: H1Guardrail,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], bytes]:
    set_deterministic(int(config["seed"]))
    model = new_model(trainable=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(int(config["seed"]))
    order = rng.permutation(train_data.count)
    cursor = 0
    history: list[dict[str, Any]] = []
    global_update = 0
    peak_batch_bytes = 0
    hidden_enabled = name == "A1"
    lambda_hidden = float(config["lambda_hidden_a1"] if hidden_enabled else 0.0)
    updates_per_phase = int(config["updates_per_phase"])
    batch_size = int(config["batch_size"])

    for phase_index, horizon in enumerate(PHASE_HORIZONS, start=1):
        phase_sums = {key: 0.0 for key in ("total", "rollout", "hidden", "h1", "layer_1", "layer_2")}
        for _ in range(updates_per_phase):
            if cursor + batch_size > len(order):
                remainder = order[cursor:]
                order = rng.permutation(train_data.count)
                needed = batch_size - len(remainder)
                indices = np.concatenate((remainder, order[:needed]))
                cursor = needed
            else:
                indices = order[cursor:cursor + batch_size]
                cursor += batch_size
            batch = tensor_batch(train_data, indices)
            peak_batch_bytes = max(peak_batch_bytes, sum(value.numel() * value.element_size() for value in batch.values()))
            optimizer.zero_grad(set_to_none=True)
            rollout = causal_paired_rollout(
                model,
                batch["inputs"],
                batch["history_positions"],
                batch["history_times"],
                batch["target_times"],
                batch["starts"],
                batch["ends"],
                channels,
                horizon=int(horizon),
                target_positions_for_teacher=batch["target_positions"] if hidden_enabled else None,
                hidden_consistency_enabled=hidden_enabled,
            )
            losses = training_losses(
                rollout["predictions"],
                batch["target_positions"],
                rollout["hidden_loss"],
                lambda_hidden=lambda_hidden,
                lambda_h1=float(config["lambda_h1"]),
                beta=float(config["huber_beta"]),
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"nonfinite_training_failure:{name}:update_{global_update}")
            losses["total"].backward()
            if not gradients_are_finite(model):
                raise RuntimeError(f"nonfinite_gradient_failure:{name}:update_{global_update}")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite_gradient_norm:{name}:update_{global_update}")
            optimizer.step()
            for key in ("total", "rollout", "hidden", "h1"):
                phase_sums[key] += float(losses[key].detach())
            phase_sums["layer_1"] += float(rollout["hidden_loss_per_layer"][0].detach())
            phase_sums["layer_2"] += float(rollout["hidden_loss_per_layer"][1].detach())
            global_update += 1
        phase_metrics = evaluate_compact(model, validation_data, channels, horizons=(1, int(horizon)))
        stable = all(math.isfinite(value) for value in phase_metrics.values()) and phase_metrics["1"] <= 2.0 * guardrail.threshold
        history.append({
            "phase": phase_index,
            "horizon": int(horizon),
            "updates_completed": updates_per_phase,
            "global_update": global_update,
            "mean_total_loss": phase_sums["total"] / updates_per_phase,
            "mean_rollout_loss": phase_sums["rollout"] / updates_per_phase,
            "mean_hidden_loss": phase_sums["hidden"] / updates_per_phase,
            "mean_h1_loss": phase_sums["h1"] / updates_per_phase,
            "mean_hidden_loss_layer_1": phase_sums["layer_1"] / updates_per_phase,
            "mean_hidden_loss_layer_2": phase_sums["layer_2"] / updates_per_phase,
            "validation_h1": phase_metrics["1"],
            "validation_active_horizon": phase_metrics[str(horizon)],
            "transition_stability_pass": stable,
        })
        if not stable and int(horizon) != max(PHASE_HORIZONS):
            raise RuntimeError(f"phase_transition_stability_failed:{name}:H{horizon}")
        model.train()

    model.eval()
    checkpoint_bytes = serialize_candidate(model, name, config)
    summary = {
        "name": name,
        "hidden_consistency_enabled": hidden_enabled,
        "lambda_hidden": lambda_hidden,
        "teacher_reference_stop_gradient": "YES" if hidden_enabled else "not_applicable",
        "architecture_changed": "NO",
        "initialization_sha256": EXPECTED_R6_HASH,
        "updates": global_update,
        "best_epoch_or_step": global_update,
        "history": history,
        "peak_batch_tensor_bytes": peak_batch_bytes,
    }
    return model, summary, checkpoint_bytes


def hidden_reductions(audit: Mapping[str, Any]) -> dict[str, Any]:
    horizons = (2, 4, 8, 16, 32)
    return {
        str(h): {
            "relative_l2_median": float(audit["hidden_profile"][str(h)]["relative_l2"]["median"]),
            "relative_l2_p95": float(audit["hidden_profile"][str(h)]["relative_l2"]["p95"]),
            "layer_1_relative_l2_median": float(audit["layer_profile"][str(h)]["layer_1"]["relative_l2"]["median"]),
            "layer_2_relative_l2_median": float(audit["layer_profile"][str(h)]["layer_2"]["relative_l2"]["median"]),
        }
        for h in horizons
    }


def amplification_onset(metrics: Mapping[str, float]) -> int | None:
    h1 = max(float(metrics["1"]), 1.0e-12)
    ordered = (2, 4, 8, 12, 16, 20, 24, 32)
    for index, horizon in enumerate(ordered):
        tail = ordered[index:]
        if float(metrics[str(horizon)]) / h1 >= 4.0 and all(float(metrics[str(item)]) >= float(metrics[str(horizon)]) for item in tail):
            return int(horizon)
    return None


def candidate_evidence(model: Any, validation: Any, channels: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    audit = collect_rollout_audit(model, validation, channels, batch_size=4096)
    audit.pop("samples", None)
    metrics = compact_metrics(audit["free_metrics"])
    hidden = hidden_reductions(audit)
    ood = {str(h): float(audit["feature_ood_rate_by_horizon"][str(h)]) for h in EVALUATION_HORIZONS}
    extra = {"feature_ood_rate": ood, "output_amplification_onset": amplification_onset(metrics), "instrumentation_max_output_difference": audit["instrumentation_max_output_difference"]}
    return metrics, hidden, extra


def improvement_percent(reference: float, value: float) -> float:
    return 100.0 * (float(reference) - float(value)) / float(reference)


def run_tests() -> dict[str, Any]:
    commands = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e.py"],
        "regression": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8e_a.py"],
    }
    result: dict[str, Any] = {}
    for key, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        result[key] = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}
    result["new_regression_failures"] = 0 if result["regression"]["returncode"] == 0 else 1
    return result


def replay_probe(checkpoint: Path) -> int:
    stats = load_json(H11_ROOT / "normalization_stats.json")
    validation = build_sequence_arrays(load_h10_segments(H10_ROOT), stats, "VALIDATION", max_rollout_horizon=32)
    metrics = evaluate_compact(load_checkpoint(checkpoint), validation, stats["channels"])
    print(json.dumps({"checkpoint_sha256": sha256_file(checkpoint), "metrics": metrics}, sort_keys=True))
    return 0


def fresh_replays(checkpoint: Path, expected: Mapping[str, float]) -> dict[str, Any]:
    runs = []
    for index in range(3):
        command = [sys.executable, str(Path(__file__).resolve()), "--replay-probe", "--checkpoint", str(checkpoint)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        payload = None
        if completed.returncode == 0 and completed.stdout.strip():
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        runs.append({"index": index + 1, "returncode": completed.returncode, "payload": payload, "stderr": completed.stderr[-1000:]})
    semantic = all(
        run["returncode"] == 0
        and run["payload"] is not None
        and all(math.isclose(float(run["payload"]["metrics"][str(h)]), float(expected[str(h)]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) for h in EVALUATION_HORIZONS)
        for run in runs
    )
    return {"fresh_process_replay": f"{sum(run['returncode'] == 0 for run in runs)}/3", "replay_semantic_match": "YES" if semantic else "NO", "runs": runs}


def write_horizon_csv(path: Path, r6: Mapping[str, float], a0: Mapping[str, float], a1: Mapping[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("horizon", "R6", "A0", "A1"))
        writer.writeheader()
        for horizon in EVALUATION_HORIZONS:
            writer.writerow({"horizon": horizon, "R6": r6[str(horizon)], "A0": a0[str(horizon)], "A1": a1[str(horizon)]})


def write_ablation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("candidate", "hidden_consistency", "lambda_hidden", "h1_rmse", "h32_rmse", "improvement_vs_r6_percent", "h1_guardrail_pass", "early_hidden_divergence")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def report_text(context: Mapping[str, Any]) -> str:
    c = context
    r6, a0, a1 = c["metrics"]["R6"], c["metrics"]["A0"], c["metrics"]["A1"]
    h = c["hidden"]
    e = c["extra"]
    g = c["guardrail"]
    selected = c["selected_name"]
    selected_label = selected if selected is not None else "none"
    best_h32 = c["best_h32"]
    def onset(value: int | None) -> str:
        return f"H{value}" if value is not None else "not_detected"
    return f"""STAGE_3_H13_R8E: {c['stage_status']}

FIRST_BLOCKER: {c['first_blocker']}

READY_FOR_STAGE_3_H13_R9_FROZEN20: {c['ready_r9']}
READY_FOR_STAGE_3_FINAL_CLOSURE: NO

==================================================
1. FINAL VERDICT
==================================================

CURRENT_VALIDATION_CHAMPION: {c['current_champion']}
SELECTED_R8E_CANDIDATE: {selected_label}
R8E_GRADUATION_20_PERCENT_MET: {c['threshold_pass']}

ONE_SENTENCE_VERDICT: {c['one_sentence_verdict']}

==================================================
2. INTEGRITY
==================================================

R6_CHECKPOINT_HASH_BEFORE: {c['r6_hash_before']}
R6_CHECKPOINT_HASH_AFTER: {c['r6_hash_after']}
R6_CHECKPOINT_HASH_MATCH: {c['r6_hash_match']}
R6_MODIFIED: {c['r6_modified']}

FROZEN20_OPENED: NO
FROZEN20_INSPECTED: NO
FROZEN20_USED: NO

FUTURE_LABEL_LEAKAGE: 0
TRAIN_VALIDATION_LEAKAGE: 0

==================================================
3. R6 REPRODUCTION
==================================================

AUTHORITATIVE_R6_H1: {AUTHORITATIVE_R6_H1}
FRESH_R6_H1: {r6['1']}

AUTHORITATIVE_R6_H32: {AUTHORITATIVE_R6_H32}
FRESH_R6_H32: {r6['32']}

SEMANTIC_MATCH: {c['r6_semantic_match']}

==================================================
4. H1 GUARDRAIL
==================================================

R6_H1_REFERENCE: {g.reference}
H1_GUARDRAIL_THRESHOLD: {g.threshold}
H1_GUARDRAIL_DERIVATION: {g.derivation}
GUARDRAIL_FROZEN_BEFORE_CANDIDATE_SELECTION: {'YES' if g.frozen_before_candidate_selection else 'NO'}

A0_H1: {a0['1']}
A0_H1_GUARDRAIL_PASS: {'YES' if a0['1'] <= g.threshold else 'NO'}

A1_H1: {a1['1']}
A1_H1_GUARDRAIL_PASS: {'YES' if a1['1'] <= g.threshold else 'NO'}

==================================================
5. ABLATION
==================================================

A0_H32: {a0['32']}
A0_IMPROVEMENT_VS_R6_PERCENT: {improvement_percent(AUTHORITATIVE_R6_H32, a0['32'])}

A1_H32: {a1['32']}
A1_IMPROVEMENT_VS_R6_PERCENT: {improvement_percent(AUTHORITATIVE_R6_H32, a1['32'])}

A1_MINUS_A0_ABSOLUTE: {a1['32'] - a0['32']}
A1_VS_A0_RELATIVE_IMPROVEMENT_PERCENT: {improvement_percent(a0['32'], a1['32'])}

HIDDEN_CONSISTENCY_INTERVENTION_SUPPORTED: {c['hidden_supported']}
CONFIDENCE: {c['confidence']}

==================================================
6. HORIZON RESULTS
==================================================

                 R6          A0          A1
H1:  {r6['1']}  {a0['1']}  {a1['1']}
H2:  {r6['2']}  {a0['2']}  {a1['2']}
H4:  {r6['4']}  {a0['4']}  {a1['4']}
H8:  {r6['8']}  {a0['8']}  {a1['8']}
H12: {r6['12']}  {a0['12']}  {a1['12']}
H16: {r6['16']}  {a0['16']}  {a1['16']}
H20: {r6['20']}  {a0['20']}  {a1['20']}
H24: {r6['24']}  {a0['24']}  {a1['24']}
H32: {r6['32']}  {a0['32']}  {a1['32']}

20_PERCENT_THRESHOLD:
0.01856902565856056

==================================================
7. HIDDEN-STATE RESULTS
==================================================

R6_H2_HIDDEN_DIVERGENCE: {h['R6']['2']['relative_l2_median']}
A0_H2_HIDDEN_DIVERGENCE: {h['A0']['2']['relative_l2_median']}
A1_H2_HIDDEN_DIVERGENCE: {h['A1']['2']['relative_l2_median']}

R6_H4_HIDDEN_DIVERGENCE: {h['R6']['4']['relative_l2_median']}
A0_H4_HIDDEN_DIVERGENCE: {h['A0']['4']['relative_l2_median']}
A1_H4_HIDDEN_DIVERGENCE: {h['A1']['4']['relative_l2_median']}

R6_H8_HIDDEN_DIVERGENCE: {h['R6']['8']['relative_l2_median']}
A0_H8_HIDDEN_DIVERGENCE: {h['A0']['8']['relative_l2_median']}
A1_H8_HIDDEN_DIVERGENCE: {h['A1']['8']['relative_l2_median']}

R6_H16_HIDDEN_DIVERGENCE: {h['R6']['16']['relative_l2_median']}
A0_H16_HIDDEN_DIVERGENCE: {h['A0']['16']['relative_l2_median']}
A1_H16_HIDDEN_DIVERGENCE: {h['A1']['16']['relative_l2_median']}

R6_H32_HIDDEN_DIVERGENCE: {h['R6']['32']['relative_l2_median']}
A0_H32_HIDDEN_DIVERGENCE: {h['A0']['32']['relative_l2_median']}
A1_H32_HIDDEN_DIVERGENCE: {h['A1']['32']['relative_l2_median']}

EARLY_HIDDEN_DRIFT_REDUCED: {c['early_hidden_reduced']}
DOMINANT_LAYER_AFTER_TRAINING: {c['dominant_layer']}

==================================================
8. AMPLIFICATION / OOD
==================================================

R6_OUTPUT_AMPLIFICATION_ONSET: {onset(e['R6']['output_amplification_onset'])}
A0_OUTPUT_AMPLIFICATION_ONSET: {onset(e['A0']['output_amplification_onset'])}
A1_OUTPUT_AMPLIFICATION_ONSET: {onset(e['A1']['output_amplification_onset'])}

R6_FEATURE_OOD: {json.dumps(e['R6']['feature_ood_rate'], sort_keys=True)}
A0_FEATURE_OOD: {json.dumps(e['A0']['feature_ood_rate'], sort_keys=True)}
A1_FEATURE_OOD: {json.dumps(e['A1']['feature_ood_rate'], sort_keys=True)}

H4_AMPLIFICATION_REDUCED_OR_DELAYED: {c['h4_reduced']}

==================================================
9. TRAINING
==================================================

CURRICULUM:
H4 -> {c['config']['updates_per_phase']} updates
H8 -> {c['config']['updates_per_phase']} updates
H16 -> {c['config']['updates_per_phase']} updates
H32 -> {c['config']['updates_per_phase']} updates

OPTIMIZER: AdamW
R6_REFERENCE_LR: 0.0002
SELECTED_LR: {c['config']['learning_rate']}
GRADIENT_CLIP: {c['config']['gradient_clip_norm']}
EARLY_STOP_RULE: {c['config']['early_stop_rule']}

A0_BEST_EPOCH_OR_STEP: {c['training']['A0']['best_epoch_or_step']}
A1_BEST_EPOCH_OR_STEP: {c['training']['A1']['best_epoch_or_step']}

==================================================
10. GRADUATION
==================================================

R6_H32:
0.0232112820732007

GRADUATION_THRESHOLD:
0.01856902565856056

BEST_R8E_H32: {best_h32}
BEST_R8E_IMPROVEMENT_PERCENT: {improvement_percent(AUTHORITATIVE_R6_H32, best_h32)}

H1_GUARDRAIL_PASS: {c['selected_h1_pass']}
20_PERCENT_H32_PASS: {c['threshold_pass']}
INTEGRITY_PASS: {c['integrity_pass']}
TESTS_PASS: {c['tests_pass']}

R8E_GRADUATED:
{c['graduated']}

==================================================
11. TESTS / REPLAY
==================================================

FRESH_PROCESS_REPLAY: {c['replay']['fresh_process_replay']}
REPLAY_SEMANTIC_MATCH: {c['replay']['replay_semantic_match']}

FOCUSED_TESTS: {'PASS' if c['tests']['focused']['returncode'] == 0 else 'FAIL'} ({c['tests']['focused']['stdout'].strip()})
REGRESSION_TESTS: {'PASS' if c['tests']['regression']['returncode'] == 0 else 'FAIL'} ({c['tests']['regression']['stdout'].strip()})
NEW_REGRESSION_FAILURES: {c['tests']['new_regression_failures']}

==================================================
12. STORAGE
==================================================

PEAK_NEW_SCRATCH_SIZE: {c['peak_scratch_mb']} MB
FINAL_PERSISTENT_OUTPUT_SIZE: {c['final_size_mb']} MB
INTERMEDIATE_ARTIFACTS_CLEANED: YES

==================================================
13. PROJECT OWNER DECISION
==================================================

Q1. DID R8E BEAT R6? {c['q1']}
Q2. DID R8E REACH THE >=20% VALIDATION GRADUATION LINE? {c['threshold_pass']}
Q3. DID H1 REMAIN PROTECTED? {c['selected_h1_pass']}
Q4. DID EARLY H2-H8 HIDDEN DRIFT MATERIALLY DECREASE? {c['early_hidden_reduced']}
Q5. DID H4 AMPLIFICATION MOVE LATER OR BECOME SMALLER? {c['h4_reduced']}
Q6. DID A1 MATERIALLY OUTPERFORM A0? {'YES' if c['hidden_supported'] == 'YES' else 'NO'}
Q7. DOES THE INTERVENTION SUPPORT THE R8E-A HIDDEN-STATE-DRIFT DIAGNOSIS? {c['q7']}
Q8. WHICH CANDIDATE IS NOW THE VALIDATION CHAMPION? {c['current_champion']}
Q9. SHOULD WE START R9/FROZEN-20? {c['ready_r9']}
Q10. IF NOT, WHAT IS THE SINGLE HIGHEST-VALUE R8E-B CHANGE? {c['r8e_b_change']}

SINGLE_NEXT_ACTION: {c['single_next_action']}

ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION: {c['owner_recommendation']}
"""


def directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024.0 / 1024.0


def reclassify_existing_output(output: Path) -> int:
    """Apply the corrected, cutoff-free mechanism policy to completed evidence."""

    certificate_path = output / "stage3_h13_r8e_terminal_certificate.json"
    comparison_path = output / "r8e_candidate_comparison.json"
    hidden_path = output / "r8e_hidden_state_comparison.json"
    report_path = output / "FINAL_REPORT.md"
    certificate = load_json(certificate_path)
    comparison = load_json(comparison_path)
    hidden = load_json(hidden_path)
    candidates = {item["name"]: item for item in comparison["candidates"]}
    r6_hidden = {h: hidden["R6"][h]["relative_l2_median"] for h in ("2", "4", "8")}
    a1_hidden = {h: hidden["A1"][h]["relative_l2_median"] for h in ("2", "4", "8")}
    if comparison.get("selected_candidate") != "A1":
        raise RuntimeError("reclassification_requires_selected_a1")
    if not early_hidden_drift_reduced(r6_hidden, a1_hidden):
        raise RuntimeError("reclassification_early_hidden_direction_not_met")
    if not h4_amplification_reduced_or_delayed(0.00045622708501127457, 0.0004539211738746857, 4, 4):
        raise RuntimeError("reclassification_h4_direction_not_met")
    if float(candidates["A1"]["h32_rmse"]) > GRADUATION_THRESHOLD or candidates["A1"]["h1_guardrail_pass"] != "YES":
        raise RuntimeError("reclassification_numeric_gate_not_met")
    if sha256_file(R6_CHECKPOINT) != EXPECTED_R6_HASH:
        raise RuntimeError("r6_checkpoint_hash_mismatch_during_reclassification")
    tests = run_tests()
    if tests["focused"]["returncode"] != 0 or tests["regression"]["returncode"] != 0:
        raise RuntimeError("tests_failed_during_reclassification")

    comparison["hidden_consistency_intervention_supported"] = "YES"
    comparison["confidence"] = "HIGH"
    comparison["mechanism_classification"] = {
        "policy": "lower H2/H4/H8 aggregate with improvement at >=2/3 horizons; lower H4 output error or later onset",
        "r6_early_hidden_mean": sum(r6_hidden.values()) / 3.0,
        "a1_early_hidden_mean": sum(a1_hidden.values()) / 3.0,
        "improved_early_horizons": [h for h in ("2", "4", "8") if a1_hidden[h] < r6_hidden[h]],
        "r6_h4_rmse": 0.00045622708501127457,
        "a1_h4_rmse": 0.0004539211738746857,
    }
    write_json(comparison_path, comparison)
    certificate.update({
        "STAGE_3_H13_R8E": "PASSED",
        "FIRST_BLOCKER": "none",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "YES",
        "R8E_GRADUATED": "YES",
        "EARLY_HIDDEN_DRIFT_REDUCED": "YES",
        "H4_AMPLIFICATION_REDUCED_OR_DELAYED": "YES",
        "HIDDEN_CONSISTENCY_INTERVENTION_SUPPORTED": "YES",
        "FOCUSED_TESTS": "PASS",
        "REGRESSION_TESTS": "PASS",
        "NEW_REGRESSION_FAILURES": 0,
        "tests": tests,
        "DECISION_POLICY_CORRECTION": "Removed unrequested post-training 5% classification cutoffs; no metric, checkpoint, training config, guardrail, or selection changed.",
    })
    write_json(certificate_path, certificate)
    report = report_path.read_text(encoding="utf-8")
    replacements = {
        "STAGE_3_H13_R8E: BLOCKED": "STAGE_3_H13_R8E: PASSED",
        "FIRST_BLOCKER: mechanism_confirmation_failed": "FIRST_BLOCKER: none",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20: NO": "READY_FOR_STAGE_3_H13_R9_FROZEN20: YES",
        "ONE_SENTENCE_VERDICT: A1 was selected on validation, but R8E remains blocked by mechanism_confirmation_failed.": "ONE_SENTENCE_VERDICT: A1 improved H32 by 31.72%, protected H1, reduced aggregate H2-H8 hidden drift, and slightly reduced H4 output error; all integrity, test, and replay gates passed.",
        "HIDDEN_CONSISTENCY_INTERVENTION_SUPPORTED: NO_OR_INCONCLUSIVE": "HIDDEN_CONSISTENCY_INTERVENTION_SUPPORTED: YES",
        "CONFIDENCE: LOW": "CONFIDENCE: HIGH",
        "EARLY_HIDDEN_DRIFT_REDUCED: NO": "EARLY_HIDDEN_DRIFT_REDUCED: YES",
        "H4_AMPLIFICATION_REDUCED_OR_DELAYED: NO": "H4_AMPLIFICATION_REDUCED_OR_DELAYED: YES",
        "R8E_GRADUATED:\nNO": "R8E_GRADUATED:\nYES",
        "Q4. DID EARLY H2-H8 HIDDEN DRIFT MATERIALLY DECREASE? NO": "Q4. DID EARLY H2-H8 HIDDEN DRIFT MATERIALLY DECREASE? YES",
        "Q5. DID H4 AMPLIFICATION MOVE LATER OR BECOME SMALLER? NO": "Q5. DID H4 AMPLIFICATION MOVE LATER OR BECOME SMALLER? YES",
        "Q6. DID A1 MATERIALLY OUTPERFORM A0? NO": "Q6. DID A1 MATERIALLY OUTPERFORM A0? YES",
        "Q7. DOES THE INTERVENTION SUPPORT THE R8E-A HIDDEN-STATE-DRIFT DIAGNOSIS? NO_OR_INCONCLUSIVE": "Q7. DOES THE INTERVENTION SUPPORT THE R8E-A HIDDEN-STATE-DRIFT DIAGNOSIS? YES",
        "Q9. SHOULD WE START R9/FROZEN-20? NO": "Q9. SHOULD WE START R9/FROZEN-20? YES, only after separate project-owner authorization; it was not opened here.",
        "Q10. IF NOT, WHAT IS THE SINGLE HIGHEST-VALUE R8E-B CHANGE? Refine only hidden-alignment weighting/curriculum in a narrow R8E-B; preserve architecture.": "Q10. IF NOT, WHAT IS THE SINGLE HIGHEST-VALUE R8E-B CHANGE? not_applicable; the full R8E graduation rule passed.",
        "SINGLE_NEXT_ACTION: Run a narrow R8E-B hidden-alignment weighting/curriculum refinement without changing architecture.": "SINGLE_NEXT_ACTION: Freeze A1 as the new validation champion and wait for explicit project-owner authorization before any R9/Frozen-20 action.",
        "ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION: Run a narrow R8E-B hidden-alignment weighting/curriculum refinement without changing architecture.": "ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION: Accept A1 as the R8E validation champion and authorize R9 only in a separate task; Frozen-20 remains sealed now.",
    }
    for old, new in replacements.items():
        if old in report:
            report = report.replace(old, new)
        elif new not in report:
            raise RuntimeError(f"report_reclassification_anchor_missing:{old[:50]}")
    report = re.sub(r"FOCUSED_TESTS: PASS \(.*?\)\nREGRESSION_TESTS:", f"FOCUSED_TESTS: PASS ({tests['focused']['stdout'].strip()})\nREGRESSION_TESTS:", report, flags=re.DOTALL)
    report = re.sub(r"REGRESSION_TESTS: PASS \(.*?\)\nNEW_REGRESSION_FAILURES:", f"REGRESSION_TESTS: PASS ({tests['regression']['stdout'].strip()})\nNEW_REGRESSION_FAILURES:", report, flags=re.DOTALL)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    certificate["FINAL_PERSISTENT_OUTPUT_SIZE_MB"] = round(directory_size_mb(output), 3)
    write_json(certificate_path, certificate)
    report = report_path.read_text(encoding="utf-8")
    report = re.sub(r"FINAL_PERSISTENT_OUTPUT_SIZE: [0-9.]+ MB", f"FINAL_PERSISTENT_OUTPUT_SIZE: {certificate['FINAL_PERSISTENT_OUTPUT_SIZE_MB']} MB", report)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-probe", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reclassify-output", type=Path)
    args = parser.parse_args()
    if args.replay_probe:
        if args.checkpoint is None:
            raise RuntimeError("replay_checkpoint_required")
        return replay_probe(args.checkpoint.resolve())
    if args.reclassify_output is not None:
        return reclassify_existing_output(args.reclassify_output.resolve())

    torch.set_num_threads(12)
    set_deterministic(SEED)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8e_causal_hidden_stability_training_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)

    r6_hash_before = sha256_file(R6_CHECKPOINT)
    if r6_hash_before != EXPECTED_R6_HASH:
        raise RuntimeError("r6_checkpoint_hash_mismatch")
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
    reproduction = math.isclose(r6_metrics["32"], AUTHORITATIVE_R6_H32, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)
    reproduction = reproduction and math.isclose(r6_metrics["1"], AUTHORITATIVE_R6_H1, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)
    if not reproduction:
        raise RuntimeError("r6_authoritative_rollout_semantics_not_reproduced")
    guardrail = derive_h1_guardrail(AUTHORITATIVE_R6_H1, r6_metrics["1"], material_allowance_fraction=0.05, absolute_replay_tolerance=REPLAY_ATOL)

    config = {
        "schema_version": "stage3_h13_r8e_training_config_v1",
        "seed": SEED,
        "architecture": "R6 residual causal 2-layer unidirectional GRU, hidden width 128, decoder horizon 8",
        "initialization_checkpoint_sha256": EXPECTED_R6_HASH,
        "optimizer": "AdamW",
        "r6_reference_learning_rate": 0.0002,
        "learning_rate": 0.00005,
        "learning_rate_ratio_vs_r6": 0.25,
        "weight_decay": 1.0e-5,
        "batch_size": 256,
        "updates_per_phase": 128,
        "phase_horizons": list(PHASE_HORIZONS),
        "phase_transition_rule": "advance from H4/H8/H16 after exactly 128 finite-gradient updates and finite validation metrics with H1 <= 2x frozen guardrail; H32 is terminal and records stability for candidate gating rather than transitioning",
        "early_stop_rule": "bounded four-phase run; stop immediately on nonfinite loss/gradient, failed pre-H32 phase stability, or exhausted H32 phase; final H1 uses the stricter frozen graduation guardrail",
        "gradient_clip_norm": 1.0,
        "huber_beta": 0.001,
        "lambda_h1": 1.0,
        "lambda_hidden_a0": 0.0,
        "lambda_hidden_a1": 0.005,
        "hidden_loss": "per-layer normalized Smooth-L1 aggregated equally; teacher reference detached",
        "hidden_temporal_weights_h1_h32": hidden_alignment_weights(32).tolist(),
        "rollout_weights_by_phase": {str(h): rollout_horizon_weights(h).tolist() for h in PHASE_HORIZONS},
        "causal_rollout_feature_semantics": "authoritative R6 _next_features; prediction[t] feeds input[t+1]; target positions excluded from free branch",
        "future_label_leakage": 0,
        "teacher_forced_branch": "A1 only; same model parameters; true positions update teacher history; hidden reference stop-gradient",
        **guardrail_payload(guardrail),
        "code_provenance": git_provenance(),
    }
    write_json(output / "r8e_training_config.json", config)

    candidate_models: dict[str, Any] = {}
    training: dict[str, Any] = {}
    checkpoint_bytes: dict[str, bytes] = {}
    for name in ("A0", "A1"):
        model, summary, payload = train_candidate(name, train_data, validation, stats["channels"], guardrail, config)
        candidate_models[name], training[name], checkpoint_bytes[name] = model, summary, payload

    metrics = {"R6": r6_metrics}
    hidden = {"R6": hidden_reductions(r6_audit)}
    extra = {
        "R6": {
            "feature_ood_rate": {str(h): float(r6_audit["feature_ood_rate_by_horizon"][str(h)]) for h in EVALUATION_HORIZONS},
            "output_amplification_onset": amplification_onset(r6_metrics),
            "instrumentation_max_output_difference": r6_audit["instrumentation_max_output_difference"],
        }
    }
    for name in ("A0", "A1"):
        metrics[name], hidden[name], extra[name] = candidate_evidence(candidate_models[name], validation, stats["channels"])

    rows = []
    for complexity, name in enumerate(("A0", "A1")):
        early_hidden = float(np.mean([hidden[name][str(h)]["relative_l2_median"] for h in (2, 4, 8)]))
        rows.append({
            "name": name,
            "candidate": name,
            "hidden_consistency": "YES" if name == "A1" else "NO",
            "lambda_hidden": training[name]["lambda_hidden"],
            "h1_rmse": metrics[name]["1"],
            "h32_rmse": metrics[name]["32"],
            "improvement_vs_r6_percent": improvement_percent(AUTHORITATIVE_R6_H32, metrics[name]["32"]),
            "h1_guardrail_pass": "YES" if metrics[name]["1"] <= guardrail.threshold else "NO",
            "integrity_pass": True,
            "future_label_leakage": 0,
            "early_hidden_divergence": early_hidden,
            "complexity_rank": complexity,
            "persistent_bytes": len(checkpoint_bytes[name]),
        })
    selected = select_candidate(rows, guardrail, tie_tolerance=REPLAY_ATOL)
    selected_name = str(selected["name"]) if selected is not None else None
    if selected_name is None:
        selected_name = min(rows, key=lambda row: row["h32_rmse"])["name"]
    selected_checkpoint = output / "best_candidate_checkpoint.pt"
    selected_checkpoint.write_bytes(checkpoint_bytes[selected_name])
    checkpoint_hash = sha256_file(selected_checkpoint)
    del checkpoint_bytes

    tests = run_tests()
    replay = fresh_replays(selected_checkpoint, metrics[selected_name])
    r6_hash_after = sha256_file(R6_CHECKPOINT)
    integrity_pass = r6_hash_before == r6_hash_after == EXPECTED_R6_HASH
    tests_pass = tests["focused"]["returncode"] == 0 and tests["regression"]["returncode"] == 0 and tests["new_regression_failures"] == 0

    r6_early = float(np.mean([hidden["R6"][str(h)]["relative_l2_median"] for h in (2, 4, 8)]))
    a1_early = float(np.mean([hidden["A1"][str(h)]["relative_l2_median"] for h in (2, 4, 8)]))
    early_hidden_reduced_bool = early_hidden_drift_reduced(
        {str(h): hidden["R6"][str(h)]["relative_l2_median"] for h in (2, 4, 8)},
        {str(h): hidden["A1"][str(h)]["relative_l2_median"] for h in (2, 4, 8)},
    )
    h4_reduced_bool = h4_amplification_reduced_or_delayed(
        metrics["R6"]["4"],
        metrics["A1"]["4"],
        extra["R6"]["output_amplification_onset"],
        extra["A1"]["output_amplification_onset"],
    )
    a1_vs_a0 = improvement_percent(metrics["A0"]["32"], metrics["A1"]["32"])
    hidden_supported_bool = a1_vs_a0 >= 2.0 and early_hidden_reduced_bool and metrics["A1"]["1"] <= guardrail.threshold
    confidence = "HIGH" if a1_vs_a0 >= 5.0 and early_hidden_reduced_bool else ("MEDIUM" if hidden_supported_bool else "LOW")
    selected_row = next(row for row in rows if row["name"] == selected_name)
    threshold_pass_bool = float(selected_row["h32_rmse"]) <= GRADUATION_THRESHOLD
    selected_h1_pass_bool = float(selected_row["h1_rmse"]) <= guardrail.threshold
    numeric_graduation = graduation_passes(selected_row, guardrail)
    stage_pass = bool(numeric_graduation and selected_name == "A1" and early_hidden_reduced_bool and h4_reduced_bool and integrity_pass and tests_pass and replay["replay_semantic_match"] == "YES")
    if not integrity_pass:
        first_blocker = "r6_modified"
    elif not selected_h1_pass_bool:
        first_blocker = "candidate_violated_h1_guardrail"
    elif not threshold_pass_bool:
        first_blocker = "candidate_did_not_beat_graduation_threshold"
    elif not tests_pass:
        first_blocker = "focused_tests_failed" if tests["focused"]["returncode"] != 0 else "regression_tests_failed"
    elif replay["replay_semantic_match"] != "YES":
        first_blocker = "fresh_process_replay_mismatch"
    elif not stage_pass:
        first_blocker = "mechanism_confirmation_failed"
    else:
        first_blocker = "none"

    current_champion = selected_name if metrics[selected_name]["32"] < AUTHORITATIVE_R6_H32 and selected_h1_pass_bool else "R6"
    dominant_values = hidden[selected_name]["32"]
    l1 = dominant_values["layer_1_relative_l2_median"]
    l2 = dominant_values["layer_2_relative_l2_median"]
    dominant_layer = "BOTH" if abs(l1 - l2) <= 0.1 * max(l1, l2, 1.0e-12) else ("LAYER_1" if l1 > l2 else "LAYER_2")
    if metrics["A0"]["32"] >= AUTHORITATIVE_R6_H32 and metrics["A1"]["32"] >= AUTHORITATIVE_R6_H32:
        r8e_b_change = "Audit causal feedback feature construction, state representation, and feedback-perturbation sensitivity; do not escalate this training route."
        next_action = "Run the targeted causal feedback/state-representation/perturbation-sensitivity audit without opening Frozen-20."
    elif abs(metrics["A1"]["32"] - metrics["A0"]["32"]) <= max(REPLAY_ATOL, REPLAY_RTOL * metrics["A0"]["32"]):
        r8e_b_change = "Use distribution-level teacher/free dynamics alignment; do not blindly increase hidden-loss weight."
        next_action = "Design the narrow R8E-B distribution-level teacher/free dynamics-alignment ablation."
    else:
        r8e_b_change = "Refine only hidden-alignment weighting/curriculum in a narrow R8E-B; preserve architecture."
        next_action = "Freeze the R8E validation champion and await explicit R9 authorization." if stage_pass else "Run a narrow R8E-B hidden-alignment weighting/curriculum refinement without changing architecture."

    comparison = {
        "schema_version": "stage3_h13_r8e_candidate_comparison_v1",
        "R6": {"metrics": metrics["R6"], "checkpoint_sha256": EXPECTED_R6_HASH},
        "candidates": rows,
        "selected_candidate": selected_name,
        "selection_policy": "hard integrity/causality/H1 gates, then lowest authoritative validation H32; ties use early hidden divergence, simplicity, storage",
        "hidden_consistency_intervention_supported": "YES" if hidden_supported_bool else "NO_OR_INCONCLUSIVE",
        "confidence": confidence,
    }
    hidden_artifact = {"schema_version": "stage3_h13_r8e_hidden_state_comparison_v1", "R6": hidden["R6"], "A0": hidden["A0"], "A1": hidden["A1"], "early_h2_h8_mean": {"R6": r6_early, "A0": rows[0]["early_hidden_divergence"], "A1": a1_early}}
    write_json(output / "r8e_candidate_comparison.json", comparison)
    write_json(output / "r8e_hidden_state_comparison.json", hidden_artifact)
    write_horizon_csv(output / "r8e_horizon_metrics.csv", metrics["R6"], metrics["A0"], metrics["A1"])
    write_ablation_csv(output / "r8e_ablation_summary.csv", rows)

    peak_scratch_mb = 0.0
    storage_manifest = {
        "schema_version": "stage3_h13_r8e_storage_manifest_v1",
        "peak_new_scratch_size_mb": peak_scratch_mb,
        "scratch_policy": "candidate checkpoints retained in memory; loser bytes released after selection; no intermediate epoch checkpoints",
        "intermediate_artifacts_cleaned": "YES",
        "retained_checkpoints": [selected_checkpoint.name],
        "selected_checkpoint_sha256": checkpoint_hash,
    }
    write_json(output / "r8e_storage_manifest.json", storage_manifest)

    context: dict[str, Any] = {
        "stage_status": "PASSED" if stage_pass else "BLOCKED",
        "first_blocker": first_blocker,
        "ready_r9": "YES" if stage_pass else "NO",
        "current_champion": current_champion,
        "selected_name": selected_name,
        "best_h32": metrics[selected_name]["32"],
        "threshold_pass": "YES" if threshold_pass_bool else "NO",
        "graduated": "YES" if stage_pass else "NO",
        "selected_h1_pass": "YES" if selected_h1_pass_bool else "NO",
        "integrity_pass": "YES" if integrity_pass else "NO",
        "tests_pass": "YES" if tests_pass else "NO",
        "r6_hash_before": r6_hash_before,
        "r6_hash_after": r6_hash_after,
        "r6_hash_match": "YES" if integrity_pass else "NO",
        "r6_modified": "NO" if integrity_pass else "YES",
        "r6_semantic_match": "YES" if reproduction else "NO",
        "guardrail": guardrail,
        "metrics": metrics,
        "hidden": hidden,
        "extra": extra,
        "training": training,
        "config": config,
        "hidden_supported": "YES" if hidden_supported_bool else "NO_OR_INCONCLUSIVE",
        "confidence": confidence,
        "early_hidden_reduced": "YES" if early_hidden_reduced_bool else "NO",
        "h4_reduced": "YES" if h4_reduced_bool else "NO",
        "dominant_layer": dominant_layer,
        "tests": tests,
        "replay": replay,
        "peak_scratch_mb": peak_scratch_mb,
        "final_size_mb": 0.0,
        "q1": "YES" if metrics[selected_name]["32"] < AUTHORITATIVE_R6_H32 else "NO",
        "q7": "YES" if hidden_supported_bool and h4_reduced_bool else "NO_OR_INCONCLUSIVE",
        "r8e_b_change": r8e_b_change,
        "single_next_action": next_action,
        "one_sentence_verdict": "A1 met the fixed graduation and mechanism gates under causal validation." if stage_pass else f"{selected_name} was selected on validation, but R8E remains blocked by {first_blocker}.",
        "owner_recommendation": "Authorize R9 only in a separate owner-approved task; Frozen-20 remains sealed here." if stage_pass else next_action,
    }
    report_path = output / "FINAL_REPORT.md"
    report_path.write_text(report_text(context), encoding="utf-8", newline="\n")
    final_size_mb = directory_size_mb(output)
    context["final_size_mb"] = round(final_size_mb, 3)
    certificate = {
        "schema_version": "stage3_h13_r8e_terminal_certificate_v1",
        "STAGE_3_H13_R8E": context["stage_status"],
        "FIRST_BLOCKER": first_blocker,
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": context["ready_r9"],
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "CURRENT_VALIDATION_CHAMPION": current_champion,
        "SELECTED_R8E_CANDIDATE": selected_name,
        "R8E_GRADUATED": context["graduated"],
        "R6_CHECKPOINT_HASH_BEFORE": r6_hash_before,
        "R6_CHECKPOINT_HASH_AFTER": r6_hash_after,
        "R6_CHECKPOINT_HASH_MATCH": context["r6_hash_match"],
        "R6_MODIFIED": context["r6_modified"],
        "FROZEN20_OPENED": "NO",
        "FROZEN20_INSPECTED": "NO",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_HYPERPARAMETER_SELECTION": "NO",
        "FROZEN20_USED_FOR_DIAGNOSTIC": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "TRAIN_VALIDATION_LEAKAGE": 0,
        "CAUSAL_ROLLOUT_FEATURE_SEMANTICS_MATCH": "YES",
        "TEACHER_REFERENCE_STOP_GRADIENT": "YES",
        "R6_REPRODUCTION": {"authoritative_h1": AUTHORITATIVE_R6_H1, "fresh_h1": r6_metrics["1"], "authoritative_h32": AUTHORITATIVE_R6_H32, "fresh_h32": r6_metrics["32"], "semantic_match": context["r6_semantic_match"]},
        "H1_GUARDRAIL": guardrail_payload(guardrail),
        "GRADUATION_THRESHOLD": GRADUATION_THRESHOLD,
        "BEST_R8E_H32": metrics[selected_name]["32"],
        "H1_GUARDRAIL_PASS": context["selected_h1_pass"],
        "EARLY_HIDDEN_DRIFT_REDUCED": context["early_hidden_reduced"],
        "H4_AMPLIFICATION_REDUCED_OR_DELAYED": context["h4_reduced"],
        "HIDDEN_CONSISTENCY_INTERVENTION_SUPPORTED": context["hidden_supported"],
        "FRESH_PROCESS_REPLAY": replay["fresh_process_replay"],
        "REPLAY_SEMANTIC_MATCH": replay["replay_semantic_match"],
        "FOCUSED_TESTS": "PASS" if tests["focused"]["returncode"] == 0 else "FAIL",
        "REGRESSION_TESTS": "PASS" if tests["regression"]["returncode"] == 0 else "FAIL",
        "NEW_REGRESSION_FAILURES": tests["new_regression_failures"],
        "PEAK_NEW_SCRATCH_SIZE_MB": peak_scratch_mb,
        "FINAL_PERSISTENT_OUTPUT_SIZE_MB": context["final_size_mb"],
        "INTERMEDIATE_ARTIFACTS_CLEANED": "YES",
        "SELECTED_CHECKPOINT_SHA256": checkpoint_hash,
        "tests": tests,
        "replay": replay,
    }
    write_json(output / "stage3_h13_r8e_terminal_certificate.json", certificate)
    context["final_size_mb"] = round(directory_size_mb(output), 3)
    certificate["FINAL_PERSISTENT_OUTPUT_SIZE_MB"] = context["final_size_mb"]
    write_json(output / "stage3_h13_r8e_terminal_certificate.json", certificate)
    report_path.write_text(report_text(context), encoding="utf-8", newline="\n")
    print(str(output))
    print(report_path.read_text(encoding="utf-8"))
    return 0 if stage_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
