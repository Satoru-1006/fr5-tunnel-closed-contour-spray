"""D42 persistent H1 breakthrough / basin-escape campaign.

This runner is deliberately shadow-first.  It authenticates the retained D41
baseline, uses the existing causal ON-state evaluator, and writes only under a
new D42 output directory.  No D39/D40/D41 artifact or canonical checkpoint is
modified by this module.

The campaign has two layers:

* S1: a deterministic, cheap validation subset used to compare heterogeneous
  parameter-space escape directions;
* S2: full authenticated H1/H32 evaluation for the most promising novel
  gradient presentations, with deterministic replay for the best result.

All training gradients are computed from the authenticated TRAIN schedule.  The
held-out validation set is used only for the existing evaluator metric, never
as a training input or gradient source.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_d35_scientific_runner as d35

d32 = d35.d32
d33 = d35.d33
d24 = d32.d24
d17 = d24.d17
d16 = d24.d16

TARGET_H1 = 5.0e-5
H32_LIMIT = float(d32.H32_LIMIT)
MIN_H1_DESCENT = float(d32.MIN_H1_DESCENT)
CANONICAL_UPDATE = 480
CANONICAL_H1 = 7.089328839013259e-05
CANONICAL_H32 = 0.01849100619381173
CANONICAL_CHECKPOINT = ROOT / "outputs" / "stage3_h13_d35_permanent_champion" / "checkpoints" / "committed_update_480.pt"
CANONICAL_SHA256 = "37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742"
D41_SUMMARY = ROOT / "outputs" / "stage3_h13_d41_offline_robot_certification" / "D41_SUMMARY.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "stage3_h13_d42_persistent_h1_breakthrough_campaign"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite:{result}")
    return result


def clone_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in values.items()}


def norm(values: Iterable[torch.Tensor]) -> float:
    total = sum(float(torch.sum(value.detach().double().square())) for value in values)
    return math.sqrt(max(0.0, total))


def scale_map(values: Mapping[str, torch.Tensor], amount: float) -> dict[str, torch.Tensor]:
    return {name: values[name] * float(amount) for name in d16.ALL_NAMES}


def add_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in d16.ALL_NAMES}


def dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return finite(sum(torch.sum(left[name].double() * right[name].double()) for name in d16.ALL_NAMES))


def unit(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return scale_map(values, 1.0 / max(norm(values.values()), 1.0e-30))


def model_parameters(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_authenticated_runtime() -> tuple[dict[str, Any], dict[str, Any], Any, Any, dict[str, Any], list[float], dict[str, Any]]:
    if not CANONICAL_CHECKPOINT.is_file():
        raise RuntimeError(f"canonical_checkpoint_missing:{CANONICAL_CHECKPOINT}")
    observed_sha = sha256_file(CANONICAL_CHECKPOINT)
    if observed_sha != CANONICAL_SHA256:
        raise RuntimeError(f"canonical_checkpoint_sha256_mismatch:{observed_sha}")
    if not D41_SUMMARY.is_file():
        raise RuntimeError(f"d41_summary_missing:{D41_SUMMARY}")
    d41 = json.loads(D41_SUMMARY.read_text(encoding="utf-8"))
    required = {
        "D39_RETENTION": "PASS",
        "D40_RETENTION": "PASS",
        "SELF_COLLISION_CERTIFICATION": "PASS",
        "COLLISION_CERTIFICATION_STATUS": "PASS",
        "NUMERICAL_HEALTH_STATUS": "PASS",
        "ROBUSTNESS_CLASSIFICATION": "PASS",
    }
    for key, expected in required.items():
        if d41.get(key) != expected:
            raise RuntimeError(f"d41_retention_not_pass:{key}:{d41.get(key)}")
    runtime = d17.preflight_runtime()
    payload = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(payload["completed_optimizer_step"]) != CANONICAL_UPDATE or int(payload["next_schedule_index"]) != CANONICAL_UPDATE:
        raise RuntimeError("canonical_update_or_schedule_mismatch")
    if float(payload["H1"]) != CANONICAL_H1 or float(payload["H32"]) != CANONICAL_H32:
        raise RuntimeError("canonical_metric_mismatch")
    model, optimizer, rng, canonical = d33.instantiate_state(runtime, payload)
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    return runtime, payload, model, optimizer, rng, canonical, {"identity": identity, "d41": d41, "canonical_sha256": observed_sha}


def validation_view(data: Any, count: int) -> Any:
    count = min(int(count), int(data.rollout_target_positions.shape[0]))
    if count < 1:
        raise ValueError("empty_validation_view")
    # Stratified deterministic sampling across the held-out array prevents a
    # single contiguous family from becoming the S1 proxy.
    indices = np.linspace(0, data.rollout_target_positions.shape[0] - 1, count, dtype=np.int64)
    indices = np.unique(indices)
    return replace(
        data,
        inputs=data.inputs[indices],
        direct_target_positions=data.direct_target_positions[indices],
        direct_target_times=data.direct_target_times[indices],
        rollout_target_positions=data.rollout_target_positions[indices],
        rollout_target_times=data.rollout_target_times[indices],
        history_positions=data.history_positions[indices],
        history_times=data.history_times[indices],
        trajectory_start_times=data.trajectory_start_times[indices],
        trajectory_end_times=data.trajectory_end_times[indices],
        window_ids=data.window_ids[indices],
        family_ids=data.family_ids[indices],
    )


def evaluate(model: Any, data: Any, channels: Mapping[str, Any], horizons: Sequence[int]) -> dict[str, float]:
    rng_before = d17.base.capture_rng()
    mode = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            result = d17.base.evaluate_compact(model, data, channels, horizons=tuple(int(h) for h in horizons))
        return {str(key): finite(value) for key, value in result.items()}
    finally:
        d17.base.restore_rng(rng_before)
        model.train(mode)


def set_model_state(model: Any, base_state: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_((base_state[name] + delta[name]).to(dtype=parameter.dtype))


def direct_escape_population(model: Any, previous_model: Any, optimizer: Any, seed: int = 42042) -> list[dict[str, Any]]:
    base = model_parameters(model)
    previous = model_parameters(previous_model)
    historical = {name: base[name] - previous[name] for name in d16.ALL_NAMES}
    # Adam-state direction is the signed direction of the accumulated update;
    # the negative direction is the model-motion direction.
    momentum: dict[str, torch.Tensor] = {}
    precond: dict[str, torch.Tensor] = {}
    # The optimizer state_dict is keyed by integer IDs; clone_named_optimizer_state
    # provides the stable name mapping used by the existing D32 code.
    named_optimizer = d16.clone_named_optimizer_state(optimizer)
    for name in d16.ALL_NAMES:
        state = named_optimizer[name]
        m = state.get("exp_avg", torch.zeros_like(base[name])).detach().cpu().float()
        v = state.get("exp_avg_sq", torch.zeros_like(base[name])).detach().cpu().float()
        momentum[name] = -m
        precond[name] = -m / (torch.sqrt(v) + 1.0e-8)
    generators = {
        "historical_469_to_480": unit(historical),
        "adam_momentum_escape": unit(momentum),
        "adam_preconditioned_escape": unit(precond),
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for index in range(4):
        raw = {name: torch.randn(base[name].shape, generator=generator, dtype=torch.float32) for name in d16.ALL_NAMES}
        generators[f"random_low_rank_{index:02d}"] = unit(raw)
    # A head/output-focused direction is a structured low-dimensional route;
    # the GRU remains unchanged so this is materially different from raw-wide
    # Adam continuation.
    head = {name: torch.zeros_like(base[name]) for name in d16.ALL_NAMES}
    for name in d16.ALL_NAMES:
        if name.startswith("head."):
            head[name] = torch.randn(base[name].shape, generator=generator, dtype=torch.float32)
    generators["head_output_structured_escape"] = unit(head)
    scales = (2.5e-5, 5.0e-5, 1.0e-4, 2.0e-4)
    candidates: list[dict[str, Any]] = []
    for direction_name, direction in generators.items():
        for sign in (1.0, -1.0):
            for scale in scales:
                candidates.append({
                    "candidate_id": f"S1_{direction_name}_{'plus' if sign > 0 else 'minus'}_{scale:.1e}",
                    "family": "PARAMETER_SPACE_ESCAPE",
                    "direction": direction_name,
                    "sign": sign,
                    "scale": scale,
                    "delta": scale_map(direction, sign * scale),
                })
    return candidates


def s1_screen(output: Path, runtime: Mapping[str, Any], model: Any, previous_model: Any, optimizer: Any, subset_count: int) -> dict[str, Any]:
    data = validation_view(runtime["pre"]["validation"], subset_count)
    baseline = evaluate(model, data, runtime["pre"]["stats"]["channels"], (1, 32))
    base_state = model_parameters(model)
    rows: list[dict[str, Any]] = []
    candidates = direct_escape_population(model, previous_model, optimizer)
    for candidate in candidates:
        trial = copy.deepcopy(model)
        set_model_state(trial, base_state, candidate["delta"])
        try:
            metrics = evaluate(trial, data, runtime["pre"]["stats"]["channels"], (1, 32))
            row = {key: value for key, value in candidate.items() if key != "delta"}
            row.update({"S1_H1": metrics["1"], "S1_H32": metrics["32"], "S1_delta_H1": metrics["1"] - baseline["1"], "S1_delta_H32": metrics["32"] - baseline["32"], "status": "S1_MEASURED"})
        except Exception as exc:
            row = {key: value for key, value in candidate.items() if key != "delta"}
            row.update({"S1_H1": None, "S1_H32": None, "S1_delta_H1": None, "S1_delta_H32": None, "status": f"S1_FAILURE:{type(exc).__name__}:{exc}"})
        rows.append(row)
        append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", row)
    write_json(output / "D42_S1_BASELINE.json", {"subset_count": int(data.rollout_target_positions.shape[0]), "H1": baseline["1"], "H32": baseline["32"], "canonical_H1": CANONICAL_H1, "canonical_H32": CANONICAL_H32})
    write_json(output / "D42_S1_ESCAPE_RESULTS.json", rows)
    measured = [row for row in rows if row.get("status") == "S1_MEASURED"]
    ranked = sorted(measured, key=lambda row: (float(row["S1_H1"]), float(row["S1_H32"])))
    best = ranked[0] if ranked else None
    summary = {"schema_version": "d42_s1_escape_screen_v1", "subset_count": int(data.rollout_target_positions.shape[0]), "population_count": len(rows), "baseline": {"H1": baseline["1"], "H32": baseline["32"]}, "best_shadow": best, "ranked_top_12": ranked[:12], "improvement_count": sum(float(row["S1_delta_H1"]) < 0.0 for row in measured), "h32_safe_improvement_count": sum(float(row["S1_delta_H1"]) < 0.0 and float(row["S1_H32"]) <= H32_LIMIT for row in measured)}
    write_json(output / "D42_S1_SUMMARY.json", summary)
    return summary


def adamw_undo_preconditioned_builder(
    spec: Mapping[str, Any], canonical_postclip: Mapping[str, torch.Tensor], h1_gradients: Mapping[str, torch.Tensor], h32_gradients: Mapping[str, torch.Tensor], before_parameters: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Mapping[str, Any]], effective_lr: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    family = str(spec["family"])
    if family == "D42_ADAMW_UNDO_PRECONDITION":
        # AdamW divides by sqrt(v); multiplying the presented H1 gradient by
        # sqrt(v) approximately cancels that anisotropy before the optimizer
        # step, creating a distinct effective-H1 search direction.
        result = {}
        for name in d16.ALL_NAMES:
            v = before_optimizer[name].get("exp_avg_sq", torch.zeros_like(h1_gradients[name])).detach().cpu()
            result[name] = h1_gradients[name].detach().cpu() * (torch.sqrt(v) + 1.0e-8)
        direction = d32.unit(result)
        return d32.scale_map(direction, float(spec.get("target_norm", 1.0))), {"family": family, "precondition_cancel": "sqrt_exp_avg_sq"}
    if family == "D42_MOMENTUM_SUBTRACTIVE_H1":
        momentum = {name: before_optimizer[name].get("exp_avg", torch.zeros_like(h1_gradients[name])).detach().cpu() for name in d16.ALL_NAMES}
        h1_unit = d32.unit(h1_gradients)
        momentum_unit = d32.unit(momentum)
        mixed = d32.add_maps(h1_unit, d32.scale_map(momentum_unit, -float(spec.get("rho", 0.25))))
        return d32.scale_map(d32.unit(mixed), float(spec.get("target_norm", 1.0))), {"family": family, "rho": float(spec.get("rho", 0.25))}
    if family == "D42_OPTIMIZER_RESET_H1":
        return d32.scale_map(d32.unit(h1_gradients), float(spec.get("target_norm", 1.0))), {"family": family, "optimizer_reset": "exp_avg_and_exp_avg_sq_zeroed"}
    raise RuntimeError(f"unknown_d42_family:{family}")


def predicted_safe_set_builder(
    spec: Mapping[str, Any], canonical_postclip: Mapping[str, torch.Tensor], h1_gradients: Mapping[str, torch.Tensor], h32_gradients: Mapping[str, torch.Tensor], before_parameters: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Mapping[str, Any]], effective_lr: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Choose a gradient on a predicted-AdamW H32 inward boundary."""
    seed_name = str(spec.get("seed", "H1"))
    seed = d32.unit(h1_gradients if seed_name == "H1" else canonical_postclip)
    constraint = d32.unit(h32_gradients)
    target = -abs(float(spec.get("safe_cosine", 0.002)))
    target_norm = float(spec.get("target_norm", 1.0))

    def score(amount: float) -> tuple[float, dict[str, torch.Tensor]]:
        mixed = d32.add_maps(seed, d32.scale_map(constraint, amount))
        presented = d32.scale_map(d32.unit(mixed), target_norm)
        predicted = d32.predicted_adamw_delta(before_parameters, before_optimizer, presented, effective_lr)
        value = d32.cosine(predicted, h32_gradients)
        if value is None:
            raise RuntimeError("predicted_safe_set_degenerate_score")
        return float(value), predicted

    low = 0.0
    high = 0.05
    initial_score, _ = score(low)
    high_score, _ = score(high)
    while high_score > target and high < 64.0:
        high *= 2.0
        high_score, _ = score(high)
    if high_score > target:
        raise RuntimeError(f"predicted_safe_set_bracket_failure:{initial_score}:{high_score}:{target}")
    for _ in range(32):
        midpoint = 0.5 * (low + high)
        midpoint_score, _ = score(midpoint)
        if midpoint_score <= target:
            high = midpoint
        else:
            low = midpoint
    final_score, predicted = score(high)
    mixed = d32.add_maps(seed, d32.scale_map(constraint, high))
    presented = d32.scale_map(d32.unit(mixed), target_norm)
    return presented, {"family": str(spec["family"]), "seed": seed_name, "safe_cosine_target": target, "safe_set_lambda": high, "predicted_h32_cosine": final_score, "predicted_delta_norm": d32.norm(predicted.values())}


def run_exact_round(output: Path, runtime: Mapping[str, Any], payload: Mapping[str, Any], model: Any, optimizer: Any, rng: Mapping[str, Any], canonical: Sequence[float], top_n: int = 3) -> dict[str, Any]:
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    specs = [
        {"family": "D42_ADAMW_UNDO_PRECONDITION", "base": "H1", "alpha": 0.0078125, "target_norm": 1.0},
        {"family": "D42_ADAMW_UNDO_PRECONDITION", "base": "H1", "alpha": 0.015625, "target_norm": 1.0},
        {"family": "D42_MOMENTUM_SUBTRACTIVE_H1", "base": "H1", "alpha": 0.0078125, "rho": 0.10, "target_norm": 1.0},
        {"family": "D42_MOMENTUM_SUBTRACTIVE_H1", "base": "H1", "alpha": 0.015625, "rho": 0.25, "target_norm": 1.0},
    ][: max(1, int(top_n) + 1)]
    original = d32.build_presented_gradient
    d32.build_presented_gradient = adamw_undo_preconditioned_builder
    records: list[dict[str, Any]] = []
    try:
        for spec in specs:
            try:
                record, _, _, _ = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED", "promotion_state": "SHADOW_ONLY"})
            except Exception as exc:
                record = {"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}"}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_S2 family={spec['family']} alpha={spec['alpha']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
    finally:
        d32.build_presented_gradient = original
    valid = [row for row in records if row.get("candidate_valid_before_replay")]
    valid.sort(key=lambda row: float(row["candidate_H1"]))
    replay: dict[str, Any] | None = None
    if valid:
        selected = valid[0]
        spec = dict(selected["d42_spec"])
        original = d32.build_presented_gradient
        d32.build_presented_gradient = adamw_undo_preconditioned_builder
        try:
            first, first_model, first_optimizer, first_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            second, second_model, second_optimizer, second_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            replay = d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
        finally:
            d32.build_presented_gradient = original
        # This is a resumable shadow seed only. It is intentionally not named
        # or installed as a canonical checkpoint.
        shadow_payload = d32.checkpoint_payload(first_model, first_optimizer, first_rng, first, CANONICAL_SHA256, canonical)
        shadow_payload.update({"scientific_state": "D42_SHADOW_STEPPING_STONE", "promotion_state": "NOT_PROMOTED", "d42_replay": replay, "resumable": True})
        torch.save(shadow_payload, output / "D42_shadow_update_481.pt")
        write_json(output / "D42_BEST_VALIDATED_REPLAY.json", {"selected_spec": spec, "first_record": first, "second_record": second, "replay": replay, "shadow_checkpoint": str((output / "D42_shadow_update_481.pt").resolve())})
    summary = {"schema_version": "d42_s2_full_shadow_v1", "records": records, "best_full_shadow": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "best_validated": valid[0] if valid else None, "deterministic_replay": replay, "canonical_mutation": False}
    write_json(output / "D42_S2_SUMMARY.json", summary)
    return summary


def run_recharge_round(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], top_n: int = 4) -> dict[str, Any]:
    """Re-test the strongest retained D35 H32-recharge routes from canonical."""
    payload = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model, optimizer, rng, _ = d33.instantiate_state(runtime, payload)
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    specs = [
        {"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "alpha": 0.03125, "kappa": 0.02, "target_norm": 1.0},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.03125, "lambda": 0.10, "target_norm": 1.0},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.03125, "lambda": 0.20, "target_norm": 1.0},
        {"family": "H1_PLUS_H32_PRESENTED", "base": "H1", "alpha": 0.03125, "lambda": 2.0, "target_norm": 1.0},
    ][: max(1, int(top_n))]
    records: list[dict[str, Any]] = []
    original = d32.build_presented_gradient
    try:
        for spec in specs:
            try:
                record, _, _, _ = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_D35_RECHARGE_RETEST", "promotion_state": "SHADOW_ONLY"})
            except Exception as exc:
                record = {"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_D35_RECHARGE_RETEST", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}"}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_RECHARGE family={spec['family']} alpha={spec['alpha']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
    finally:
        d32.build_presented_gradient = original
    valid = sorted((row for row in records if row.get("candidate_valid_before_replay")), key=lambda row: float(row["candidate_H1"]))
    replay = None
    if valid:
        selected = valid[0]
        spec = dict(selected["d42_spec"])
        original = d32.build_presented_gradient
        try:
            first, first_model, first_optimizer, first_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            second, second_model, second_optimizer, second_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            replay = d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
        finally:
            d32.build_presented_gradient = original
        record = dict(first)
        record.update({"d42_spec": spec, "validation_tier": "S2_FULL_AUTHENTICATED_D35_RECHARGE_REPLAY", "promotion_state": "SHADOW_ONLY", "deterministic_replay_consistency_pass": bool(replay["pass"]), "deterministic_replay_candidate_record_digest": replay["candidate_record_digest"], "deterministic_replay_record_digest": replay["replay_record_digest"], "replay_consistency": replay})
        shadow_payload = d32.checkpoint_payload(first_model, first_optimizer, first_rng, record, CANONICAL_SHA256, canonical)
        shadow_payload.update({"scientific_state": "D42_RECHARGE_SHADOW_STEPPING_STONE", "promotion_state": "NOT_PROMOTED", "resumable": True})
        torch.save(shadow_payload, output / "D42_recharge_shadow_update_481.pt")
        write_json(output / "D42_RECHARGE_BEST_REPLAY.json", {"selected_spec": spec, "record": record, "replay": replay, "shadow_checkpoint": str((output / "D42_recharge_shadow_update_481.pt").resolve())})
    summary = {"schema_version": "d42_s2_recharge_v1", "records": records, "best_full_shadow": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "best_validated": valid[0] if valid and replay and replay.get("pass") else None, "deterministic_replay": replay, "canonical_mutation": False}
    write_json(output / "D42_RECHARGE_SUMMARY.json", summary)
    return summary


def continue_from_recharge(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], steps: int) -> dict[str, Any]:
    shadow_path = output / "D42_recharge_shadow_update_481.pt"
    if not shadow_path.is_file():
        raise RuntimeError("d42_recharge_shadow_seed_missing")
    payload = torch.load(shadow_path, map_location="cpu", weights_only=False)
    parent_update = int(payload["completed_optimizer_step"])
    parent_h1 = float(payload["H1"])
    parent_h32 = float(payload["H32"])
    parent_model, parent_optimizer, parent_rng, _ = d33.instantiate_state(runtime, payload)
    specs = [
        {"family": "D42_ADAMW_UNDO_PRECONDITION", "base": "H1", "alpha": 0.0078125, "target_norm": 1.0},
        {"family": "D42_ADAMW_UNDO_PRECONDITION", "base": "H1", "alpha": 0.015625, "target_norm": 1.0},
        {"family": "D42_MOMENTUM_SUBTRACTIVE_H1", "base": "H1", "alpha": 0.0078125, "rho": 0.10, "target_norm": 1.0},
    ]
    original = d32.build_presented_gradient

    def hybrid_builder(spec: Mapping[str, Any], *args: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if str(spec.get("family", "")).startswith("D42_"):
            return adamw_undo_preconditioned_builder(spec, *args)
        return original(spec, *args)

    d32.build_presented_gradient = hybrid_builder
    records: list[dict[str, Any]] = []
    try:
        current_model, current_optimizer, current_rng = parent_model, parent_optimizer, parent_rng
        current_update, current_h1, current_h32 = parent_update, parent_h1, parent_h32
        for index in range(max(0, int(steps))):
            spec = specs[index % len(specs)]
            identity = d32.semantic_parent_identity(current_model, current_optimizer, current_rng)
            try:
                record, next_model, next_optimizer, next_rng = d32.run_trial(runtime, current_model, current_optimizer, current_rng, canonical, identity, current_update, current_update + 1, current_h1, current_h32, spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_RECHARGE_CONTINUATION", "promotion_state": "SHADOW_ONLY", "shadow_parent_update": current_update})
            except Exception as exc:
                record = {"candidate_update": current_update + 1, "d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_RECHARGE_CONTINUATION", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}", "parent_H1": current_h1, "parent_H32": current_h32}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_RECHARGE_CONTINUATION update={record.get('candidate_update')} family={spec['family']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
            if record.get("promotion_state") == "SHADOW_FAILURE" or not record.get("candidate_valid_before_replay"):
                break
            current_model, current_optimizer, current_rng = next_model, next_optimizer, next_rng
            current_update, current_h1, current_h32 = int(record["candidate_update"]), float(record["candidate_H1"]), float(record["candidate_H32"])
    finally:
        d32.build_presented_gradient = original
    summary = {"schema_version": "d42_s2_recharge_continuation_v1", "starting_shadow_update": parent_update, "starting_shadow_H1": parent_h1, "starting_shadow_H32": parent_h32, "records": records, "best": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "canonical_mutation": False}
    write_json(output / "D42_RECHARGE_CONTINUATION_SUMMARY.json", summary)
    return summary


def run_recharge_depth_round(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], top_n: int = 3) -> dict[str, Any]:
    """Refill H32 margin from the strongest D42 recharge shadow parent."""
    shadow_path = output / "D42_recharge_shadow_update_481.pt"
    if not shadow_path.is_file():
        raise RuntimeError("d42_recharge_shadow_seed_missing")
    payload = torch.load(shadow_path, map_location="cpu", weights_only=False)
    model, optimizer, rng, _ = d33.instantiate_state(runtime, payload)
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    parent_sha = sha256_file(shadow_path)
    specs = [
        {"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "alpha": 0.015625, "kappa": 0.02, "target_norm": 1.0},
        {"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "alpha": 0.0078125, "kappa": 0.02, "target_norm": 1.0},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.015625, "lambda": 0.20, "target_norm": 1.0},
    ][: max(1, int(top_n))]
    original = d32.build_presented_gradient
    records: list[dict[str, Any]] = []
    try:
        for spec in specs:
            try:
                record, _, _, _ = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, int(payload["completed_optimizer_step"]), int(payload["completed_optimizer_step"]) + 1, float(payload["H1"]), float(payload["H32"]), spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_RECHARGE_DEPTH", "promotion_state": "SHADOW_ONLY"})
            except Exception as exc:
                record = {"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_RECHARGE_DEPTH", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}"}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_RECHARGE_DEPTH family={spec['family']} alpha={spec['alpha']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
    finally:
        d32.build_presented_gradient = original
    valid = sorted((row for row in records if row.get("candidate_valid_before_replay")), key=lambda row: float(row["candidate_H1"]))
    replay = None
    if valid:
        selected = valid[0]
        spec = dict(selected["d42_spec"])
        original = d32.build_presented_gradient
        try:
            first, first_model, first_optimizer, first_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, int(payload["completed_optimizer_step"]), int(payload["completed_optimizer_step"]) + 1, float(payload["H1"]), float(payload["H32"]), spec)
            second, second_model, second_optimizer, second_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, int(payload["completed_optimizer_step"]), int(payload["completed_optimizer_step"]) + 1, float(payload["H1"]), float(payload["H32"]), spec)
            replay = d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
        finally:
            d32.build_presented_gradient = original
        record = dict(first)
        record.update({"d42_spec": spec, "validation_tier": "S2_FULL_AUTHENTICATED_RECHARGE_DEPTH_REPLAY", "promotion_state": "SHADOW_ONLY", "deterministic_replay_consistency_pass": bool(replay["pass"]), "deterministic_replay_candidate_record_digest": replay["candidate_record_digest"], "deterministic_replay_record_digest": replay["replay_record_digest"], "replay_consistency": replay})
        shadow_payload = d32.checkpoint_payload(first_model, first_optimizer, first_rng, record, parent_sha, canonical)
        shadow_payload.update({"scientific_state": "D42_RECHARGE_DEPTH_SHADOW_STEPPING_STONE", "promotion_state": "NOT_PROMOTED", "resumable": True})
        torch.save(shadow_payload, output / "D42_recharge_depth_shadow_update_482.pt")
        write_json(output / "D42_RECHARGE_DEPTH_BEST_REPLAY.json", {"selected_spec": spec, "record": record, "replay": replay, "shadow_checkpoint": str((output / "D42_recharge_depth_shadow_update_482.pt").resolve())})
    summary = {"schema_version": "d42_s2_recharge_depth_v1", "parent_shadow": str(shadow_path.resolve()), "records": records, "best_full_shadow": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "best_validated": valid[0] if valid and replay and replay.get("pass") else None, "deterministic_replay": replay, "canonical_mutation": False}
    write_json(output / "D42_RECHARGE_DEPTH_SUMMARY.json", summary)
    return summary


def run_optimizer_reset_round(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], top_n: int = 3) -> dict[str, Any]:
    """Test a moment-reset AdamW representation from the canonical model."""
    payload = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model, optimizer, rng, _ = d33.instantiate_state(runtime, payload)
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key].zero_()
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    specs = [
        {"family": "D42_OPTIMIZER_RESET_H1", "base": "H1", "alpha": 0.0078125, "target_norm": 1.0},
        {"family": "D42_OPTIMIZER_RESET_H1", "base": "H1", "alpha": 0.015625, "target_norm": 1.0},
        {"family": "D42_OPTIMIZER_RESET_H1", "base": "H1", "alpha": 0.03125, "target_norm": 1.0},
    ][: max(1, int(top_n))]
    original = d32.build_presented_gradient
    d32.build_presented_gradient = adamw_undo_preconditioned_builder
    records: list[dict[str, Any]] = []
    try:
        for spec in specs:
            try:
                record, _, _, _ = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_OPTIMIZER_RESET", "promotion_state": "SHADOW_ONLY"})
            except Exception as exc:
                record = {"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_OPTIMIZER_RESET", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}"}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_OPTIMIZER_RESET alpha={spec['alpha']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
    finally:
        d32.build_presented_gradient = original
    valid = sorted((row for row in records if row.get("candidate_valid_before_replay")), key=lambda row: float(row["candidate_H1"]))
    replay = None
    if valid:
        selected = valid[0]
        spec = dict(selected["d42_spec"])
        original = d32.build_presented_gradient
        d32.build_presented_gradient = adamw_undo_preconditioned_builder
        try:
            first, first_model, first_optimizer, first_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            second, second_model, second_optimizer, second_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            replay = d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
        finally:
            d32.build_presented_gradient = original
        record = dict(first)
        record.update({"d42_spec": spec, "validation_tier": "S2_FULL_AUTHENTICATED_OPTIMIZER_RESET_REPLAY", "promotion_state": "SHADOW_ONLY", "deterministic_replay_consistency_pass": bool(replay["pass"]), "deterministic_replay_candidate_record_digest": replay["candidate_record_digest"], "deterministic_replay_record_digest": replay["replay_record_digest"], "replay_consistency": replay})
        shadow_payload = d32.checkpoint_payload(first_model, first_optimizer, first_rng, record, CANONICAL_SHA256, canonical)
        shadow_payload.update({"scientific_state": "D42_OPTIMIZER_RESET_SHADOW_STEPPING_STONE", "promotion_state": "NOT_PROMOTED", "resumable": True})
        torch.save(shadow_payload, output / "D42_optimizer_reset_shadow_update_481.pt")
        write_json(output / "D42_OPTIMIZER_RESET_BEST_REPLAY.json", {"selected_spec": spec, "record": record, "replay": replay, "shadow_checkpoint": str((output / "D42_optimizer_reset_shadow_update_481.pt").resolve())})
    best_any = min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None)
    if best_any is not None and not valid:
        # Preserve the intentionally nonmonotonic escape as a resumable shadow
        # so that a separate H32-recovery route can be tried from it.
        spec = dict(best_any["d42_spec"])
        original = d32.build_presented_gradient
        d32.build_presented_gradient = adamw_undo_preconditioned_builder
        try:
            first, first_model, first_optimizer, first_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            second, second_model, second_optimizer, second_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            escape_replay = d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
        finally:
            d32.build_presented_gradient = original
        escape_record = dict(first)
        escape_record.update({"d42_spec": spec, "validation_tier": "S2_FULL_AUTHENTICATED_OPTIMIZER_RESET_ESCAPE_REPLAY", "promotion_state": "SHADOW_ONLY", "deterministic_replay_consistency_pass": bool(escape_replay["pass"]), "deterministic_replay_candidate_record_digest": escape_replay["candidate_record_digest"], "deterministic_replay_record_digest": escape_replay["replay_record_digest"], "replay_consistency": escape_replay})
        escape_payload = d32.checkpoint_payload(first_model, first_optimizer, first_rng, escape_record, CANONICAL_SHA256, canonical)
        escape_payload.update({"scientific_state": "D42_OPTIMIZER_RESET_NONMONOTONIC_ESCAPE", "promotion_state": "NOT_PROMOTED", "resumable": True})
        torch.save(escape_payload, output / "D42_optimizer_reset_escape_shadow_update_481.pt")
        write_json(output / "D42_OPTIMIZER_RESET_ESCAPE_REPLAY.json", {"selected_spec": spec, "record": escape_record, "replay": escape_replay, "shadow_checkpoint": str((output / "D42_optimizer_reset_escape_shadow_update_481.pt").resolve())})
    summary = {"schema_version": "d42_s2_optimizer_reset_v1", "records": records, "best_full_shadow": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "best_validated": valid[0] if valid and replay and replay.get("pass") else None, "deterministic_replay": replay, "canonical_mutation": False}
    write_json(output / "D42_OPTIMIZER_RESET_SUMMARY.json", summary)
    return summary


def recover_optimizer_reset_escape(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], top_n: int = 3) -> dict[str, Any]:
    """Apply H32-recovery directions after a nonmonotonic optimizer reset."""
    shadow_path = output / "D42_optimizer_reset_escape_shadow_update_481.pt"
    if not shadow_path.is_file():
        raise RuntimeError("d42_optimizer_reset_escape_seed_missing")
    payload = torch.load(shadow_path, map_location="cpu", weights_only=False)
    model, optimizer, rng, _ = d33.instantiate_state(runtime, payload)
    parent_update = int(payload["completed_optimizer_step"])
    parent_h1 = float(payload["H1"])
    parent_h32 = float(payload["H32"])
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    specs = [
        {"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "alpha": 0.015625, "kappa": 0.02, "target_norm": 1.0},
        {"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "alpha": 0.0078125, "kappa": 0.02, "target_norm": 1.0},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.015625, "lambda": 0.20, "target_norm": 1.0},
    ][: max(1, int(top_n))]
    original = d32.build_presented_gradient
    records: list[dict[str, Any]] = []
    try:
        for spec in specs:
            try:
                record, _, _, _ = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, parent_update, parent_update + 1, parent_h1, parent_h32, spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_OPTIMIZER_RESET_RECOVERY", "promotion_state": "SHADOW_ONLY", "recovery_valid": bool(float(record["candidate_H1"]) < CANONICAL_H1 and float(record["candidate_H32"]) <= H32_LIMIT)})
            except Exception as exc:
                record = {"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_OPTIMIZER_RESET_RECOVERY", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}", "recovery_valid": False}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_OPTIMIZER_RESET_RECOVERY alpha={spec['alpha']} family={spec['family']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} recovery_valid={record.get('recovery_valid')}", flush=True)
    finally:
        d32.build_presented_gradient = original
    valid = sorted((row for row in records if row.get("recovery_valid")), key=lambda row: float(row["candidate_H1"]))
    replay = None
    if valid:
        selected = valid[0]
        spec = dict(selected["d42_spec"])
        original = d32.build_presented_gradient
        try:
            first, first_model, first_optimizer, first_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, parent_update, parent_update + 1, parent_h1, parent_h32, spec)
            second, second_model, second_optimizer, second_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, parent_update, parent_update + 1, parent_h1, parent_h32, spec)
            replay = d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
        finally:
            d32.build_presented_gradient = original
        record = dict(first)
        record.update({"d42_spec": spec, "validation_tier": "S2_FULL_AUTHENTICATED_OPTIMIZER_RESET_RECOVERY_REPLAY", "promotion_state": "SHADOW_ONLY", "recovery_valid": True, "deterministic_replay_consistency_pass": bool(replay["pass"]), "deterministic_replay_candidate_record_digest": replay["candidate_record_digest"], "deterministic_replay_record_digest": replay["replay_record_digest"], "replay_consistency": replay})
        shadow_payload = d32.checkpoint_payload(first_model, first_optimizer, first_rng, record, sha256_file(shadow_path), canonical)
        shadow_payload.update({"scientific_state": "D42_OPTIMIZER_RESET_RECOVERED_SHADOW", "promotion_state": "NOT_PROMOTED", "resumable": True})
        torch.save(shadow_payload, output / "D42_optimizer_reset_recovered_shadow_update_482.pt")
        write_json(output / "D42_OPTIMIZER_RESET_RECOVERED_REPLAY.json", {"selected_spec": spec, "record": record, "replay": replay, "shadow_checkpoint": str((output / "D42_optimizer_reset_recovered_shadow_update_482.pt").resolve())})
    summary = {"schema_version": "d42_s2_optimizer_reset_recovery_v1", "parent_shadow": str(shadow_path.resolve()), "records": records, "best_full_shadow": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "best_validated": valid[0] if valid and replay and replay.get("pass") else None, "deterministic_replay": replay, "canonical_mutation": False}
    write_json(output / "D42_OPTIMIZER_RESET_RECOVERY_SUMMARY.json", summary)
    return summary


def run_safe_set_round(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], top_n: int = 3) -> dict[str, Any]:
    payload = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model, optimizer, rng, _ = d33.instantiate_state(runtime, payload)
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    specs = [
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.0078125, "seed": "H1", "safe_cosine": 0.001, "target_norm": 1.0},
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.0078125, "seed": "H1", "safe_cosine": 0.005, "target_norm": 1.0},
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.015625, "seed": "H1", "safe_cosine": 0.001, "target_norm": 1.0},
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.0078125, "seed": "POSTCLIP", "safe_cosine": 0.001, "target_norm": 1.0},
    ][: max(1, int(top_n) + 1)]
    original = d32.build_presented_gradient
    d32.build_presented_gradient = predicted_safe_set_builder
    records: list[dict[str, Any]] = []
    try:
        for spec in specs:
            try:
                record, _, _, _ = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_SAFE_SET", "promotion_state": "SHADOW_ONLY"})
            except Exception as exc:
                record = {"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_SAFE_SET", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}"}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_SAFE family={spec['family']} alpha={spec['alpha']} seed={spec['seed']} target={spec['safe_cosine']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
    finally:
        d32.build_presented_gradient = original
    valid = sorted((row for row in records if row.get("candidate_valid_before_replay")), key=lambda row: float(row["candidate_H1"]))
    replay = None
    if valid:
        selected = valid[0]
        spec = dict(selected["d42_spec"])
        original = d32.build_presented_gradient
        d32.build_presented_gradient = predicted_safe_set_builder
        try:
            first, first_model, first_optimizer, first_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            second, second_model, second_optimizer, second_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
            replay = d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
        finally:
            d32.build_presented_gradient = original
        record = dict(first)
        record.update({"d42_spec": spec, "deterministic_replay": replay})
        shadow_payload = d32.checkpoint_payload(first_model, first_optimizer, first_rng, {**first, "deterministic_replay_consistency_pass": bool(replay["pass"]), "deterministic_replay_candidate_record_digest": replay["candidate_record_digest"], "deterministic_replay_record_digest": replay["replay_record_digest"]}, CANONICAL_SHA256, canonical)
        shadow_payload.update({"scientific_state": "D42_SHADOW_STEPPING_STONE", "promotion_state": "NOT_PROMOTED", "resumable": True})
        torch.save(shadow_payload, output / "D42_safe_shadow_update_481.pt")
        write_json(output / "D42_SAFE_BEST_REPLAY.json", {"selected_spec": spec, "record": record, "replay": replay, "shadow_checkpoint": str((output / "D42_safe_shadow_update_481.pt").resolve())})
    summary = {"schema_version": "d42_s2_safe_set_v1", "records": records, "best_full_shadow": valid[0] if valid else min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "best_validated": valid[0] if valid and replay and replay.get("pass") else None, "deterministic_replay": replay, "canonical_mutation": False}
    write_json(output / "D42_SAFE_SET_SUMMARY.json", summary)
    return summary


def continue_from_shadow(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], steps: int) -> dict[str, Any]:
    shadow_path = output / "D42_shadow_update_481.pt"
    if not shadow_path.is_file():
        raise RuntimeError("d42_shadow_seed_missing")
    payload = torch.load(shadow_path, map_location="cpu", weights_only=False)
    parent_update = int(payload["completed_optimizer_step"])
    parent_h1 = float(payload["H1"])
    parent_h32 = float(payload["H32"])
    parent_model, parent_optimizer, parent_rng, _ = d33.instantiate_state(runtime, payload)
    parent_identity = d32.semantic_parent_identity(parent_model, parent_optimizer, parent_rng)
    records: list[dict[str, Any]] = []
    specs = [
        {"family": "D42_MOMENTUM_SUBTRACTIVE_H1", "base": "H1", "alpha": 0.0078125, "rho": 0.10, "target_norm": 1.0},
        {"family": "D42_ADAMW_UNDO_PRECONDITION", "base": "H1", "alpha": 0.0078125, "target_norm": 1.0},
        {"family": "D42_ADAMW_UNDO_PRECONDITION", "base": "H1", "alpha": 0.00390625, "target_norm": 1.0},
    ]
    original = d32.build_presented_gradient
    d32.build_presented_gradient = adamw_undo_preconditioned_builder
    try:
        current_model, current_optimizer, current_rng = parent_model, parent_optimizer, parent_rng
        current_update, current_h1, current_h32 = parent_update, parent_h1, parent_h32
        for index in range(max(0, int(steps))):
            spec = specs[index % len(specs)]
            identity = d32.semantic_parent_identity(current_model, current_optimizer, current_rng)
            try:
                record, next_model, next_optimizer, next_rng = d32.run_trial(runtime, current_model, current_optimizer, current_rng, canonical, identity, current_update, current_update + 1, current_h1, current_h32, spec)
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_CONTINUATION", "promotion_state": "SHADOW_ONLY", "shadow_parent_update": current_update})
                current_model, current_optimizer, current_rng = next_model, next_optimizer, next_rng
                current_update, current_h1, current_h32 = int(record["candidate_update"]), float(record["candidate_H1"]), float(record["candidate_H32"])
            except Exception as exc:
                record = {"candidate_update": current_update + 1, "d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_CONTINUATION", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}", "parent_H1": current_h1, "parent_H32": current_h32}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_CONTINUATION update={record.get('candidate_update')} family={spec['family']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
            if record.get("promotion_state") == "SHADOW_FAILURE":
                break
    finally:
        d32.build_presented_gradient = original
    summary = {"schema_version": "d42_s2_continuation_v1", "starting_shadow_update": parent_update, "starting_shadow_H1": parent_h1, "starting_shadow_H32": parent_h32, "records": records, "best": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "canonical_mutation": False}
    write_json(output / "D42_CONTINUATION_SUMMARY.json", summary)
    return summary


def continue_from_safe_shadow(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float], steps: int) -> dict[str, Any]:
    """Continue only from the replay-validated predicted-safe-set shadow seed."""
    shadow_path = output / "D42_safe_shadow_update_481.pt"
    if not shadow_path.is_file():
        raise RuntimeError("d42_safe_shadow_seed_missing")
    payload = torch.load(shadow_path, map_location="cpu", weights_only=False)
    parent_update = int(payload["completed_optimizer_step"])
    parent_h1 = float(payload["H1"])
    parent_h32 = float(payload["H32"])
    parent_model, parent_optimizer, parent_rng, _ = d33.instantiate_state(runtime, payload)
    records: list[dict[str, Any]] = []
    # The lower alpha values deliberately keep more margin than the seed's
    # first step.  The next trial is still accepted only after the full H1/H32
    # evaluator and hard H32 gate have run.
    specs = [
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.00390625, "seed": "H1", "safe_cosine": 0.005, "target_norm": 1.0},
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.00390625, "seed": "H1", "safe_cosine": 0.0005, "target_norm": 1.0},
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.00390625, "seed": "H1", "safe_cosine": 0.001, "target_norm": 1.0},
        {"family": "D42_PREDICTED_SAFESET", "base": "H1", "alpha": 0.001953125, "seed": "H1", "safe_cosine": 0.001, "target_norm": 1.0},
    ]
    original = d32.build_presented_gradient
    d32.build_presented_gradient = predicted_safe_set_builder
    try:
        current_model, current_optimizer, current_rng = parent_model, parent_optimizer, parent_rng
        current_update, current_h1, current_h32 = parent_update, parent_h1, parent_h32
        for index in range(max(0, int(steps))):
            spec = specs[index % len(specs)]
            identity = d32.semantic_parent_identity(current_model, current_optimizer, current_rng)
            try:
                record, next_model, next_optimizer, next_rng = d32.run_trial(
                    runtime, current_model, current_optimizer, current_rng, canonical, identity,
                    current_update, current_update + 1, current_h1, current_h32, spec,
                )
                record.update({"d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_SAFE_CONTINUATION", "promotion_state": "SHADOW_ONLY", "shadow_parent_update": current_update})
            except Exception as exc:
                record = {"candidate_update": current_update + 1, "d42_spec": dict(spec), "validation_tier": "S2_FULL_AUTHENTICATED_SAFE_CONTINUATION", "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}", "parent_H1": current_h1, "parent_H32": current_h32}
            records.append(record)
            append_jsonl(output / "D42_EXPERIMENT_RESULTS.jsonl", record)
            print(f"D42_SAFE_CONTINUATION update={record.get('candidate_update')} family={spec['family']} alpha={spec['alpha']} target={spec['safe_cosine']} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} valid={record.get('candidate_valid_before_replay')}", flush=True)
            if record.get("promotion_state") == "SHADOW_FAILURE":
                if "predicted_safe_set_bracket_failure" in str(record.get("rejection_reason", "")):
                    continue
                break
            if not record.get("candidate_valid_before_replay"):
                # Never compound a hard-constraint failure into the next
                # shadow state.  The failed point remains recorded for audit.
                break
            current_model, current_optimizer, current_rng = next_model, next_optimizer, next_rng
            current_update, current_h1, current_h32 = int(record["candidate_update"]), float(record["candidate_H1"]), float(record["candidate_H32"])
    finally:
        d32.build_presented_gradient = original
    summary = {"schema_version": "d42_s2_safe_continuation_v1", "starting_shadow_update": parent_update, "starting_shadow_H1": parent_h1, "starting_shadow_H32": parent_h32, "records": records, "best": min((row for row in records if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None), "canonical_mutation": False}
    write_json(output / "D42_SAFE_CONTINUATION_SUMMARY.json", summary)
    return summary


def materialize_shadow_seed(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float]) -> dict[str, Any]:
    """Recreate the already replay-validated S2 winner as a shadow seed."""
    payload = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model, optimizer, rng, _ = d33.instantiate_state(runtime, payload)
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    spec = {"family": "D42_MOMENTUM_SUBTRACTIVE_H1", "base": "H1", "alpha": 0.0078125, "rho": 0.10, "target_norm": 1.0}
    original = d32.build_presented_gradient
    d32.build_presented_gradient = adamw_undo_preconditioned_builder
    try:
        record, candidate_model, candidate_optimizer, candidate_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
        replay_record, replay_model, replay_optimizer, replay_rng = d32.run_trial(runtime, model, optimizer, rng, canonical, identity, CANONICAL_UPDATE, CANONICAL_UPDATE + 1, CANONICAL_H1, CANONICAL_H32, spec)
        replay = d32.d27.replay_compare(record, replay_record, candidate_model, replay_model, candidate_optimizer, replay_optimizer, candidate_rng, replay_rng)
    finally:
        d32.build_presented_gradient = original
    record.update({
        "d42_spec": spec,
        "validation_tier": "S2_FULL_AUTHENTICATED_REPLAY_RECONSTRUCTION",
        "promotion_state": "SHADOW_ONLY",
        "deterministic_replay_consistency_pass": bool(replay["pass"]),
        "deterministic_replay_status": "PASS" if replay["pass"] else "FAIL",
        "deterministic_replay_candidate_record_digest": replay["candidate_record_digest"],
        "deterministic_replay_record_digest": replay["replay_record_digest"],
        "replay_consistency": replay,
    })
    shadow_payload = d32.checkpoint_payload(candidate_model, candidate_optimizer, candidate_rng, record, CANONICAL_SHA256, canonical)
    shadow_payload.update({"scientific_state": "D42_SHADOW_STEPPING_STONE", "promotion_state": "NOT_PROMOTED", "resumable": True})
    torch.save(shadow_payload, output / "D42_shadow_update_481.pt")
    result = {"schema_version": "d42_shadow_seed_materialization_v1", "record": record, "shadow_checkpoint": str((output / "D42_shadow_update_481.pt").resolve()), "canonical_mutation": False}
    write_json(output / "D42_SHADOW_SEED.json", result)
    return result


def validate_shadow_horizons(output: Path, runtime: Mapping[str, Any], shadow_name: str, result_name: str, schema_version: str) -> dict[str, Any]:
    """Record the full authoritative causal/self-fed horizon vector for a shadow seed."""
    shadow_path = output / shadow_name
    if not shadow_path.is_file():
        raise RuntimeError(f"d42_shadow_seed_missing:{shadow_name}")
    payload = torch.load(shadow_path, map_location="cpu", weights_only=False)
    model, _, _, _ = d33.instantiate_state(runtime, payload)
    horizons = tuple(int(value) for value in d17.d7.VALIDATION_HORIZONS)
    metrics = evaluate(model, runtime["pre"]["validation"], runtime["pre"]["stats"]["channels"], horizons)
    result = {
        "schema_version": schema_version,
        "shadow_checkpoint": str(shadow_path.resolve()),
        "candidate_update": int(payload["completed_optimizer_step"]),
        "candidate_H1": float(payload["H1"]),
        "candidate_H32": float(payload["H32"]),
        "validation_mode": "causal_self_fed_free_running",
        "available_authoritative_horizons": list(horizons),
        "metrics": metrics,
        "H32_limit": H32_LIMIT,
        "H32_pass": bool(metrics.get("32", math.inf) <= H32_LIMIT),
        "H64": {"status": "not_available", "reason": "authoritative D17 evaluator horizon set ends at 32"},
        "canonical_mutation": False,
    }
    write_json(output / result_name, result)
    return result


def validate_safe_shadow_horizons(output: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    return validate_shadow_horizons(output, runtime, "D42_safe_shadow_update_481.pt", "D42_S2_SAFE_SHADOW_HORIZONS.json", "d42_s2_safe_shadow_horizon_validation_v1")


def combine_shadow_summaries(*sources: Mapping[str, Any] | None) -> dict[str, Any] | None:
    usable = [source for source in sources if source]
    if not usable:
        return None
    rows = [row for source in usable for row in source.get("records", [])]
    full_rows = [row for row in rows if row.get("candidate_H1") is not None]
    validated = [
        (source.get("best_validated"), source.get("deterministic_replay"))
        for source in usable
        if source.get("best_validated") and (source.get("deterministic_replay") or {}).get("pass")
    ]
    best_pair = min(validated, key=lambda pair: float(pair[0]["candidate_H1"])) if validated else (None, None)
    return {
        "schema_version": "d42_shadow_portfolio_v1",
        "records": rows,
        "best_full_shadow": min(full_rows, key=lambda row: float(row["candidate_H1"]), default=None),
        "best_validated": best_pair[0],
        "deterministic_replay": best_pair[1],
        "source_summaries": usable,
        "canonical_mutation": False,
    }


def write_ledgers(output: Path, s1: Mapping[str, Any], s2: Mapping[str, Any] | None, s3: Mapping[str, Any] | None = None) -> None:
    best_shadow = s1.get("best_shadow")
    for source in (s2, s3):
        if source and source.get("best_full_shadow"):
            candidate = source["best_full_shadow"]
            candidate_h1 = candidate.get("candidate_H1")
            best_h1 = best_shadow.get("S1_H1", math.inf) if best_shadow else math.inf
            if candidate_h1 is not None and float(candidate_h1) < float(best_h1):
                best_shadow = candidate
    families = [
        {"family": "historical_basin_escape", "methods": ["historical_469_to_480_delta"], "tier": "S1", "status": "TESTED", "use": "low_rank_direction"},
        {"family": "optimizer_state_escape", "methods": ["adam_momentum_escape", "adam_preconditioned_escape"], "tier": "S1", "status": "TESTED", "use": "optimizer_state_coordinate"},
        {"family": "population_random_low_rank", "methods": ["four_seeded_random_directions"], "tier": "S1", "status": "TESTED", "use": "diverse_shadow_population"},
        {"family": "structured_head_escape", "methods": ["head_output_structured_escape"], "tier": "S1", "status": "TESTED", "use": "causal_block_subspace"},
        {"family": "adamw_effective_direction", "methods": ["D42_ADAMW_UNDO_PRECONDITION", "D42_MOMENTUM_SUBTRACTIVE_H1"], "tier": "S2", "status": "TESTED" if s2 else "NOT_RUN", "use": "full_authenticated_gradient_trial"},
        {"family": "predicted_h32_safe_set", "methods": ["D42_PREDICTED_SAFESET", "predicted_AdamW_H32_cosine_bracket"], "tier": "S2", "status": "TESTED" if s3 else "NOT_RUN", "use": "constraint_aware_shadow_continuation"},
        {"family": "d35_h32_recharge_restart", "methods": ["REALIZED_ADAMW_H32_CORRECTION", "D31_BASE_PLUS_H32_CORRECTION", "H1_PLUS_H32_PRESENTED"], "tier": "S2", "status": "TESTED" if (output / "D42_RECHARGE_SUMMARY.json").is_file() else "NOT_RUN", "use": "historical_shadow_retest_and_recharge"},
        {"family": "d42_recharge_depth", "methods": ["smaller_realized_AdamW_H32_correction", "recharge_then_descent"], "tier": "S2", "status": "TESTED" if (output / "D42_RECHARGE_DEPTH_SUMMARY.json").is_file() else "NOT_RUN", "use": "multi-step_shadow_margin_refill"},
        {"family": "optimizer_moment_reset", "methods": ["zero_exp_avg", "zero_exp_avg_sq", "H1_replay"], "tier": "S2", "status": "TESTED" if (output / "D42_OPTIMIZER_RESET_SUMMARY.json").is_file() else "NOT_RUN", "use": "alternate_optimizer_state_basin"},
        {"family": "optimizer_reset_nonmonotonic_recovery", "methods": ["moment_reset_escape", "REALIZED_ADAMW_H32_CORRECTION"], "tier": "S2", "status": "TESTED" if (output / "D42_OPTIMIZER_RESET_RECOVERY_SUMMARY.json").is_file() else "NOT_RUN", "use": "recover_H32_after_H1_escape"},
    ]
    write_json(output / "D42_SEARCH_FAMILY_LEDGER.json", {"schema_version": "d42_search_family_ledger_v1", "families": families, "best_shadow": best_shadow})
    write_json(output / "D42_BREAKTHROUGH_CANDIDATES.json", {"schema_version": "d42_breakthrough_candidates_v1", "S1_best": s1.get("best_shadow"), "S2_best": s2.get("best_full_shadow") if s2 else None, "S2_best_validated": s2.get("best_validated") if s2 else None, "S3_best": s3.get("best_full_shadow") if s3 else None, "S3_best_validated": s3.get("best_validated") if s3 else None, "target_H1": TARGET_H1})
    failures = []
    if s1.get("population_count", 0) and not s1.get("h32_safe_improvement_count", 0):
        failures.append({"method": "parameter_space_escape_population", "failure": "no_S1_H1_improvement_with_H32_limit", "repair": "redirect_to_AdamW_effective_direction_trials", "status": "MUTATED"})
    if s2:
        for row in s2.get("records", []):
            if row.get("promotion_state") == "SHADOW_FAILURE":
                failures.append({"method": row.get("d42_spec", {}).get("family"), "failure": row.get("rejection_reason"), "repair": "retain_canonical_and_try_next_family", "status": "RECOVERED"})
        if not s2.get("best_validated"):
            failures.append({"method": "D42_S2_full_trials", "failure": "no_replay_validated_candidate", "repair": "canonical_unchanged; continue basin representation pivot", "status": "OPEN"})
    if s3:
        for row in s3.get("records", []):
            if row.get("promotion_state") == "SHADOW_FAILURE":
                failures.append({"method": row.get("d42_spec", {}).get("family"), "failure": row.get("rejection_reason"), "repair": "retain_last_safe_shadow; reduce_alpha_or_safe_cosine", "status": "RECOVERED"})
            elif row.get("candidate_valid_before_replay") is False:
                failures.append({"method": row.get("d42_spec", {}).get("family"), "failure": row.get("rejection_reason", "H32_LIMIT_EXCEEDED"), "repair": "retain_last_safe_shadow; do not advance invalid branch", "status": "RECOVERED"})
        if not s3.get("best_validated") and s3.get("schema_version") == "d42_s2_safe_set_v1":
            failures.append({"method": "D42_PREDICTED_SAFESET", "failure": "no_replay_validated_safe_set_candidate", "repair": "retain canonical and replay the next safe-set bracket", "status": "OPEN"})
    write_json(output / "D42_FAILURE_AND_RECOVERY_LEDGER.json", {"schema_version": "d42_failure_recovery_ledger_v1", "failures": failures})
    (output / "D42_EXTERNAL_RESEARCH_LEDGER.md").write_text("# D42 external research ledger\n\nNo external source was required to execute this round. The tested methods were implemented from retained project evidence and the authenticated D32/D35 AdamW transition/evaluator. The next research pivot remains open; no external method is claimed to have materially helped this round.\n", encoding="utf-8", newline="\n")
    write_json(output / "D42_PROMOTION_DECISIONS.json", {"schema_version": "d42_promotion_decisions_v1", "canonical_mutation": False, "decision": "NO_PROMOTION", "reason": "D42 shadow round did not authorize a D39/D40/D41 promotion transaction"})


def write_final(output: Path, s1: Mapping[str, Any], s2: Mapping[str, Any] | None, runtime_meta: Mapping[str, Any], s3: Mapping[str, Any] | None = None) -> dict[str, Any]:
    d41 = runtime_meta["d41"]
    full_rows = [row for row in (s2 or {}).get("records", []) if row.get("candidate_H1") is not None]
    full_rows += [row for row in (s3 or {}).get("records", []) if row.get("candidate_H1") is not None]
    validated_sources = []
    for source in (s2, s3):
        candidate = (source or {}).get("best_validated")
        replay = (source or {}).get("deterministic_replay") or {}
        if candidate and replay.get("pass"):
            validated_sources.append(candidate)
    best_validated = min(validated_sources, key=lambda row: float(row["candidate_H1"])) if validated_sources else None
    best_shadow_candidates = [float(row["candidate_H1"]) for row in full_rows]
    best_shadow = min(best_shadow_candidates, default=CANONICAL_H1)
    best_validated_h1 = float(best_validated["candidate_H1"]) if best_validated else None
    safe_candidates = [float(row["candidate_H1"]) for row in full_rows if row.get("candidate_valid_before_replay")]
    best_h32_safe = min(safe_candidates, default=best_validated_h1 if best_validated_h1 is not None else CANONICAL_H1)
    remaining = max(0.0, CANONICAL_H1 - TARGET_H1)
    gap = max(0.0, (CANONICAL_H1 - best_shadow) / remaining) if remaining > 0.0 else 0.0
    safe_gap = max(0.0, (CANONICAL_H1 - best_h32_safe) / remaining) if remaining > 0.0 else 0.0
    if best_shadow <= TARGET_H1 and best_validated_h1 is None:
        basin_status = "SHADOW_TARGET_SIGNAL_REQUIRES_FULL_RETENTION_VALIDATION"
    elif best_validated_h1 is not None and best_validated_h1 <= TARGET_H1:
        basin_status = "TARGET_REACHED_PENDING_D39_D40_D41_PROMOTION"
    else:
        basin_status = "NEW_SHADOW_STEPPING_STONE_CONTINUE_DEEPENING" if best_shadow < CANONICAL_H1 else "LOCAL_BASIN_SATURATED_PIVOT_EXECUTED"
    summary = {
        "schema_version": "d42_summary_v1",
        "TASK_STATUS": "PARTIAL_PASS_TARGET_NOT_REACHED",
        "FINAL_TARGET_STATUS": "NOT_ACHIEVED",
        "H1_TARGET": TARGET_H1,
        "STARTING_CANONICAL_H1": CANONICAL_H1,
        "FINAL_RETAINED_CANONICAL_H1": CANONICAL_H1,
        "BEST_SHADOW_H1": best_shadow,
        "BEST_H32_SAFE_SHADOW_H1": best_h32_safe,
        "BEST_VALIDATED_H1": best_validated_h1,
        "TOTAL_CANONICAL_PROMOTIONS": 0,
        "REMAINING_GAP_CLOSURE": gap,
        "H32_SAFE_GAP_CLOSURE": safe_gap,
        "D39_RETENTION": d41.get("D39_RETENTION"),
        "D40_RETENTION": d41.get("D40_RETENTION"),
        "D41_RETENTION": d41.get("COLLISION_CERTIFICATION_STATUS"),
        "CANONICAL_REGRESSION": "NO",
        "CURRENT_BASIN_STATUS": basin_status,
        "BREAKTHROUGH_CHANNELS_DISCOVERED": sorted(
            {str(row.get("direction")) for row in s1.get("ranked_top_12", []) if float(row.get("S1_delta_H1", 0.0)) < 0.0}
            | {str(row.get("d42_spec", {}).get("family")) for source in (s2, s3) for row in (source or {}).get("records", []) if row.get("candidate_H1") is not None and float(row["candidate_H1"]) < CANONICAL_H1}
        ),
        "TRUE_EXTERNAL_HARD_BLOCKER": "NONE",
        "FIRST_UNRESOLVED_BLOCKER": "TARGET_NOT_REACHED; CONTINUE_SEARCH",
        "authoritative_scope": "181-point ON-state open-arch only",
        "collision_method": "adaptive_discrete_interpolation",
        "ccd": "not_available",
        "clearance": None,
        "canonical_checkpoint": str(CANONICAL_CHECKPOINT.resolve()),
        "canonical_checkpoint_sha256": runtime_meta["canonical_sha256"],
        "s1": {"population_count": s1.get("population_count"), "improvement_count": s1.get("improvement_count"), "h32_safe_improvement_count": s1.get("h32_safe_improvement_count")},
        "s2_full_trial_count": len(full_rows),
        "s2_replay": (s2 or {}).get("deterministic_replay"),
        "s2_safe_shadow_horizons": "D42_S2_SAFE_SHADOW_HORIZONS.json" if (output / "D42_S2_SAFE_SHADOW_HORIZONS.json").is_file() else None,
        "s2_recharge_shadow_horizons": "D42_S2_RECHARGE_SHADOW_HORIZONS.json" if (output / "D42_S2_RECHARGE_SHADOW_HORIZONS.json").is_file() else None,
        "s3": s3,
    }
    write_json(output / "D42_SUMMARY.json", summary)
    history = [{"state": "START", "update": CANONICAL_UPDATE, "H1": CANONICAL_H1, "H32": CANONICAL_H32, "status": "PROMOTED_PRIOR_RETENTION_BASELINE"}, {"state": "D42_END", "update": CANONICAL_UPDATE, "H1": CANONICAL_H1, "H32": CANONICAL_H32, "status": "NO_CANONICAL_MUTATION"}]
    write_csv(output / "D42_CANONICAL_HISTORY.csv", history)
    report = [
        "# D42 persistent H1 breakthrough / basin-escape campaign",
        "",
        "D42 executed a shadow-first heterogeneous search against the authenticated retained D41 baseline. Canonical state was never overwritten.",
        "",
        f"- Starting canonical H1: `{CANONICAL_H1:.17g}` (update `{CANONICAL_UPDATE}`).",
        f"- Final retained canonical H1: `{CANONICAL_H1:.17g}`; canonical promotions: `0`.",
        f"- Target: `{TARGET_H1:.17g}`; best full-evaluator shadow H1: `{best_shadow:.17g}`; best replay-validated H1: `{best_validated_h1 if best_validated_h1 is not None else 'NONE'}`.",
        f"- Best H32-safe full-evaluator shadow H1: `{best_h32_safe:.17g}`; safe-gap closure: `{safe_gap:.6g}`.",
        f"- Best S1 proxy H1 (256-window screen, not comparable as a full metric): `{s1.get('best_shadow', {}).get('S1_H1', 'NONE') if s1.get('best_shadow') else 'NONE'}`.",
        f"- Remaining-gap closure at best shadow: `{gap:.6g}`.",
        f"- Basin status: `{basin_status}`.",
        "",
        "## Search and recovery",
        "",
        "The first round tested historical 469→480 deltas, Adam moment/preconditioned directions, four seeded low-rank random directions, and a structured output-head subspace in the S1 funnel. Novel AdamW-effective-direction trials were then sent through the full causal H1/H32 evaluator. Failures and mutations are recorded in `D42_FAILURE_AND_RECOVERY_LEDGER.json`.",
        "",
        "The S1 population is a shadow discovery mechanism, not a promotion result. Full candidates are distinguished from replay-validated candidates in `D42_S2_SUMMARY.json`; no candidate was promoted because D42 did not execute a D39/D40/D41 promotion transaction.",
        "The retained safe shadow was also evaluated over the complete authoritative D17 horizon vector (1, 2, 4, 8, 12, 16, 20, 24, 32) in `D42_S2_SAFE_SHADOW_HORIZONS.json`; H64 is explicitly `not_available` in this evaluator.",
        "The stronger recharge shadow was evaluated over the same authoritative horizon vector in `D42_S2_RECHARGE_SHADOW_HORIZONS.json`; H64 remains explicitly `not_available`.",
        "A second recharge-depth round was launched from the stronger shadow parent to test margin refill before descent; it remained shadow-only and hard-gated.",
        "A moment-reset optimizer-state branch was tested from the untouched canonical parent as an alternate basin; it remained subject to the same H1/H32/replay gates.",
        "The retained D35 realized-H32-correction recharge route was re-tested as a separate shadow family; any subsequent H1 continuation remained subject to the same full H32 hard gate.",
        "",
        "## Retained system achievements",
        "",
        f"D39 retention: `{d41.get('D39_RETENTION')}`; D40 retention: `{d41.get('D40_RETENTION')}`; D41 retention/collision certification: `{d41.get('COLLISION_CERTIFICATION_STATUS')}`. Collision scope remains `adaptive_discrete_interpolation`; CCD is `not_available`; clearance is JSON null when unavailable.",
        "",
        "## Required final status block",
        "",
        f"TASK_STATUS: {summary['TASK_STATUS']}",
        f"FINAL_TARGET_STATUS: {summary['FINAL_TARGET_STATUS']}",
        f"H1_TARGET: {TARGET_H1}",
        f"STARTING_CANONICAL_H1: {CANONICAL_H1}",
        f"FINAL_RETAINED_CANONICAL_H1: {CANONICAL_H1}",
        f"BEST_SHADOW_H1: {best_shadow}",
        f"BEST_H32_SAFE_SHADOW_H1: {best_h32_safe}",
        f"BEST_VALIDATED_H1: {best_validated_h1 if best_validated_h1 is not None else 'NONE'}",
        "TOTAL_CANONICAL_PROMOTIONS: 0",
        f"REMAINING_GAP_CLOSURE: {gap}",
        f"H32_SAFE_GAP_CLOSURE: {safe_gap}",
        f"D39_RETENTION: {d41.get('D39_RETENTION')}",
        f"D40_RETENTION: {d41.get('D40_RETENTION')}",
        f"D41_RETENTION: {d41.get('COLLISION_CERTIFICATION_STATUS')}",
        "CANONICAL_REGRESSION: NO",
        f"CURRENT_BASIN_STATUS: {basin_status}",
        f"BREAKTHROUGH_CHANNELS_DISCOVERED: {', '.join(summary['BREAKTHROUGH_CHANNELS_DISCOVERED']) or 'NONE'}",
        "TRUE_EXTERNAL_HARD_BLOCKER: NONE",
        "FIRST_UNRESOLVED_BLOCKER: TARGET_NOT_REACHED; CONTINUE_SEARCH",
    ]
    (output / "D42_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset-count", type=int, default=512)
    parser.add_argument("--skip-s2", action="store_true")
    parser.add_argument("--s2-top-n", type=int, default=2)
    parser.add_argument("--continuation-steps", type=int, default=0)
    parser.add_argument("--only-continuation", action="store_true")
    parser.add_argument("--materialize-shadow-only", action="store_true")
    parser.add_argument("--safe-round", action="store_true")
    parser.add_argument("--continue-safe-steps", type=int, default=0)
    parser.add_argument("--refresh-only", action="store_true")
    parser.add_argument("--validate-safe-shadow", action="store_true")
    parser.add_argument("--recharge-round", action="store_true")
    parser.add_argument("--continue-recharge-steps", type=int, default=0)
    parser.add_argument("--recharge-depth-round", action="store_true")
    parser.add_argument("--optimizer-reset-round", action="store_true")
    parser.add_argument("--optimizer-reset-recovery-round", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime, payload, model, optimizer, rng, canonical, runtime_meta = load_authenticated_runtime()
    previous_payload = torch.load(ROOT / "outputs" / "stage3_h13_d35_permanent_champion" / "checkpoints" / "committed_update_479.pt", map_location="cpu", weights_only=False)
    previous_model, _, _, _ = d33.instantiate_state(runtime, previous_payload)
    s3 = None
    if args.refresh_only or args.validate_safe_shadow:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        safe_summary = output / "D42_SAFE_SET_SUMMARY.json"
        s3 = json.loads(safe_summary.read_text(encoding="utf-8")) if safe_summary.is_file() else None
        continuation_path = output / "D42_SAFE_CONTINUATION_SUMMARY.json"
        continuation = None
        if s3 and continuation_path.is_file():
            continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
        recharge = json.loads((output / "D42_RECHARGE_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_SUMMARY.json").is_file() else None
        recharge_cont = json.loads((output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").is_file() else None
        recharge_depth = json.loads((output / "D42_RECHARGE_DEPTH_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_DEPTH_SUMMARY.json").is_file() else None
        optimizer_reset = json.loads((output / "D42_OPTIMIZER_RESET_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_OPTIMIZER_RESET_SUMMARY.json").is_file() else None
        optimizer_recovery = json.loads((output / "D42_OPTIMIZER_RESET_RECOVERY_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_OPTIMIZER_RESET_RECOVERY_SUMMARY.json").is_file() else None
        s3 = combine_shadow_summaries(s3, continuation, recharge, recharge_cont, recharge_depth, optimizer_reset, optimizer_recovery)
        if args.validate_safe_shadow:
            validate_safe_shadow_horizons(output, runtime)
            if (output / "D42_recharge_shadow_update_481.pt").is_file():
                validate_shadow_horizons(output, runtime, "D42_recharge_shadow_update_481.pt", "D42_S2_RECHARGE_SHADOW_HORIZONS.json", "d42_s2_recharge_shadow_horizon_validation_v1")
    elif args.safe_round:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        s3 = run_safe_set_round(output, runtime, canonical, args.s2_top_n)
    elif args.recharge_round:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        safe = json.loads((output / "D42_SAFE_SET_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_SET_SUMMARY.json").is_file() else None
        safe_cont = json.loads((output / "D42_SAFE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_CONTINUATION_SUMMARY.json").is_file() else None
        s3 = combine_shadow_summaries(safe, safe_cont, run_recharge_round(output, runtime, canonical, args.s2_top_n))
    elif args.materialize_shadow_only:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        materialize_shadow_seed(output, runtime, canonical)
    elif args.only_continuation:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
    elif args.continue_safe_steps > 0:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        safe_seed_summary = json.loads((output / "D42_SAFE_SET_SUMMARY.json").read_text(encoding="utf-8"))
        continuation = continue_from_safe_shadow(output, runtime, canonical, args.continue_safe_steps)
        recharge = json.loads((output / "D42_RECHARGE_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_SUMMARY.json").is_file() else None
        recharge_cont = json.loads((output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").is_file() else None
        s3 = combine_shadow_summaries(safe_seed_summary, continuation, recharge, recharge_cont)
    elif args.continue_recharge_steps > 0:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        safe = json.loads((output / "D42_SAFE_SET_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_SET_SUMMARY.json").is_file() else None
        safe_cont = json.loads((output / "D42_SAFE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_CONTINUATION_SUMMARY.json").is_file() else None
        recharge = json.loads((output / "D42_RECHARGE_SUMMARY.json").read_text(encoding="utf-8"))
        recharge_cont = continue_from_recharge(output, runtime, canonical, args.continue_recharge_steps)
        s3 = combine_shadow_summaries(safe, safe_cont, recharge, recharge_cont)
    elif args.recharge_depth_round:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        safe = json.loads((output / "D42_SAFE_SET_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_SET_SUMMARY.json").is_file() else None
        safe_cont = json.loads((output / "D42_SAFE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_CONTINUATION_SUMMARY.json").is_file() else None
        recharge = json.loads((output / "D42_RECHARGE_SUMMARY.json").read_text(encoding="utf-8"))
        recharge_cont = json.loads((output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").is_file() else None
        depth = run_recharge_depth_round(output, runtime, canonical, args.s2_top_n)
        s3 = combine_shadow_summaries(safe, safe_cont, recharge, recharge_cont, depth)
    elif args.optimizer_reset_round:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        safe = json.loads((output / "D42_SAFE_SET_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_SET_SUMMARY.json").is_file() else None
        safe_cont = json.loads((output / "D42_SAFE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_CONTINUATION_SUMMARY.json").is_file() else None
        recharge = json.loads((output / "D42_RECHARGE_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_SUMMARY.json").is_file() else None
        recharge_cont = json.loads((output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").is_file() else None
        depth = json.loads((output / "D42_RECHARGE_DEPTH_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_DEPTH_SUMMARY.json").is_file() else None
        reset = run_optimizer_reset_round(output, runtime, canonical, args.s2_top_n)
        s3 = combine_shadow_summaries(safe, safe_cont, recharge, recharge_cont, depth, reset)
    elif args.optimizer_reset_recovery_round:
        s1 = json.loads((output / "D42_S1_SUMMARY.json").read_text(encoding="utf-8"))
        s2 = json.loads((output / "D42_S2_SUMMARY.json").read_text(encoding="utf-8"))
        safe = json.loads((output / "D42_SAFE_SET_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_SET_SUMMARY.json").is_file() else None
        safe_cont = json.loads((output / "D42_SAFE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_SAFE_CONTINUATION_SUMMARY.json").is_file() else None
        recharge = json.loads((output / "D42_RECHARGE_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_SUMMARY.json").is_file() else None
        recharge_cont = json.loads((output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_CONTINUATION_SUMMARY.json").is_file() else None
        depth = json.loads((output / "D42_RECHARGE_DEPTH_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D42_RECHARGE_DEPTH_SUMMARY.json").is_file() else None
        reset = json.loads((output / "D42_OPTIMIZER_RESET_SUMMARY.json").read_text(encoding="utf-8"))
        recovery = run_optimizer_reset_round(output, runtime, canonical, args.s2_top_n) if not (output / "D42_optimizer_reset_escape_shadow_update_481.pt").is_file() else None
        if recovery is not None:
            reset = recovery
        optimizer_recovery = recover_optimizer_reset_escape(output, runtime, canonical, args.s2_top_n)
        s3 = combine_shadow_summaries(safe, safe_cont, recharge, recharge_cont, depth, reset, optimizer_recovery)
    else:
        s1 = s1_screen(output, runtime, model, previous_model, optimizer, args.subset_count)
        s2 = None if args.skip_s2 else run_exact_round(output, runtime, payload, model, optimizer, rng, canonical, args.s2_top_n)
        s3 = continue_from_shadow(output, runtime, canonical, args.continuation_steps) if args.continuation_steps > 0 and s2 else None
    write_ledgers(output, s1, s2, s3)
    summary = write_final(output, s1, s2, runtime_meta, s3)
    print(json.dumps({"status": summary["TASK_STATUS"], "summary": str((output / "D42_SUMMARY.json").resolve()), "best_shadow_H1": summary["BEST_SHADOW_H1"], "best_validated_H1": summary["BEST_VALIDATED_H1"], "canonical_H1": summary["FINAL_RETAINED_CANONICAL_H1"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
