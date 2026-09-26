"""D34 isolated Champion-467 breakthrough screening.

This module never mutates the D33 archive.  It loads the authenticated update-467
checkpoint, evaluates direction generators in cloned inherited-AdamW branches,
and persists each shadow result immediately.  Canonical promotion is deliberately
separate and requires deterministic replay plus an explicit primary decision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_d33_inheritance_preserving_completion as d33

d32 = d33.d32

START_UPDATE = 467
START_H1 = 7.183493728036637e-05
START_H32 = 0.018447555404027906
TARGET_H1 = 5.0e-05
EXPECTED_SHA256 = "34dbeef6b51568f82db64e81d1b8f89786201bea1f64ec128b730e40251b6c42"
EXPECTED_PARENT_SHA256 = "73cc8bebb8e4203a84997360a7a1ad0c386a4198e497dde6708eff3249128837"
CHECKPOINT = (
    ROOT
    / "outputs"
    / "stage3_h13_d33_inheritance_preserving_completion_20260824T180000+0800"
    / "checkpoints"
    / "committed_update_467.pt"
)
MATERIAL_GAIN = 1.0e-7
STRONG_GAIN = 3.0e-7
D34_ADAPTIVE_PROMOTION_RESERVE = 7.5e-5


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zeros_like(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in values.items()}


def _masked(values: Mapping[str, torch.Tensor], prefixes: Sequence[str]) -> dict[str, torch.Tensor]:
    return {
        name: value.clone() if any(name.startswith(prefix) for prefix in prefixes) else torch.zeros_like(value)
        for name, value in values.items()
    }


def _parameter_blocks(values: Mapping[str, torch.Tensor]) -> list[tuple[str, ...]]:
    candidates = (
        ("gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0"),
        ("gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1"),
        ("head.0.",),
        ("head.2.",),
    )
    return [block for block in candidates if any(any(name.startswith(prefix) for prefix in block) for name in values)]


def blockwise_pcgrad(
    objective: Mapping[str, torch.Tensor], constraint: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Asymmetric PCGrad, independently projected within semantic parameter blocks."""
    result = _zeros_like(objective)
    conflicts = 0
    coefficients: list[float] = []
    for block in _parameter_blocks(objective):
        left = _masked(objective, block)
        right = _masked(constraint, block)
        denominator = d32.dot(right, right)
        coefficient = d32.dot(left, right) / max(denominator, 1.0e-30)
        if coefficient < 0.0 and denominator > 0.0:
            left = d32.add_maps(left, d32.scale_map(right, -coefficient))
            conflicts += 1
        coefficients.append(coefficient)
        result = d32.add_maps(result, left)
    return result, {"block_conflicts": conflicts, "block_projection_coefficients": coefficients}


def blockwise_tangent(
    objective: Mapping[str, torch.Tensor], constraint: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    result = _zeros_like(objective)
    coefficients: list[float] = []
    for block in _parameter_blocks(objective):
        left = _masked(objective, block)
        right = _masked(constraint, block)
        denominator = d32.dot(right, right)
        if denominator <= 1.0e-30:
            projected, coefficient = left, 0.0
        else:
            projected, coefficient = d32.project_tangent(left, right)
        result = d32.add_maps(result, projected)
        coefficients.append(coefficient)
    return result, {"block_projection_coefficients": coefficients}


def mgda_direction(
    objective: Mapping[str, torch.Tensor], constraint: Mapping[str, torch.Tensor],
    minimum_h1_weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Closed-form two-task minimum-norm direction with an H1-priority floor."""
    first, second = d32.unit(objective), d32.unit(constraint)
    difference = d32.add_maps(first, d32.scale_map(second, -1.0))
    denominator = d32.dot(difference, difference)
    unconstrained = -d32.dot(second, difference) / max(denominator, 1.0e-30)
    h1_weight = min(1.0, max(float(minimum_h1_weight), unconstrained))
    direction = d32.add_maps(d32.scale_map(first, h1_weight), d32.scale_map(second, 1.0 - h1_weight))
    return direction, {"mgda_unconstrained_h1_weight": unconstrained, "mgda_h1_weight": h1_weight}


def moment_target_direction(
    objective: Mapping[str, torch.Tensor],
    before_optimizer: Mapping[str, Mapping[str, Any]],
    target_fraction: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Choose a gradient whose next first moment points toward H1 descent.

    This preserves, rather than resets, inherited moments.  The proposed gradient
    analytically compensates the existing beta1 history before normalisation; the
    real inherited AdamW transition remains the source of truth in the shadow.
    """
    beta1 = float(d32.consistency.BETAS[0])
    old_m = {
        name: state.get("exp_avg", torch.zeros_like(objective[name])).detach().cpu().contiguous()
        for name, state in before_optimizer.items()
    }
    old_norm = d32.norm(old_m.values())
    target = d32.scale_map(d32.unit(objective), max(old_norm, 1.0e-12) * float(target_fraction))
    proposed = {
        name: (target[name] - beta1 * old_m[name]) / (1.0 - beta1)
        for name in objective
    }
    return proposed, {"inherited_first_moment_norm": old_norm, "moment_target_fraction": target_fraction}


def realized_active_set_correction(
    seed: Mapping[str, torch.Tensor],
    constraint: Mapping[str, torch.Tensor],
    before_parameters: Mapping[str, torch.Tensor],
    before_optimizer: Mapping[str, Mapping[str, Any]],
    effective_lr: float,
    target_norm: float,
    inward_cosine: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Correct a seed in predicted inherited-AdamW step space.

    The smallest nonnegative constraint-gradient coefficient whose predicted
    parameter step meets the requested inward cosine is found by bracketing and
    bisection.  Actual post-step H32 remains authoritative.
    """
    seed_unit, constraint_unit = d32.unit(seed), d32.unit(constraint)

    def score(amount: float) -> tuple[float, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        mixed = d32.add_maps(seed_unit, d32.scale_map(constraint_unit, amount))
        presented = d32.scale_map(mixed, target_norm / max(d32.norm(mixed.values()), 1.0e-30))
        predicted = d32.predicted_adamw_delta(before_parameters, before_optimizer, presented, effective_lr)
        value = d32.dot(predicted, constraint) / max(
            d32.norm(predicted.values()) * d32.norm(constraint.values()), 1.0e-30
        )
        return value, presented, predicted

    initial_score, initial_presented, initial_predicted = score(0.0)
    if initial_score <= -inward_cosine:
        return initial_presented, {
            "active_set_lambda": 0.0,
            "predicted_realized_h32_cosine_before": initial_score,
            "predicted_realized_h32_cosine_after": initial_score,
            "predicted_delta_norm": d32.norm(initial_predicted.values()),
        }
    low, high = 0.0, 0.05
    high_score, high_presented, high_predicted = score(high)
    while high_score > -inward_cosine and high < 64.0:
        high *= 2.0
        high_score, high_presented, high_predicted = score(high)
    if high_score > -inward_cosine:
        raise RuntimeError("D34_REALIZED_ACTIVE_SET_BRACKET_FAILURE")
    for _ in range(30):
        midpoint = 0.5 * (low + high)
        midpoint_score, midpoint_presented, midpoint_predicted = score(midpoint)
        if midpoint_score <= -inward_cosine:
            high, high_score, high_presented, high_predicted = midpoint, midpoint_score, midpoint_presented, midpoint_predicted
        else:
            low = midpoint
    return high_presented, {
        "active_set_lambda": high,
        "predicted_realized_h32_cosine_before": initial_score,
        "predicted_realized_h32_cosine_after": high_score,
        "predicted_delta_norm": d32.norm(high_predicted.values()),
    }


_INHERITED_BUILD = d32.build_presented_gradient


def build_d34_presented_gradient(
    spec: Mapping[str, Any],
    canonical_postclip: Mapping[str, torch.Tensor],
    h1_gradients: Mapping[str, torch.Tensor],
    h32_gradients: Mapping[str, torch.Tensor],
    before_parameters: Mapping[str, torch.Tensor],
    before_optimizer: Mapping[str, Mapping[str, Any]],
    effective_lr: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    family = str(spec["family"])
    target_norm = float(spec.get("target_norm", 1.0))
    metrics: dict[str, Any]
    if family == "D34_BLOCKWISE_PCGRAD":
        direction, metrics = blockwise_pcgrad(h1_gradients, h32_gradients)
    elif family == "D34_REALIZED_ACTIVE_SET_PCGRAD":
        seed, block_metrics = blockwise_pcgrad(h1_gradients, h32_gradients)
        presented, metrics = realized_active_set_correction(
            seed, h32_gradients, before_parameters, before_optimizer, effective_lr,
            target_norm, float(spec.get("inward_cosine", 0.0)),
        )
        metrics.update(block_metrics)
        metrics.update(
            {
                "used_h32_lambda": float(metrics["active_set_lambda"]),
                "presented_dot_h32": d32.dot(presented, h32_gradients),
                "presented_cosine_h1": d32.cosine(presented, h1_gradients),
                "presented_cosine_h32": d32.cosine(presented, h32_gradients),
                "d34_direction_generator": family,
            }
        )
        return presented, metrics
    elif family == "D34_BLOCKWISE_TANGENT":
        direction, metrics = blockwise_tangent(h1_gradients, h32_gradients)
    elif family.startswith("D34_BLOCK_H1_"):
        block_name = family.removeprefix("D34_BLOCK_H1_")
        prefixes = {
            "GRU0": ("gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0"),
            "GRU1": ("gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1"),
            "HEAD0": ("head.0.",),
            "HEAD2": ("head.2.",),
            "HEAD": ("head.",),
        }[block_name]
        direction, metrics = _masked(h1_gradients, prefixes), {"active_block": block_name}
    elif family == "D34_MGDA_H1_PRIORITY":
        direction, metrics = mgda_direction(h1_gradients, h32_gradients, float(spec.get("h1_weight", 0.9)))
    elif family == "D34_MOMENT_TARGET_H1":
        direction, metrics = moment_target_direction(h1_gradients, before_optimizer, float(spec.get("moment_fraction", 1.0)))
    else:
        return _INHERITED_BUILD(
            spec, canonical_postclip, h1_gradients, h32_gradients,
            before_parameters, before_optimizer, effective_lr,
        )
    direction_norm = d32.norm(direction.values())
    if not math.isfinite(direction_norm) or direction_norm <= 1.0e-30:
        raise RuntimeError(f"D34_ZERO_OR_NONFINITE_DIRECTION:{family}")
    presented = d32.scale_map(direction, target_norm / direction_norm)
    metrics.update(
        {
            "used_h32_lambda": float(spec.get("lambda", 0.0)),
            "presented_dot_h32": d32.dot(presented, h32_gradients),
            "presented_cosine_h1": d32.cosine(presented, h1_gradients),
            "presented_cosine_h32": d32.cosine(presented, h32_gradients),
            "d34_direction_generator": family,
        }
    )
    return presented, metrics


def first_wave_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    scales = (0.015625, 0.03125, 0.0625, 0.125)
    specs.extend({"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": scale} for scale in scales)
    for inward_cosine in (5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3):
        for scale in (0.03125, 0.0625):
            specs.append(
                {
                    "family": "D34_REALIZED_ACTIVE_SET_PCGRAD",
                    "base": "H1",
                    "alpha": scale,
                    "inward_cosine": inward_cosine,
                    "lambda": inward_cosine,
                }
            )
    specs.extend({"family": "D34_BLOCKWISE_TANGENT", "base": "H1", "alpha": scale} for scale in scales)
    for family in ("D34_BLOCK_H1_GRU0", "D34_BLOCK_H1_GRU1", "D34_BLOCK_H1_HEAD0", "D34_BLOCK_H1_HEAD2", "D34_BLOCK_H1_HEAD"):
        specs.extend({"family": family, "base": "H1", "alpha": scale} for scale in (0.03125, 0.0625, 0.125))
    for weight in (0.75, 0.9, 0.97):
        for scale in (0.03125, 0.0625, 0.125):
            specs.append({"family": "D34_MGDA_H1_PRIORITY", "base": "H1", "alpha": scale, "h1_weight": weight, "lambda": weight})
    for fraction in (0.5, 1.0, 2.0):
        for scale in (0.015625, 0.03125, 0.0625):
            specs.append({"family": "D34_MOMENT_TARGET_H1", "base": "H1", "alpha": scale, "moment_fraction": fraction, "lambda": fraction})
    return specs


def _screen_id(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_champion(runtime: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], list[float]]:
    if sha256_file(CHECKPOINT) != EXPECTED_SHA256:
        raise RuntimeError("UPDATE_467_CHECKPOINT_SHA256_MISMATCH")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if str(payload.get("parent_checkpoint_sha256")) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("UPDATE_467_PARENT_CHECKPOINT_SHA256_MISMATCH")
    d33.verify_loaded_payload(payload, START_UPDATE, START_H1, START_H32, 466, EXPECTED_PARENT_SHA256)
    return d33.instantiate_state(runtime, payload)


def screen(output: Path, limit: int | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    runtime = d32.d24.d17.preflight_runtime()
    if START_UPDATE + 1 > len(runtime["authority"]["schedule_rows"]):
        raise RuntimeError("AUTHENTICATED_SCHEDULE_EXHAUSTED")
    model, optimizer, rng, canonical = load_champion(runtime)
    identity = d32.semantic_parent_identity(model, optimizer, rng)
    args = (runtime, model, optimizer, rng, canonical, identity, START_UPDATE, START_UPDATE + 1, START_H1, START_H32)
    results_path = output / "D34_FIRST_WAVE_SHADOWS.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else []
    completed = {str(item["d34_spec_id"]) for item in results}
    specs = first_wave_specs()
    if limit is not None:
        specs = specs[:limit]
    original = d32.build_presented_gradient
    d32.build_presented_gradient = build_d34_presented_gradient
    try:
        for spec in specs:
            identity_value = _screen_id(spec)
            if identity_value in completed:
                continue
            try:
                record, _, _, _ = d32.run_trial(*args, spec)
                record["d34_spec_id"] = identity_value
                record["d34_spec"] = dict(spec)
                record["parent_champion_id"] = f"update-{START_UPDATE}:{EXPECTED_SHA256}"
                improvement = START_H1 - float(record["candidate_H1"])
                record["gain_class"] = (
                    "STRONG_BREAKTHROUGH" if improvement >= STRONG_GAIN else
                    "MATERIAL_GAIN" if improvement >= MATERIAL_GAIN else
                    "MICRO_OR_NO_GAIN"
                )
            except Exception as exc:
                record = {
                    "d34_spec_id": identity_value,
                    "d34_spec": dict(spec),
                    "parent_champion_id": f"update-{START_UPDATE}:{EXPECTED_SHA256}",
                    "trial_state": "SHADOW_FAILURE",
                    "rejection_reason": f"{type(exc).__name__}:{exc}",
                }
            results.append(record)
            completed.add(identity_value)
            atomic_json(results_path, results)
            if record.get("candidate_H1") is not None:
                print(
                    f"D34_SCREEN family={spec['family']} alpha={spec['alpha']:.8g} "
                    f"H1={record['candidate_H1']:.17g} H32={record['candidate_H32']:.17g} "
                    f"class={record['gain_class']}", flush=True,
                )
            else:
                print(f"D34_SCREEN_FAILURE family={spec['family']} reason={record['rejection_reason']}", flush=True)
    finally:
        d32.build_presented_gradient = original
    valid = [item for item in results if item.get("candidate_H1") is not None]
    ranked = sorted(valid, key=lambda item: float(item["candidate_H1"]))
    legal_ranked = [item for item in ranked if bool(item.get("candidate_valid_before_replay"))]
    summary = {
        "schema_version": "d34_first_wave_v1",
        "scientific_state": "SHADOW_ONLY_NO_CANONICAL_MUTATION",
        "parent_champion": {"update": START_UPDATE, "H1": START_H1, "H32": START_H32, "sha256": EXPECTED_SHA256},
        "candidate_specifications": len(results),
        "shadow_executions": len(results),
        "method_families": sorted({str(item["d34_spec"]["family"]) for item in results}),
        "best_legal": legal_ranked[0] if legal_ranked else None,
        "best_unconstrained_shadow": ranked[0] if ranked else None,
        "material_legal_candidates": [
            item for item in legal_ranked if START_H1 - float(item["candidate_H1"]) >= MATERIAL_GAIN
        ],
        "canonical_mutation": False,
    }
    atomic_json(output / "D34_FIRST_WAVE_SUMMARY.json", summary)
    return summary


def trajectory_profiles() -> dict[str, list[dict[str, Any]]]:
    recovery_small = {
        "family": "D34_REALIZED_ACTIVE_SET_PCGRAD", "base": "H1", "alpha": 0.03125,
        "inward_cosine": 5.0e-3, "lambda": 5.0e-3,
    }
    recovery_large = {
        "family": "D34_REALIZED_ACTIVE_SET_PCGRAD", "base": "H1", "alpha": 0.0625,
        "inward_cosine": 5.0e-3, "lambda": 5.0e-3,
    }
    return {
        "RECOVER_SMALL_THEN_PCGRAD_1_32": [recovery_small, {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": 0.03125}],
        "RECOVER_SMALL_THEN_PCGRAD_5_128": [recovery_small, {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": 0.0390625}],
        "RECOVER_SMALL_THEN_PCGRAD_3_64": [recovery_small, {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": 0.046875}],
        "RECOVER_SMALL_THEN_PCGRAD_1_16": [recovery_small, {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": 0.0625}],
        "RECOVER_LARGE_THEN_PCGRAD_1_16": [recovery_large, {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": 0.0625}],
    }


def _run_path(
    runtime: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], Any, Any, dict[str, Any]]:
    model, optimizer, rng, canonical = load_champion(runtime)
    update, h1, h32 = START_UPDATE, START_H1, START_H32
    records: list[dict[str, Any]] = []
    for spec in specs:
        identity = d32.semantic_parent_identity(model, optimizer, rng)
        child_update = update + 1
        record, model, optimizer, rng = d32.run_trial(
            runtime, model, optimizer, rng, canonical, identity,
            update, child_update, h1, h32, spec,
        )
        record["d34_spec"] = dict(spec)
        record["parent_champion_id"] = f"update-{START_UPDATE}:{EXPECTED_SHA256}"
        records.append(record)
        update, h1, h32 = child_update, float(record["candidate_H1"]), float(record["candidate_H32"])
    return records, model, optimizer, rng


def validate_trajectories(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    runtime = d32.d24.d17.preflight_runtime()
    profiles = trajectory_profiles()
    path = output / "D34_ADAPTIVE_TRAJECTORIES.json"
    retained = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    completed = {str(item["trajectory_id"]) for item in retained}
    original = d32.build_presented_gradient
    d32.build_presented_gradient = build_d34_presented_gradient
    try:
        for trajectory_id, specs in profiles.items():
            if trajectory_id in completed:
                continue
            try:
                first, model, optimizer, rng = _run_path(runtime, specs)
                replay, replay_model, replay_optimizer, replay_rng = _run_path(runtime, specs)
                replay_check = d32.d27.replay_compare(
                    first[-1], replay[-1], model, replay_model,
                    optimizer, replay_optimizer, rng, replay_rng,
                )
                all_step_safety = all(
                    bool(record.get("finite_state_pass"))
                    and bool(record.get("optimizer_consistency_pass"))
                    and bool(record.get("rng_identity_pass"))
                    and float(record["candidate_H32"]) <= d32.H32_LIMIT
                    for record in first
                )
                endpoint_gain = START_H1 - float(first[-1]["candidate_H1"])
                endpoint_valid = bool(
                    all_step_safety and replay_check["pass"]
                    and endpoint_gain > d32.MIN_H1_DESCENT
                )
                item = {
                    "trajectory_id": trajectory_id,
                    "parent_champion_id": f"update-{START_UPDATE}:{EXPECTED_SHA256}",
                    "specs": specs,
                    "steps": first,
                    "trajectory_length": len(first),
                    "endpoint_H1": float(first[-1]["candidate_H1"]),
                    "endpoint_delta_H1": float(first[-1]["candidate_H1"]) - START_H1,
                    "endpoint_H32": float(first[-1]["candidate_H32"]),
                    "endpoint_H32_margin": d32.H32_LIMIT - float(first[-1]["candidate_H32"]),
                    "minimum_H32_margin": min(d32.H32_LIMIT - float(record["candidate_H32"]) for record in first),
                    "all_intermediate_hard_safety_pass": all_step_safety,
                    "deterministic_replay": replay_check,
                    "endpoint_valid": endpoint_valid,
                    "promotion_state": "SHADOW_ONLY_NOT_PROMOTED",
                }
            except Exception as exc:
                item = {
                    "trajectory_id": trajectory_id,
                    "parent_champion_id": f"update-{START_UPDATE}:{EXPECTED_SHA256}",
                    "specs": specs,
                    "endpoint_valid": False,
                    "promotion_state": "SHADOW_FAILURE",
                    "rejection_reason": f"{type(exc).__name__}:{exc}",
                }
            retained.append(item)
            completed.add(trajectory_id)
            atomic_json(path, retained)
            print(
                f"D34_TRAJECTORY id={trajectory_id} valid={item.get('endpoint_valid')} "
                f"H1={item.get('endpoint_H1')} H32={item.get('endpoint_H32')} "
                f"replay={item.get('deterministic_replay', {}).get('pass')}", flush=True,
            )
    finally:
        d32.build_presented_gradient = original
    valid = sorted(
        (item for item in retained if item.get("endpoint_valid")),
        key=lambda item: float(item["endpoint_H1"]),
    )
    summary = {
        "schema_version": "d34_adaptive_trajectory_v1",
        "scientific_state": "SHADOW_ONLY_NO_CANONICAL_MUTATION",
        "parent_champion": {"update": START_UPDATE, "H1": START_H1, "H32": START_H32, "sha256": EXPECTED_SHA256},
        "trajectories_tested": len(retained),
        "best_valid_trajectory": valid[0] if valid else None,
        "canonical_mutation": False,
    }
    atomic_json(output / "D34_ADAPTIVE_TRAJECTORY_SUMMARY.json", summary)
    return summary


def promote_trajectory(output: Path, trajectory_id: str) -> dict[str, Any]:
    """Materialize one independently replayed trajectory endpoint as Champion.

    The intermediate optimizer step is retained in trajectory evidence only.  The
    resumable endpoint remains at its true optimizer/schedule step (469) and links
    directly to the prior canonical Champion checkpoint (467).
    """
    evidence_path = output / "D34_ADAPTIVE_TRAJECTORIES.json"
    if not evidence_path.is_file():
        raise RuntimeError("D34_TRAJECTORY_EVIDENCE_MISSING")
    evidence = {str(item["trajectory_id"]): item for item in json.loads(evidence_path.read_text(encoding="utf-8"))}
    if trajectory_id not in evidence:
        raise RuntimeError("D34_SELECTED_TRAJECTORY_MISSING")
    retained = evidence[trajectory_id]
    if not retained.get("endpoint_valid") or not retained.get("deterministic_replay", {}).get("pass"):
        raise RuntimeError("D34_SELECTED_TRAJECTORY_NOT_REPLAY_VALID")
    if -float(retained["endpoint_delta_H1"]) < MATERIAL_GAIN:
        raise RuntimeError("D34_SELECTED_TRAJECTORY_NOT_MATERIAL")
    if float(retained["endpoint_H32_margin"]) < D34_ADAPTIVE_PROMOTION_RESERVE:
        raise RuntimeError("D34_SELECTED_TRAJECTORY_RESERVE_TOO_SMALL")
    if retained.get("parent_champion_id") != f"update-{START_UPDATE}:{EXPECTED_SHA256}":
        raise RuntimeError("D34_STALE_TRAJECTORY_PARENT")
    runtime = d32.d24.d17.preflight_runtime()
    specs = list(retained["specs"])
    original = d32.build_presented_gradient
    d32.build_presented_gradient = build_d34_presented_gradient
    try:
        first, model, optimizer, rng = _run_path(runtime, specs)
        replay, replay_model, replay_optimizer, replay_rng = _run_path(runtime, specs)
    finally:
        d32.build_presented_gradient = original
    replay_check = d32.d27.replay_compare(
        first[-1], replay[-1], model, replay_model,
        optimizer, replay_optimizer, rng, replay_rng,
    )
    if not replay_check["pass"]:
        raise RuntimeError("D34_PROMOTION_REPLAY_FAILED")
    prior_digest = str(retained["deterministic_replay"]["candidate_record_digest"])
    if replay_check["candidate_record_digest"] != prior_digest:
        raise RuntimeError("D34_PROMOTION_REPLAY_DIFFERS_FROM_RETAINED_EVIDENCE")
    final = dict(first[-1])
    final.update(
        {
            "deterministic_replay_consistency_pass": True,
            "deterministic_replay_status": "PASS",
            "deterministic_replay_candidate_record_digest": replay_check["candidate_record_digest"],
            "deterministic_replay_record_digest": replay_check["replay_record_digest"],
            "candidate_valid": True,
            "selection_status": "D34_CHAMPION_PROMOTED",
            "trial_state": "COMMITTED_FORWARD_STATE",
        }
    )
    canonical = [float(value) for value in torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["canonical_base_lr"]]
    payload = d32.checkpoint_payload(model, optimizer, rng, final, EXPECTED_SHA256, canonical)
    payload.update(
        {
            "schema_version": "stage3_h13_d34_resumable_champion_checkpoint_v1",
            "experiment_id": "D34_CHAMPION_SNOWBALL_BREAKTHROUGH_SEARCH",
            "mode": "D34_ADAPTIVE_TWO_STEP_TRAJECTORY_ENDPOINT",
            "parent_update": START_UPDATE,
            "parent_checkpoint_sha256": EXPECTED_SHA256,
            "champion_parent_update": START_UPDATE,
            "champion_trajectory_updates": [int(record["candidate_update"]) for record in first],
            "champion_trajectory_length": len(first),
            "champion_trajectory_id": trajectory_id,
            "intermediate_states_canonical": False,
            "adaptive_H32_reserve": D34_ADAPTIVE_PROMOTION_RESERVE,
            "trajectory_evidence": copy.deepcopy(first),
            "scientific_state": "COMMITTED_FORWARD_STATE",
            "resumable": True,
        }
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    destination = checkpoints / "committed_update_469.pt"
    if destination.exists():
        raise RuntimeError("D34_CHAMPION_CHECKPOINT_ALREADY_EXISTS")
    temporary = checkpoints / "committed_update_469.pt.tmp"
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    states = loaded["optimizer_state_dict"]["state"].values()
    validation = {
        "parent_update_pass": int(loaded["parent_update"]) == START_UPDATE,
        "parent_sha256_pass": str(loaded["parent_checkpoint_sha256"]) == EXPECTED_SHA256,
        "completed_step_pass": int(loaded["completed_optimizer_step"]) == 469 and int(loaded["next_schedule_index"]) == 469,
        "optimizer_steps_pass": bool(states) and all(int(state.get("step", -1)) == 469 for state in states),
        "model_state_pass": d32.d27.tensor_state_equal(loaded["model_state_dict"], model.state_dict()),
        "optimizer_state_pass": d32.d27.tensor_state_equal(loaded["optimizer_state_dict"], optimizer.state_dict()),
        "rng_state_pass": d32.d27.tensor_state_equal(loaded["rng_state"], rng),
        "metric_pass": float(loaded["H1"]) == float(final["candidate_H1"]) and float(loaded["H32"]) == float(final["candidate_H32"]),
        "replay_pass": replay_check["pass"],
        "adaptive_reserve_pass": d32.H32_LIMIT - float(loaded["H32"]) >= D34_ADAPTIVE_PROMOTION_RESERVE,
    }
    validation["pass"] = all(validation.values())
    if not validation["pass"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"D34_CHAMPION_CHECKPOINT_VALIDATION_FAILED:{validation}")
    temporary.replace(destination)
    checkpoint_sha = sha256_file(destination)
    result = {
        "status": "PASS_CHAMPION_PROMOTED",
        "champion_update": 469,
        "parent_champion_update": START_UPDATE,
        "H1": float(final["candidate_H1"]),
        "delta_H1": float(final["candidate_H1"]) - START_H1,
        "H32": float(final["candidate_H32"]),
        "H32_margin": d32.H32_LIMIT - float(final["candidate_H32"]),
        "trajectory_id": trajectory_id,
        "trajectory_length": len(first),
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "validation": validation,
        "target_reached": float(final["candidate_H1"]) <= TARGET_H1,
    }
    atomic_json(output / "D34_CURRENT_CHAMPION.json", result)
    atomic_json(output / "validation" / "UPDATE_469_CHECKPOINT_VALIDATION.json", result)
    return result


def screen_current_champion(output: Path) -> dict[str, Any]:
    champion_meta_path = output / "D34_CURRENT_CHAMPION.json"
    if not champion_meta_path.is_file():
        raise RuntimeError("D34_CURRENT_CHAMPION_METADATA_MISSING")
    champion = json.loads(champion_meta_path.read_text(encoding="utf-8"))
    checkpoint = Path(champion["checkpoint"])
    if sha256_file(checkpoint) != str(champion["checkpoint_sha256"]):
        raise RuntimeError("D34_CURRENT_CHAMPION_SHA256_MISMATCH")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    current_update = int(champion["champion_update"])
    current_h1, current_h32 = float(champion["H1"]), float(champion["H32"])
    if current_update != 469 or int(payload["completed_optimizer_step"]) != current_update:
        raise RuntimeError("D34_CURRENT_CHAMPION_STEP_MISMATCH")
    runtime = d32.d24.d17.preflight_runtime()
    model, optimizer, rng, canonical = d33.instantiate_state(runtime, payload)
    parent_identity = d32.semantic_parent_identity(model, optimizer, rng)
    args = (
        runtime, model, optimizer, rng, canonical, parent_identity,
        current_update, current_update + 1, current_h1, current_h32,
    )
    specs = [
        {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": 0.015625},
        {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": 0.03125},
        {"family": "D34_REALIZED_ACTIVE_SET_PCGRAD", "base": "H1", "alpha": 0.03125, "inward_cosine": 0.002, "lambda": 0.002},
        {"family": "D34_REALIZED_ACTIVE_SET_PCGRAD", "base": "H1", "alpha": 0.0390625, "inward_cosine": 0.002, "lambda": 0.002},
        {"family": "D34_REALIZED_ACTIVE_SET_PCGRAD", "base": "H1", "alpha": 0.03125, "inward_cosine": 0.005, "lambda": 0.005},
        {"family": "D34_MGDA_H1_PRIORITY", "base": "H1", "alpha": 0.03125, "h1_weight": 0.9, "lambda": 0.9},
        {"family": "D34_BLOCK_H1_GRU0", "base": "H1", "alpha": 0.03125},
        {"family": "D34_BLOCK_H1_GRU1", "base": "H1", "alpha": 0.03125},
        {"family": "D34_BLOCK_H1_HEAD0", "base": "H1", "alpha": 0.03125},
        {"family": "D34_BLOCK_H1_HEAD2", "base": "H1", "alpha": 0.03125},
        {"family": "D34_BLOCK_H1_HEAD", "base": "H1", "alpha": 0.03125},
    ]
    results_path = output / "D34_POST_469_SHADOWS.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else []
    completed = {str(item["d34_spec_id"]) for item in results}
    original = d32.build_presented_gradient
    d32.build_presented_gradient = build_d34_presented_gradient
    try:
        for spec in specs:
            identity = _screen_id(spec)
            if identity in completed:
                continue
            try:
                record, _, _, _ = d32.run_trial(*args, spec)
                record.update(
                    {
                        "d34_spec_id": identity,
                        "d34_spec": dict(spec),
                        "parent_champion_id": f"update-{current_update}:{champion['checkpoint_sha256']}",
                        "promotion_state": "SHADOW_ONLY_NOT_REPLAYED",
                    }
                )
            except Exception as exc:
                record = {
                    "d34_spec_id": identity, "d34_spec": dict(spec),
                    "parent_champion_id": f"update-{current_update}:{champion['checkpoint_sha256']}",
                    "promotion_state": "SHADOW_FAILURE", "rejection_reason": f"{type(exc).__name__}:{exc}",
                }
            results.append(record)
            completed.add(identity)
            atomic_json(results_path, results)
            print(
                f"D34_POST469 family={spec['family']} alpha={spec['alpha']} "
                f"H1={record.get('candidate_H1')} H32={record.get('candidate_H32')}", flush=True,
            )
    finally:
        d32.build_presented_gradient = original
    valid = sorted(
        (item for item in results if item.get("candidate_valid_before_replay")),
        key=lambda item: float(item["candidate_H1"]),
    )
    summary = {
        "schema_version": "d34_post_champion_screen_v1",
        "scientific_state": "SHADOW_ONLY_NO_CANONICAL_MUTATION",
        "parent_champion": champion,
        "candidate_specifications": len(results),
        "best_pre_replay_legal": valid[0] if valid else None,
        "canonical_mutation": False,
    }
    atomic_json(output / "D34_POST_469_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument("--promote-trajectory")
    parser.add_argument("--screen-current-champion", action="store_true")
    args = parser.parse_args()
    try:
        if args.screen_current_champion:
            summary = screen_current_champion(args.output.resolve())
        elif args.promote_trajectory:
            summary = promote_trajectory(args.output.resolve(), args.promote_trajectory)
        elif args.trajectories:
            summary = validate_trajectories(args.output.resolve())
        else:
            summary = screen(args.output.resolve(), args.limit)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 1
    status = summary.get("status") or ("PASS_SHADOW_TRAJECTORIES" if args.trajectories else "PASS_SHADOW_SCREEN")
    print(json.dumps({"status": status, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
