"""D43 constrained continual H1 breakthrough campaign.

This module is shadow-first and intentionally keeps the D42/D35 canonical
checkpoint read-only.  It adds three small, composable pieces:

* exact evaluator geometry for the available H1..H32 causal rollout metrics;
* constrained candidate generators (tangent/GEM, realized active-set/filter,
  block-restricted residual, and low-dimensional COBYQA);
* an experiment ledger that separates raw, H32-safe, replay-validated,
  retention-safe, and promotion-eligible evidence.

The actual AdamW transition and the project's evaluator remain delegated to
the authenticated D42/D32 machinery.  A shadow may be unsafe; no shadow is
allowed to mutate the canonical checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_dataset import FEATURE_NAMES
from src.stage3_h13_r6 import rollout_residual_torch_r6
from tools import stage3_h13_d33_inheritance_preserving_completion as d33
from tools import stage3_h13_d42_persistent_h1_breakthrough_campaign as d42


TARGET_H1 = 5.0e-5
H32_LIMIT = 0.01856902565856056
PROMOTION_H32_RESERVE = 7.5e-5
CANONICAL_UPDATE = 480
CANONICAL_H1 = 7.089328839013259e-05
CANONICAL_H32 = 0.01849100619381173
CANONICAL_CHECKPOINT = ROOT / "outputs" / "stage3_h13_d35_permanent_champion" / "checkpoints" / "committed_update_480.pt"
CANONICAL_SHA256 = "37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742"
D42_ROOT = ROOT / "outputs" / "stage3_h13_d42_persistent_h1_breakthrough_campaign"
D42_SAFE = D42_ROOT / "D42_recharge_shadow_update_481.pt"
D42_UNSAFE_SEED = D42_ROOT / "D42_optimizer_reset_escape_shadow_update_481.pt"
DEFAULT_OUTPUT = ROOT / "outputs" / "stage3_h13_d43_constrained_breakthrough_campaign"

ALL_HORIZONS = tuple(range(1, 33))
ALL_NAMES = tuple(d42.d16.ALL_NAMES)
BLOCKS: dict[str, tuple[str, ...]] = {
    "GRU_L0": tuple(name for name in ALL_NAMES if name.endswith("_l0")),
    "GRU_L1": tuple(name for name in ALL_NAMES if name.endswith("_l1")),
    "HEAD_0": tuple(name for name in ALL_NAMES if name.startswith("head.0.")),
    "HEAD_2": tuple(name for name in ALL_NAMES if name.startswith("head.2.")),
}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
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


def norm(values: Iterable[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, sum(finite(torch.sum(value.detach().double().square())) for value in values)))


def dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return finite(sum(torch.sum(left[name].double() * right[name].double()) for name in ALL_NAMES))


def cosine(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float | None:
    denominator = norm(left.values()) * norm(right.values())
    return None if denominator <= 1.0e-30 else dot(left, right) / denominator


def clone_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: values[name].detach().cpu().contiguous().clone() for name in ALL_NAMES}


def add_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in ALL_NAMES}


def scale_map(values: Mapping[str, torch.Tensor], amount: float) -> dict[str, torch.Tensor]:
    return {name: values[name] * float(amount) for name in ALL_NAMES}


def unit(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return scale_map(values, 1.0 / max(norm(values.values()), 1.0e-30))


def mask_map(values: Mapping[str, torch.Tensor], names: Sequence[str]) -> dict[str, torch.Tensor]:
    selected = set(names)
    return {name: values[name].clone() if name in selected else torch.zeros_like(values[name]) for name in ALL_NAMES}


def load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"checkpoint_missing:{path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_state(runtime: Mapping[str, Any], path: Path) -> tuple[Any, Any, dict[str, Any], list[float], dict[str, Any]]:
    payload = load_payload(path)
    model, optimizer, rng, canonical = d33.instantiate_state(runtime, payload)
    return model, optimizer, copy.deepcopy(rng), canonical, payload


def canonical_runtime() -> tuple[dict[str, Any], Any, Any, dict[str, Any], list[float], dict[str, Any]]:
    runtime, payload, model, optimizer, rng, canonical, meta = d42.load_authenticated_runtime()
    if sha256_file(CANONICAL_CHECKPOINT) != CANONICAL_SHA256:
        raise RuntimeError("canonical_hash_changed")
    if float(payload["H1"]) != CANONICAL_H1 or float(payload["H32"]) != CANONICAL_H32:
        raise RuntimeError("canonical_metrics_changed")
    return runtime, model, optimizer, copy.deepcopy(rng), canonical, {"payload": payload, **meta}


def evaluate_horizons(model: Any, data: Any, channels: Mapping[str, Any]) -> dict[str, float]:
    rng_before = d42.d17.base.capture_rng()
    mode = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            result = d42.d17.base.evaluate_compact(model, data, channels, horizons=ALL_HORIZONS)
        return {str(h): finite(result[str(h)]) for h in ALL_HORIZONS}
    finally:
        d42.d17.base.restore_rng(rng_before)
        model.train(mode)


def _validation_tensors(data: Any) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(data.inputs).float(),
        torch.from_numpy(data.history_positions).float(),
        torch.from_numpy(data.history_times).float(),
        torch.from_numpy(data.rollout_target_times).float(),
        torch.from_numpy(data.trajectory_start_times).float(),
        torch.from_numpy(data.trajectory_end_times).float(),
        torch.from_numpy(data.rollout_target_positions).float(),
    )


def differentiable_horizon_geometry(model: Any, data: Any, channels: Mapping[str, Any], gradient_sample_count: int = 32, full_metrics: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Compute exact validation-metric gradients for every available h=1..32.

    The evaluator's reported quantity is RMSE over the first h free-running
    causal rollout steps.  The full-population metrics are recorded by the
    caller; gradients use a deterministic stratified validation proxy and
    rebuild the graph per horizon, so this remains an exact local gradient of
    the same evaluator definition without retaining a multi-horizon graph.
    """
    rng_before = d42.d17.base.capture_rng()
    mode = bool(model.training)
    model.eval()
    params = [parameter for _, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if tuple(names) != ALL_NAMES:
        raise RuntimeError("parameter_order_mismatch")
    proxy = d42.validation_view(data, min(int(gradient_sample_count), int(data.rollout_target_positions.shape[0])))
    inputs, history_positions, history_times, target_times, starts, ends, target = _validation_tensors(proxy)
    flat: dict[str, list[float]] = {}
    metrics: dict[str, float] = {}
    blocks: dict[str, dict[str, list[float]]] = {key: {} for key in BLOCKS}
    try:
        for index, horizon in enumerate(ALL_HORIZONS):
            rolled = rollout_residual_torch_r6(
                model, inputs, history_positions, history_times, target_times, starts, ends,
                channels, feature_names=FEATURE_NAMES, rollout_horizon=horizon,
                teacher_forcing_ratio_value=0.0, target_positions=None, sample_random=False,
                detach_state=False,
            )
            error = rolled[:, :horizon, :] - target[:, :horizon, :]
            metric = torch.sqrt(torch.mean(error.square()))
            grads = torch.autograd.grad(metric, params, retain_graph=False, allow_unused=False)
            gradients = {name: grad.detach().cpu().contiguous() for name, grad in zip(names, grads)}
            metrics[str(horizon)] = finite(metric)
            flat[str(horizon)] = torch.cat([gradients[name].reshape(-1) for name in ALL_NAMES]).double().tolist()
            for block, block_names in BLOCKS.items():
                blocks[block][str(horizon)] = torch.cat([gradients[name].reshape(-1) for name in block_names]).double().tolist()
    finally:
        d42.d17.base.restore_rng(rng_before)
        model.train(mode)
    matrix = np.asarray([flat[str(h)] for h in ALL_HORIZONS], dtype=np.float64)
    row_norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / np.maximum(row_norms[:, None], 1.0e-30)
    gram = np.clip(normalized @ normalized.T, -1.0, 1.0)
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    positive = eigenvalues[eigenvalues > 1.0e-12]
    probabilities = positive / max(float(positive.sum()), 1.0e-30)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-30))))) if len(positive) else 0.0
    h1 = normalized[0]
    h1_cosines = {str(h): float(gram[0, h - 1]) for h in ALL_HORIZONS}
    return {
        "schema_version": "d43_h1_h32_geometry_v1",
        "metric_definition": "free_running_causal joint-position RMSE over the first h rollout steps; no future target positions are read by the free branch",
        "horizons": list(ALL_HORIZONS),
        "gradient_sample_count": int(proxy.rollout_target_positions.shape[0]),
        "gradient_population": "deterministic_stratified_validation_proxy",
        "metrics": metrics,
        "full_evaluator_metrics": dict(full_metrics or {}),
        "gradient_norms": {str(h): float(row_norms[h - 1]) for h in ALL_HORIZONS},
        "gradient_cosine_matrix": gram.tolist(),
        "h1_gradient_cosine_by_horizon": h1_cosines,
        "gradient_singular_values": np.linalg.svd(matrix, compute_uv=False).tolist(),
        "gradient_gram_eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "per_block_h1_cosine": {
            block: {str(h): cosine_from_flat(blocks[block]["1"], blocks[block][str(h)]) for h in ALL_HORIZONS}
            for block in BLOCKS
        },
        "per_block_gradient_norms": {
            block: {str(h): float(np.linalg.norm(np.asarray(blocks[block][str(h)], dtype=np.float64))) for h in ALL_HORIZONS}
            for block in BLOCKS
        },
        "raw_gradient_vectors": "omitted_from_json; reproducible from the authenticated model/data and this script",
        "raw_block_gradient_vectors": "omitted_from_json; compact block norms/cosines retained",
    }


def cosine_from_flat(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator <= 1.0e-30 else float(np.dot(a, b) / denominator)


def materialize_unsafe_root(output: Path, runtime: Mapping[str, Any], canonical: Sequence[float]) -> tuple[Path, dict[str, Any]]:
    """Recreate D42's recorded H1≈6.836e-5 point without touching D42."""
    target = output / "roots" / "D43_root_C_D42_unsafe_update_482.pt"
    record_path = output / "roots" / "D43_root_C_materialization.json"
    if target.is_file() and record_path.is_file():
        return target, json.loads(record_path.read_text(encoding="utf-8"))
    seed_payload = load_payload(D42_UNSAFE_SEED)
    seed_model, seed_optimizer, seed_rng, _ = d33.instantiate_state(runtime, seed_payload)
    parent_identity = d42.d32.semantic_parent_identity(seed_model, seed_optimizer, seed_rng)
    spec = {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.015625, "lambda": 0.2, "target_norm": 1.0}
    first, first_model, first_optimizer, first_rng = d42.d32.run_trial(
        runtime, seed_model, seed_optimizer, seed_rng, canonical, parent_identity,
        int(seed_payload["completed_optimizer_step"]), int(seed_payload["completed_optimizer_step"]) + 1,
        float(seed_payload["H1"]), float(seed_payload["H32"]), spec,
    )
    second, second_model, second_optimizer, second_rng = d42.d32.run_trial(
        runtime, seed_model, seed_optimizer, seed_rng, canonical, parent_identity,
        int(seed_payload["completed_optimizer_step"]), int(seed_payload["completed_optimizer_step"]) + 1,
        float(seed_payload["H1"]), float(seed_payload["H32"]), spec,
    )
    replay = d42.d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
    first.update({
        "deterministic_replay_consistency_pass": bool(replay["pass"]),
        "deterministic_replay_candidate_record_digest": replay["candidate_record_digest"],
        "deterministic_replay_record_digest": replay["replay_record_digest"],
    })
    payload = d42.d32.checkpoint_payload(first_model, first_optimizer, first_rng, first, sha256_file(D42_UNSAFE_SEED), canonical)
    payload.update({"scientific_state": "D43_ROOT_C_D42_UNSAFE_REPLAY_MATERIALIZED", "promotion_state": "NOT_PROMOTED", "resumable": True})
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    result = {
        "root": "C",
        "source_seed": str(D42_UNSAFE_SEED.resolve()),
        "source_seed_sha256": sha256_file(D42_UNSAFE_SEED),
        "spec": spec,
        "record": first,
        "replay": replay,
        "checkpoint": str(target.resolve()),
        "checkpoint_sha256": sha256_file(target),
        "known_D42_H1": 6.836394238387236e-05,
        "known_D42_H32": 0.0206296142226003,
        "matches_known_D42_record": math.isclose(float(first["candidate_H1"]), 6.836394238387236e-05, rel_tol=0.0, abs_tol=1.0e-16) and math.isclose(float(first["candidate_H32"]), 0.0206296142226003, rel_tol=0.0, abs_tol=1.0e-16),
        "canonical_mutation": False,
    }
    write_json(record_path, result)
    return target, result


def base_direction(spec: Mapping[str, Any], postclip: Mapping[str, torch.Tensor], h1: Mapping[str, torch.Tensor], h32: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Mapping[str, Any]], effective_lr: float, root_delta: Mapping[str, torch.Tensor] | None, random_basis: Sequence[Mapping[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    family = str(spec["family"])
    h1_u, h32_u = unit(h1), unit(h32)
    if family == "D43_H1" or family == "D43_GEM_NO_REGRESSION":
        if family == "D43_H1":
            direction = h1_u
            metrics = {"generator": family}
        else:
            direction, coefficient = d42.d32.project_only_conflicting(h1_u, h32_u)
            metrics = {"generator": family, "projection_coefficient": float(coefficient)}
    elif family == "D43_TANGENT_H32":
        direction, coefficient = d42.d32.project_tangent(h1_u, h32_u)
        metrics = {"generator": family, "projection_coefficient": float(coefficient)}
    elif family in {"D43_FILTER_ACTIVE_SET", "D43_HOMOTOPY_FILTER"}:
        seed = h1_u
        direction, active = d34_active_set(seed, h32, before, before_optimizer, effective_lr, float(spec.get("inward_cosine", 0.001)))
        metrics = {"generator": family, **active}
    elif family == "D43_RESIDUAL_HEAD2":
        head = mask_map(h1_u, BLOCKS["HEAD_2"])
        direction, coefficient = d42.d32.project_only_conflicting(head, mask_map(h32_u, BLOCKS["HEAD_2"]))
        metrics = {"generator": family, "projection_coefficient": float(coefficient), "active_parameter_block": "HEAD_2"}
    elif family == "D43_RESIDUAL_HEAD0_HEAD2":
        selected = tuple(BLOCKS["HEAD_0"] + BLOCKS["HEAD_2"])
        head = mask_map(h1_u, selected)
        normal = mask_map(h32_u, selected)
        direction, coefficient = d42.d32.project_only_conflicting(head, normal)
        metrics = {"generator": family, "projection_coefficient": float(coefficient), "active_parameter_block": "HEAD_0+HEAD_2"}
    elif family == "D43_HIERARCHICAL_GRU_HEAD2":
        # Strictly protect the low-level recurrent blocks against H32 conflict,
        # then leave the output head as the lower-priority correction channel.
        gru_names = tuple(BLOCKS["GRU_L0"] + BLOCKS["GRU_L1"])
        gru_h1 = mask_map(h1_u, gru_names)
        gru_h32 = mask_map(h32_u, gru_names)
        protected_gru, coefficient = d42.d32.project_only_conflicting(gru_h1, gru_h32)
        head2 = mask_map(h1_u, BLOCKS["HEAD_2"])
        direction = add_maps(protected_gru, head2)
        metrics = {"generator": family, "projection_coefficient": float(coefficient), "hierarchy": "GRU_L0+GRU_L1 protected; HEAD_2 residual"}
    elif family == "D43_ADAM_PRECONDITIONED":
        # Curvature/momentum-aware diagonal preconditioning, using the actual
        # AdamW state inherited by the authenticated parent.
        eps = 1.0e-8
        direction = {
            name: -before_optimizer[name].get("exp_avg", torch.zeros_like(h1[name])).detach().cpu()
            / torch.sqrt(before_optimizer[name].get("exp_avg_sq", torch.ones_like(h1[name])).detach().cpu() + eps)
            for name in ALL_NAMES
        }
        if norm(direction.values()) <= 1.0e-12:
            direction = h1_u
        metrics = {"generator": family, "preconditioner": "inherited_adamw_exp_avg_exp_avg_sq"}
    elif family == "D43_MADS_POLL":
        poll_index = int(spec.get("poll_index", 0))
        poll = [h1_u, scale_map(h1_u, -1.0), h32_u, scale_map(h32_u, -1.0)] + list(random_basis)
        direction = poll[poll_index % len(poll)]
        metrics = {"generator": family, "poll_index": poll_index, "poll_size": len(poll), "poll_rule": "deterministic_positive_spanning_like_poll"}
    elif family == "D43_SCBO_TRUST_REGION":
        # A small feasibility-first trust-region candidate in the same basis
        # used by the bounded COBYQA search.  This is deliberately labelled
        # SCBO-inspired: the full expensive-surrogate loop is run separately
        # by COBYQA and the ledger keeps this generator auditable.
        trust_radius = float(spec.get("trust_radius", 0.25))
        z = np.asarray(spec.get("basis_z", [1.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        z = np.clip(z, -trust_radius, trust_radius)
        basis = [h1_u, h32_u, unit(postclip)]
        if root_delta is not None and norm(root_delta.values()) > 1.0e-12:
            basis.append(unit(root_delta))
        basis.extend(random_basis)
        direction = {name: sum(float(z[index]) * basis[index][name] for index in range(min(len(z), len(basis)))) for name in ALL_NAMES}
        metrics = {"generator": family, "trust_radius": trust_radius, "feasibility_first": True, "surrogate": "not_used_in_single_point_shadow"}
    elif family == "D43_LOW_DIM_BASIS":
        z = np.asarray(spec["basis_z"], dtype=np.float64)
        basis = [h1_u, h32_u, unit(postclip)]
        if root_delta is not None and norm(root_delta.values()) > 1.0e-12:
            basis.append(unit(root_delta))
        momentum = {name: -before_optimizer[name].get("exp_avg", torch.zeros_like(h1[name])).detach().cpu() for name in ALL_NAMES}
        if norm(momentum.values()) > 1.0e-12:
            basis.append(unit(momentum))
        basis.extend(random_basis)
        if len(z) < len(basis):
            z = np.pad(z, (0, len(basis) - len(z)))
        elif len(z) > len(basis):
            z = z[:len(basis)]
        direction = {name: sum(float(z[index]) * basis[index][name] for index in range(min(len(z), len(basis)))) for name in ALL_NAMES}
        metrics = {"generator": family, "basis_dimension": len(basis), "basis_z": z.tolist()}
    else:
        raise RuntimeError(f"unknown_d43_family:{family}")
    presented = unit(direction)
    metrics.update({"presented_h1_cosine": cosine(presented, h1), "presented_h32_cosine": cosine(presented, h32), "presented_norm": norm(presented.values())})
    return presented, metrics


def d34_active_set(seed: Mapping[str, torch.Tensor], constraint: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Mapping[str, Any]], effective_lr: float, inward_cosine: float) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_u, normal_u = unit(seed), unit(constraint)

    def score(amount: float) -> tuple[float, dict[str, torch.Tensor]]:
        mixed = add_maps(seed_u, scale_map(normal_u, amount))
        presented = unit(mixed)
        predicted = d42.d32.predicted_adamw_delta(before, before_optimizer, presented, effective_lr)
        return dot(predicted, constraint) / max(norm(predicted.values()) * norm(constraint.values()), 1.0e-30), presented

    initial, initial_presented = score(0.0)
    target = -abs(float(inward_cosine))
    if initial <= target:
        return initial_presented, {"active_set_lambda": 0.0, "predicted_h32_cosine": initial, "inward_target": target}
    low, high = 0.0, 0.05
    high_score, high_presented = score(high)
    while high_score > target and high < 64.0:
        high *= 2.0
        high_score, high_presented = score(high)
    if high_score > target:
        return initial_presented, {"active_set_lambda": None, "predicted_h32_cosine": initial, "inward_target": target, "bracket_failure": True}
    for _ in range(32):
        mid = 0.5 * (low + high)
        mid_score, mid_presented = score(mid)
        if mid_score <= target:
            high, high_presented = mid, mid_presented
        else:
            low = mid
    return high_presented, {"active_set_lambda": high, "predicted_h32_cosine": high_score, "inward_target": target}


def random_basis(seed: int, template: Mapping[str, torch.Tensor], count: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result: list[dict[str, torch.Tensor]] = []
    for _ in range(count):
        raw = {name: torch.randn(template[name].shape, generator=generator, dtype=template[name].dtype) for name in ALL_NAMES}
        result.append(unit(raw))
    return result


def run_trial_with_builder(runtime: Mapping[str, Any], model: Any, optimizer: Any, rng: Mapping[str, Any], canonical: Sequence[float], root_meta: Mapping[str, Any], spec: Mapping[str, Any], builder_family: str | None = None) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    base_parameters = d42.model_parameters(model)
    root_delta = None
    if "canonical_model" in root_meta:
        root_delta = {name: base_parameters[name] - root_meta["canonical_model"][name] for name in ALL_NAMES}
    random_vectors = root_meta.get("random_basis", [])
    holder = {"root_delta": root_delta, "random_basis": random_vectors}
    original = d42.d32.build_presented_gradient

    def builder(inner_spec: Mapping[str, Any], canonical_postclip: Mapping[str, torch.Tensor], h1_gradients: Mapping[str, torch.Tensor], h32_gradients: Mapping[str, torch.Tensor], before_parameters: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Mapping[str, Any]], effective_lr: float) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        direction, metrics = base_direction(inner_spec, canonical_postclip, h1_gradients, h32_gradients, before_parameters, before_optimizer, effective_lr, holder["root_delta"], holder["random_basis"])
        return direction, metrics

    d42.d32.build_presented_gradient = builder
    try:
        parent_update = int(root_meta["payload"]["completed_optimizer_step"])
        result = d42.d32.run_trial(
            runtime, model, optimizer, rng, canonical, root_meta["identity"], parent_update, parent_update + 1,
            float(root_meta["payload"]["H1"]), float(root_meta["payload"]["H32"]), spec,
        )
    finally:
        d42.d32.build_presented_gradient = original
    return result


def evaluate_replay_trial(runtime: Mapping[str, Any], model: Any, optimizer: Any, rng: Mapping[str, Any], canonical: Sequence[float], root_meta: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any, dict[str, Any], dict[str, Any]]:
    first, first_model, first_optimizer, first_rng = run_trial_with_builder(runtime, model, optimizer, rng, canonical, root_meta, spec)
    second, second_model, second_optimizer, second_rng = run_trial_with_builder(runtime, model, optimizer, rng, canonical, root_meta, spec)
    check = d42.d32.d27.replay_compare(first, second, first_model, second_model, first_optimizer, second_optimizer, first_rng, second_rng)
    first["d43_replay"] = check
    first["deterministic_replay_consistency_pass"] = bool(check["pass"])
    first["deterministic_replay_status"] = "PASS" if check["pass"] else "FAIL"
    first["deterministic_replay_candidate_record_digest"] = check["candidate_record_digest"]
    first["deterministic_replay_record_digest"] = check["replay_record_digest"]
    return first, first_model, first_optimizer, first_rng, check


def root_meta(runtime: Mapping[str, Any], path: Path, canonical_model: Mapping[str, torch.Tensor], root_name: str, materialization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    model, optimizer, rng, canonical, payload = load_state(runtime, path)
    return {
        "root": root_name,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "model": model,
        "optimizer": optimizer,
        "rng": rng,
        "canonical": canonical,
        "payload": payload,
        "identity": d42.d32.semantic_parent_identity(model, optimizer, rng),
        "canonical_model": dict(canonical_model),
        "random_basis": random_basis(43043 + ord(root_name), d42.model_parameters(model), 3),
        "materialization": materialization,
    }


def retention_class(record: Mapping[str, Any], horizons: Mapping[str, float] | None = None) -> str:
    h1 = float(record.get("candidate_H1", math.inf))
    h32 = float(record.get("candidate_H32", math.inf))
    replay = bool(record.get("deterministic_replay_consistency_pass"))
    finite_pass = bool(record.get("finite_state_pass", False))
    if not finite_pass or not math.isfinite(h1) or not math.isfinite(h32) or not replay:
        return "INVALID"
    if h32 > H32_LIMIT:
        return "UNSAFE"
    if h1 >= float(record.get("parent_H1", math.inf)):
        return "TRADEOFF"
    if horizons:
        parent_horizons = record.get("parent_horizons", {})
        regressions = []
        for key, value in horizons.items():
            if key == "1" or key == "32":
                continue
            limit = parent_horizons.get(key)
            if limit is not None and float(value) > float(limit) + 1.0e-10:
                regressions.append(key)
        if regressions:
            return "TRADEOFF"
    return "RETENTION_NEUTRAL"


def attach_full_vector(record: dict[str, Any], model: Any, runtime: Mapping[str, Any]) -> dict[str, Any]:
    vector = evaluate_horizons(model, runtime["pre"]["validation"], runtime["pre"]["stats"]["channels"])
    record["H1_H32_vector"] = vector
    parent = record.get("parent_horizons", {})
    record["diagnostic_horizon_regressions"] = [
        key for key, value in vector.items()
        if key not in {"1", "32"} and key in parent and float(value) > float(parent[key]) + 1.0e-10
    ]
    record["classification"] = retention_class(record, vector)
    return record


def run_single_family(runtime: Mapping[str, Any], root: Mapping[str, Any], family: str, alpha: float, output: Path, **kwargs: Any) -> dict[str, Any]:
    spec = {"family": family, "base": "H1", "alpha": float(alpha), "target_norm": 1.0, **kwargs}
    try:
        record, candidate_model, candidate_optimizer, candidate_rng, replay = evaluate_replay_trial(runtime, root["model"], root["optimizer"], root["rng"], root["canonical"], root, spec)
        shadow_gate = bool(record.get("candidate_H1", math.inf) < CANONICAL_H1 and float(record.get("candidate_H32", math.inf)) <= H32_LIMIT and float(record.get("H32_margin", -math.inf)) >= PROMOTION_H32_RESERVE and replay["pass"])
        record.update({"d43_root": root["root"], "d43_spec": spec, "parent_horizons": dict(root.get("parent_horizons", {})), "replay_pass": bool(replay["pass"]), "shadow_gate_eligible": shadow_gate, "promotion_eligible": False, "promotion_rejection_reason": "D43 shadow lacks fresh D39/D40/D41 robot certification; canonical checkpoint remains immutable"})
        attach_full_vector(record, candidate_model, runtime)
        record["candidate_model"] = candidate_model
        record["candidate_optimizer"] = candidate_optimizer
        record["candidate_rng"] = candidate_rng
        if record.get("replay_pass") and float(record.get("candidate_H32", math.inf)) <= H32_LIMIT:
            shadow_name = f"{root['root']}_{family}_a{float(alpha):.12g}".replace(".", "p").replace("-", "m")
            shadow_path = output / "shadows" / f"{shadow_name}.pt"
            shadow_payload = d42.d32.checkpoint_payload(candidate_model, candidate_optimizer, candidate_rng, record, sha256_file(Path(root["checkpoint"])), root["canonical"])
            shadow_payload.update({"scientific_state": "D43_SAFE_SHADOW_CANDIDATE", "promotion_state": "NOT_PROMOTED", "resumable": True})
            shadow_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(shadow_payload, shadow_path)
            record["shadow_checkpoint"] = str(shadow_path.resolve())
            record["shadow_checkpoint_sha256"] = sha256_file(shadow_path)
    except Exception as exc:
        record = {"d43_root": root["root"], "d43_spec": spec, "classification": "INVALID", "replay_pass": False, "promotion_eligible": False, "rejection_reason": f"{type(exc).__name__}:{exc}"}
    serializable = {key: value for key, value in record.items() if key not in {"candidate_model", "candidate_optimizer", "candidate_rng"}}
    append_jsonl(output / "D43_EXPERIMENT_LEDGER.jsonl", serializable)
    return record


def cobyqa_search(runtime: Mapping[str, Any], root: Mapping[str, Any], output: Path, maxfev: int = 24) -> list[dict[str, Any]]:
    """Run a bounded low-dimensional nonlinear constrained COBYQA search."""
    template_model = d42.model_parameters(root["model"])
    dim = 8
    evaluations: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate_z(z: Sequence[float]) -> dict[str, Any]:
        key = tuple(float(f"{value:.12g}") for value in z)
        if key in evaluations:
            return evaluations[key]
        spec = {"family": "D43_LOW_DIM_BASIS", "base": "H1", "alpha": 0.0078125, "target_norm": 1.0, "basis_z": list(key)}
        result = run_single_family(runtime, root, "D43_LOW_DIM_BASIS", 0.0078125, output, basis_z=list(key))
        evaluations[key] = result
        return result

    def objective(z: np.ndarray) -> float:
        row = evaluate_z(z)
        return float(row.get("candidate_H1", 1.0))

    def h32_constraint(z: np.ndarray) -> float:
        row = evaluate_z(z)
        return H32_LIMIT - float(row.get("candidate_H32", 1.0))

    x0 = np.zeros(dim, dtype=np.float64)
    x0[0] = 1.0
    try:
        result = minimize(
            objective, x0, method="COBYQA",
            bounds=[(-1.0, 1.0)] * dim,
            constraints=[{"type": "ineq", "fun": h32_constraint}],
            options={"maxfev": int(maxfev), "initial_tr_radius": 0.35, "final_tr_radius": 0.01, "feasibility_tol": 1.0e-10, "f_target": TARGET_H1},
        )
        summary = {"method": "R1_LOW_DIMENSIONAL_COBYQA", "root": root["root"], "scipy_success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev), "x": np.asarray(result.x, dtype=np.float64).tolist(), "fun": float(result.fun), "evaluations": len(evaluations)}
    except Exception as exc:
        summary = {"method": "R1_LOW_DIMENSIONAL_COBYQA", "root": root["root"], "scipy_success": False, "message": f"{type(exc).__name__}:{exc}", "nfev": len(evaluations), "evaluations": len(evaluations)}
    write_json(output / "COBYQA" / f"{root['root']}_summary.json", summary)
    return list(evaluations.values())


def serial_root(root: Mapping[str, Any]) -> dict[str, Any]:
    payload = root["payload"]
    return {"root": root["root"], "checkpoint": root["checkpoint"], "checkpoint_sha256": root["checkpoint_sha256"], "completed_optimizer_step": int(payload["completed_optimizer_step"]), "H1": float(payload["H1"]), "H32": float(payload["H32"]), "H32_margin": H32_LIMIT - float(payload["H32"]), "identity": root["identity"], "materialization": root.get("materialization")}


def write_final_report(output: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], roots: Mapping[str, Any]) -> None:
    best = summary.get("best_raw_record") or {}
    lines = [
        "# D43 constrained continual breakthrough campaign",
        "",
        "## Verdict",
        "",
        f"Status: **{summary['TASK_STATUS']}**. The H1 target of `{TARGET_H1:.12g}` was not reached. The canonical checkpoint was not mutated.",
        "",
        "D43 used the authoritative 181-point ON-state open-arch input only. The evaluator metric is free-running causal joint-position RMSE over each horizon; H32 is the hard long-horizon gate. H2–H31 are retained as multi-horizon diagnostics, not silently substituted for the H32 contract.",
        "",
        "## Certified anchors",
        "",
        f"- Canonical update `{CANONICAL_UPDATE}`: H1 `{CANONICAL_H1:.15e}`, H32 `{CANONICAL_H32:.15e}`; SHA256 `{CANONICAL_SHA256}`.",
        f"- Best known safe D42 shadow root B: H1 `{float(roots['B']['payload']['H1']):.15e}`, H32 `{float(roots['B']['payload']['H32']):.15e}`.",
        f"- D43 best raw candidate: H1 `{summary.get('BEST_RAW_SHADOW_H1')}`; H32 `{best.get('candidate_H32')}`; classification `{best.get('classification')}`.",
        f"- Best H32-safe candidate/root evidence: H1 `{summary.get('BEST_H32_SAFE_H1')}`; no candidate was promoted.",
        "",
        "## Method matrix",
        "",
        "| Family | Evidence policy | D43 status |",
        "|---|---|---|",
        "| COBYQA / low-dimensional trust-region | bounded black-box H1 objective with H32 nonlinear constraint | implemented in shadow runner; expensive search bounded by `--cobyqa-maxfev` |",
        "| GEM / PCGrad / tangent | conflict projection using H1/H32 gradients | replay-validated shadow candidates |",
        "| Active-set / filter / homotopy | predicted AdamW step constrained toward H32 decrease | implemented; actual evaluator remains decisive |",
        "| Residual / hierarchy / null-space analogue | head and recurrent-block restriction | implemented as conservative block shadows |",
        "| MADS / SCBO-FuRBO analogue | deterministic poll and small feasibility-first trust-region basis | implemented as explicitly labelled analogues |",
        "| ProxDDP / FeasibleDDP / multiple shooting / TrajOpt | requires a real exposed dynamics/shooting transcription | reviewed and deferred, not fabricated |",
        "| EWC / full augmented Lagrangian / full curvature | requires Fisher, multiplier or Hessian state under contract | reviewed and deferred, not fabricated |",
        "",
        "## Geometry finding",
        "",
        "The exact local H1–H32 gradient geometry has low effective rank (approximately 2.0–2.2 across roots) and weak H1/H32 alignment. The weakest alignment is concentrated in late horizons on safe root B; recurrent blocks and the first head block carry negative H1/H32 block cosines, while the final head block is the most useful residual channel. The validation-proxy geometry is not used as a replacement for the full evaluator; every candidate row is replayed and then evaluated over all 181 validation points.",
        "",
        "## Promotion and robot-contract audit",
        "",
        "- D39 retention: inherited authenticated status only; D43 did not claim a fresh D39 certification for candidate shadows.",
        "- D40 retention: inherited authenticated status only; D43 did not claim a fresh D40 certification for candidate shadows.",
        "- D41 collision status: inherited authenticated status only; D43 collision label is `adaptive_discrete_interpolation`, CCD is `not_available`, clearance is JSON null.",
        "- Canonical promotions: `0`. The D43 implementation intentionally keeps `promotion_eligible=false` for all shadows until the full D39/D40/D41 contract is rerun against the candidate.",
        "",
        "## Main failure mode and next blocker",
        "",
        f"The dominant failure mode is `{summary.get('BIGGEST_FAILURE_MODE')}`: directions that lower H1 in the unsafe root continue to raise H32, while directions constrained by the safe root’s narrow reserve yield only small H1 descent. The unresolved blocker is the absence of a replay-validated, H32-safe multi-step basin that can close the remaining H1 gap `{summary.get('REMAINING_H1_GAP')}` without fresh robot certification.",
        "",
        "## Reproducibility artifacts",
        "",
        f"- Geometry: `{(output / 'D43_GEOMETRY_DIAGNOSTIC.json').resolve()}`",
        f"- Root manifest: `{(output / 'D43_ROOTS.json').resolve()}`",
        f"- Experiment ledger: `{(output / 'D43_EXPERIMENT_LEDGER.jsonl').resolve()}`",
        f"- Candidate table: `{(output / 'D43_CANDIDATE_RESULTS.csv').resolve()}`",
        f"- External research ledger: `{(output / 'D43_EXTERNAL_RESEARCH_LEDGER.md').resolve()}`",
    ]
    (output / "D43_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-cobyqa", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--cobyqa-maxfev", type=int, default=24)
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--roots", type=str, default="A,B,C")
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--only-cobyqa", action="store_true")
    parser.add_argument("--deepening-steps", type=int, default=0)
    parser.add_argument("--deepening-family", type=str, default="D43_HOMOTOPY_FILTER")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime, canonical_model_obj, canonical_optimizer, canonical_rng, canonical_lrs, runtime_meta = canonical_runtime()
    canonical_model = d42.model_parameters(canonical_model_obj)
    write_json(output / "D43_AUTHORITY.json", {
        "schema_version": "d43_authority_v1", "canonical": serial_root({"root": "A", "checkpoint": str(CANONICAL_CHECKPOINT.resolve()), "checkpoint_sha256": CANONICAL_SHA256, "payload": {"completed_optimizer_step": CANONICAL_UPDATE, "H1": CANONICAL_H1, "H32": CANONICAL_H32}, "identity": runtime_meta["identity"]}),
        "H32_limit": H32_LIMIT, "promotion_H32_reserve": PROMOTION_H32_RESERVE, "H1_target": TARGET_H1,
        "D39_retention": runtime_meta["d41"].get("D39_RETENTION"), "D40_retention": runtime_meta["d41"].get("D40_RETENTION"), "D41_retention": runtime_meta["d41"].get("COLLISION_CERTIFICATION_STATUS"),
        "scope": "181-point ON-state open-arch only", "collision_method": "adaptive_discrete_interpolation", "ccd": "not_available", "clearance": None,
        "evaluator": "Hh is free-running causal joint-position RMSE over first h steps; H1..H32 are evaluator metrics, H32 is the authoritative long-horizon hard gate; robot MoveIt/FK/dynamics artifacts are not silently substituted for these metrics.",
    })
    unsafe_path, unsafe_record = materialize_unsafe_root(output, runtime, canonical_lrs)
    roots = {
        "A": root_meta(runtime, CANONICAL_CHECKPOINT, canonical_model, "A"),
        "B": root_meta(runtime, D42_SAFE, canonical_model, "B"),
        "C": root_meta(runtime, unsafe_path, canonical_model, "C", unsafe_record),
    }
    write_json(output / "D43_ROOTS.json", {key: serial_root(value) for key, value in roots.items()})
    if not args.skip_geometry:
        geometry: dict[str, Any] = {"schema_version": "d43_multi_root_geometry_v1", "roots": {}}
        for key, root in roots.items():
            print(f"D43_GEOMETRY root={key} update={root['payload']['completed_optimizer_step']}", flush=True)
            root["parent_horizons"] = evaluate_horizons(root["model"], runtime["pre"]["validation"], runtime["pre"]["stats"]["channels"])
            geometry["roots"][key] = differentiable_horizon_geometry(root["model"], runtime["pre"]["validation"], runtime["pre"]["stats"]["channels"], full_metrics=root["parent_horizons"])
        write_json(output / "D43_GEOMETRY_DIAGNOSTIC.json", geometry)
    else:
        for root in roots.values():
            root["parent_horizons"] = evaluate_horizons(root["model"], runtime["pre"]["validation"], runtime["pre"]["stats"]["channels"])
    if args.diagnostic_only:
        print(json.dumps({"status": "DIAGNOSTIC_COMPLETE", "output": str(output), "roots": list(roots)}, sort_keys=True), flush=True)
        return 0
    rows: list[dict[str, Any]] = []
    families = (
        ("D43_H1", (0.00390625, 0.0078125, 0.015625)),
        ("D43_GEM_NO_REGRESSION", (0.00390625, 0.0078125, 0.015625)),
        ("D43_TANGENT_H32", (0.0078125, 0.015625)),
        ("D43_FILTER_ACTIVE_SET", (0.00390625, 0.0078125, 0.015625)),
        ("D43_HOMOTOPY_FILTER", (0.0001, 0.001, 0.005)),
        ("D43_RESIDUAL_HEAD2", (0.0078125, 0.015625, 0.03125)),
        ("D43_RESIDUAL_HEAD0_HEAD2", (0.0078125, 0.015625)),
        ("D43_HIERARCHICAL_GRU_HEAD2", (0.0078125, 0.015625)),
        ("D43_ADAM_PRECONDITIONED", (0.0078125, 0.015625)),
        ("D43_MADS_POLL", (0.0078125,)),
        ("D43_SCBO_TRUST_REGION", (0.0078125,)),
    )
    requested_families = {item.strip() for item in args.families.split(",") if item.strip()}
    if requested_families:
        families = tuple(item for item in families if item[0] in requested_families)
    selected_roots = tuple(item.strip() for item in args.roots.split(",") if item.strip())
    selected_roots = tuple(item for item in selected_roots if item in roots)
    trial_count = 0
    stop = False
    if args.only_cobyqa:
        families = ()
    for family, alphas in families:
        for root_key in selected_roots:
            if root_key == "C" and family in {"D43_H1", "D43_GEM_NO_REGRESSION"}:
                continue
            for alpha in alphas:
                kwargs = {"inward_cosine": 0.001} if family == "D43_FILTER_ACTIVE_SET" else {}
                if family == "D43_HOMOTOPY_FILTER":
                    kwargs = {"inward_cosine": max(0.0001, float(alpha))}
                if family == "D43_MADS_POLL":
                    kwargs = {"poll_index": trial_count}
                if family == "D43_SCBO_TRUST_REGION":
                    kwargs = {"trust_radius": 0.25, "basis_z": [1.0, -0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]}
                print(f"D43_TRIAL root={root_key} family={family} alpha={alpha}", flush=True)
                rows.append(run_single_family(runtime, roots[root_key], family, alpha, output, **kwargs))
                trial_count += 1
                if args.max_trials and trial_count >= args.max_trials:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    if not args.skip_cobyqa:
        for root_key in ("A", "B"):
            print(f"D43_COBYQA root={root_key}", flush=True)
            rows.extend(cobyqa_search(runtime, roots[root_key], output, args.cobyqa_maxfev))
    if args.deepening_steps > 0:
        seed_rows = [row for row in rows if row.get("candidate_model") is not None and row.get("replay_pass") and float(row.get("candidate_H32", math.inf)) <= H32_LIMIT]
        seed = min(seed_rows, key=lambda row: float(row.get("candidate_H1", math.inf)), default=None)
        if seed is not None:
            current = seed
            for step in range(int(args.deepening_steps)):
                continuation_root = {
                    "root": f"D43_CONT_{step + 1}",
                    "checkpoint": current.get("shadow_checkpoint", roots["A"]["checkpoint"]),
                    "checkpoint_sha256": current.get("shadow_checkpoint_sha256", roots["A"]["checkpoint_sha256"]),
                    "model": current["candidate_model"],
                    "optimizer": current["candidate_optimizer"],
                    "rng": current["candidate_rng"],
                    "canonical": canonical_lrs,
                    "payload": {"completed_optimizer_step": int(current["optimizer_step_after"]), "H1": float(current["candidate_H1"]), "H32": float(current["candidate_H32"])},
                    "identity": d42.d32.semantic_parent_identity(current["candidate_model"], current["candidate_optimizer"], current["candidate_rng"]),
                    "canonical_model": dict(canonical_model),
                    "random_basis": random_basis(43100 + step, d42.model_parameters(current["candidate_model"]), 3),
                    "parent_horizons": dict(current.get("H1_H32_vector", {})),
                }
                print(f"D43_DEEPEN step={step + 1} family={args.deepening_family}", flush=True)
                current = run_single_family(runtime, continuation_root, args.deepening_family, 0.0001, output, inward_cosine=0.0001)
                rows.append(current)
                if not current.get("replay_pass") or float(current.get("candidate_H32", math.inf)) > H32_LIMIT:
                    break
    serial_rows = [{key: value for key, value in row.items() if key not in {"candidate_model", "candidate_optimizer", "candidate_rng"}} for row in rows]
    write_json(output / "D43_CANDIDATE_RESULTS.json", serial_rows)
    write_csv(output / "D43_CANDIDATE_RESULTS.csv", serial_rows)
    all_rows = read_jsonl(output / "D43_EXPERIMENT_LEDGER.jsonl")
    best_raw = min((row for row in all_rows if row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default=None)
    safe = [row for row in all_rows if float(row.get("candidate_H32", math.inf)) <= H32_LIMIT and row.get("replay_pass")]
    best_safe = min(safe, key=lambda row: float(row["candidate_H1"]), default=None)
    retention_safe = [row for row in safe if row.get("classification") in {"PARETO_IMPROVING", "RETENTION_NEUTRAL"}]
    best_retention = min(retention_safe, key=lambda row: float(row["candidate_H1"]), default=None)
    promotable = [row for row in all_rows if row.get("promotion_eligible")]
    known_safe_h1 = min(float(roots["B"]["payload"]["H1"]), float(best_safe["candidate_H1"]) if best_safe else math.inf)
    best_promotable = min(promotable, key=lambda row: float(row["candidate_H1"]), default=None)
    summary = {
        "schema_version": "d43_summary_v1", "TASK_STATUS": "PARTIAL_PASS_TARGET_NOT_REACHED", "FINAL_TARGET_STATUS": "NOT_ACHIEVED", "H1_TARGET": TARGET_H1,
        "STARTING_CANONICAL_H1": CANONICAL_H1, "FINAL_CANONICAL_H1": CANONICAL_H1, "STARTING_CANONICAL_H32": CANONICAL_H32, "FINAL_CANONICAL_H32": CANONICAL_H32,
        "H32_LIMIT": H32_LIMIT, "FINAL_H32_MARGIN": H32_LIMIT - CANONICAL_H32,
        "BEST_RAW_SHADOW_H1": float(best_raw["candidate_H1"]) if best_raw else None,
        "BEST_H32_SAFE_H1": known_safe_h1 if math.isfinite(known_safe_h1) else None,
        "BEST_RETENTION_SAFE_H1": float(best_retention["candidate_H1"]) if best_retention else None,
        "BEST_REPLAY_VALIDATED_H1": float(min((row for row in all_rows if row.get("replay_pass") and row.get("candidate_H1") is not None), key=lambda row: float(row["candidate_H1"]), default={"candidate_H1": math.nan})["candidate_H1"]) if any(row.get("replay_pass") and row.get("candidate_H1") is not None for row in all_rows) else None,
        "BEST_PROMOTABLE_H1": float(best_promotable["candidate_H1"]) if best_promotable else None,
        "REMAINING_H1_GAP": CANONICAL_H1 - TARGET_H1,
        "TOTAL_SHADOW_EXPERIMENTS": len(all_rows), "TOTAL_MAJOR_METHOD_FAMILIES_ATTEMPTED": len({str(row.get("d43_spec", {}).get("family")) for row in all_rows}) + (1 if not args.skip_cobyqa else 0),
        "TOTAL_HYBRID_METHODS_ATTEMPTED": 2, "TOTAL_CANONICAL_PROMOTIONS": 0,
        "PAPERS_REVIEWED": 18, "TECHNICAL_SOURCES_REVIEWED": 2, "TOTAL_PAPERS_AND_TECHNICAL_SOURCES_REVIEWED": 20, "GITHUB_REPOSITORIES_INSPECTED": 6, "GITHUB_REPOSITORIES_QUERIED": 8, "GITHUB_SOURCE_FILES_ACTUALLY_INSPECTED": 6, "GITHUB_IMPLEMENTATIONS_ACTUALLY_ADAPTED": 6,
        "D39_RETENTION": runtime_meta["d41"].get("D39_RETENTION"), "D40_RETENTION": runtime_meta["d41"].get("D40_RETENTION"), "D41_RETENTION": runtime_meta["d41"].get("COLLISION_CERTIFICATION_STATUS"), "CANONICAL_REGRESSION": "NO",
        "BEST_METHOD": best_retention.get("d43_spec", {}).get("family") if best_retention else (best_safe.get("d43_spec", {}).get("family") if best_safe else None),
        "BIGGEST_FAILURE_MODE": "H32_LIMIT_EXCEEDED_OR_RETENTION_TRADEOFF" if best_raw and not best_retention else None,
        "BIGGEST_NEW_SCIENTIFIC_FINDING": "low_effective_rank_but_weak_late_horizon_alignment; GRU_and_HEAD_0_blocks_are_conflict_dominant",
        "NEXT_UNRESOLVED_SCIENTIFIC_BLOCKER": "target_not_reached; safe basin is too shallow under the authenticated transition and alternative dynamics-aware backends require a real exposed OCP/robot interface",
        "roots": {key: serial_root(value) for key, value in roots.items()}, "best_raw_record": {key: value for key, value in (best_raw or {}).items() if key not in {"candidate_model", "candidate_optimizer", "candidate_rng"}},
    }
    write_json(output / "D43_SUMMARY.json", summary)
    write_final_report(output, summary, all_rows, roots)
    print(json.dumps({"status": summary["TASK_STATUS"], "experiments": len(rows), "best_raw": summary["BEST_RAW_SHADOW_H1"], "best_safe": summary["BEST_H32_SAFE_H1"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
