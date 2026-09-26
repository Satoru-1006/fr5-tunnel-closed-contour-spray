"""D44 realized-response trust-region campaign.

This is a shadow-only experiment around the authenticated D32 AdamW transition.
The proposal variables are coefficients in an adaptive, robot-aware parameter
subspace, but the response oracle is always the real transition followed by
the full causal H1..H32 evaluator.  Surrogates are used only to rank the next
shadow query; they are never used as a safety certificate.

The campaign deliberately does not mutate the D35 canonical checkpoint or any
D39/D40/D41 artifact.  Candidate robot metrics remain JSON null unless a real
robot backend is run for that candidate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_d42_persistent_h1_breakthrough_campaign as d42
from tools import stage3_h13_d43_constrained_breakthrough_campaign as d43


TARGET_H1 = 5.0e-5
H32_LIMIT = 0.01856902565856056
PROMOTION_H32_RESERVE = 7.5e-5
D44_FULLY_READ_PAPERS = 4
D44_FULLY_INSPECTED_REPOSITORIES = 1
CANONICAL_H1 = 7.089328839013259e-05
CANONICAL_H32 = 0.01849100619381173
CANONICAL_UPDATE = 480
CANONICAL_CHECKPOINT = ROOT / "outputs" / "stage3_h13_d35_permanent_champion" / "checkpoints" / "committed_update_480.pt"
D42_SAFE = ROOT / "outputs" / "stage3_h13_d42_persistent_h1_breakthrough_campaign" / "D42_recharge_shadow_update_481.pt"
D43_UNSAFE = ROOT / "outputs" / "stage3_h13_d43_constrained_breakthrough_campaign" / "roots" / "D43_root_C_D42_unsafe_update_482.pt"
DEFAULT_OUTPUT = ROOT / "outputs" / "stage3_h13_d44_realized_response_campaign"
HORIZONS = tuple(range(1, 33))
OUTPUT_NAMES = tuple(f"H{h}" for h in HORIZONS)
ALPHA_LEVELS = (0.00048828125, 0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625)
PARAMETER_NAMES = tuple(d43.ALL_NAMES)


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite:{result}")
    return result


def vector_norm(values: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(max(0.0, sum(finite(torch.sum(value.detach().double().square())) for value in values.values())))


def vector_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return finite(sum(torch.sum(left[name].double() * right[name].double()) for name in PARAMETER_NAMES))


def unit(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    scale = 1.0 / max(vector_norm(values), 1.0e-30)
    return {name: values[name].detach().cpu().contiguous().clone() * scale for name in PARAMETER_NAMES}


def add_scaled(target: dict[str, torch.Tensor], source: Mapping[str, torch.Tensor], amount: float) -> None:
    for name in PARAMETER_NAMES:
        target[name] = target[name] + float(amount) * source[name]


def parameters(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def evaluate_horizons(model: Any, runtime: Mapping[str, Any]) -> dict[str, float]:
    return d43.evaluate_horizons(model, runtime["pre"]["validation"], runtime["pre"]["stats"]["channels"])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


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


def load_root(runtime: Mapping[str, Any], path: Path, name: str, canonical_model: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"root_checkpoint_missing:{path}")
    model, optimizer, rng, canonical, payload = d43.load_state(runtime, path)
    horizons = evaluate_horizons(model, runtime)
    return {
        "root": name,
        "checkpoint": str(path.resolve()),
        "model": model,
        "optimizer": optimizer,
        "rng": copy.deepcopy(rng),
        "canonical": canonical,
        "payload": payload,
        "identity": d42.d32.semantic_parent_identity(model, optimizer, rng),
        "parent_horizons": horizons,
        "canonical_model": dict(canonical_model),
        "initial_horizons": dict(horizons),
        "history_deltas": [],
        "accepted_count": 0,
    }


def relative_violation(horizons: Mapping[str, float], reference: Mapping[str, float], funnel_reference: Mapping[str, float] | None = None) -> float:
    """Normalized filter violation; objective and feasibility stay separate."""
    h32_term = max(0.0, float(horizons["32"]) - H32_LIMIT) / max(H32_LIMIT, 1.0e-12)
    ref = funnel_reference or reference
    retained = []
    for h in range(2, 32):
        key = str(h)
        excess = max(0.0, float(horizons[key]) - float(ref[key]) - 1.0e-10)
        retained.append(excess / max(abs(float(ref[key])), 1.0e-8))
    return float(h32_term + (sum(retained) / max(len(retained), 1)))


def filter_pair(horizons: Mapping[str, float], reference: Mapping[str, float], funnel_reference: Mapping[str, float] | None = None) -> tuple[float, float]:
    return float(horizons["1"]), relative_violation(horizons, reference, funnel_reference)


def model_features(z: Sequence[float], alpha: float, dimension: int) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    if values.size != dimension:
        raise ValueError(f"coefficient_dimension:{values.size}:{dimension}")
    base = max(ALPHA_LEVELS[3], 1.0e-12)
    return np.concatenate((values, np.asarray([math.log2(max(float(alpha), 1.0e-12) / base)], dtype=np.float64)))


def quadratic_features(x: np.ndarray) -> np.ndarray:
    products = [x[i] * x[j] for i in range(len(x)) for j in range(i, len(x))]
    return np.concatenate((np.asarray([1.0]), x, np.asarray(products, dtype=np.float64)))


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1.0e-6) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0, y.shape[1]), dtype=np.float64)
    gram = x.T @ x
    regularizer = np.eye(x.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = ridge * 1.0e-3
    return np.linalg.solve(gram + regularizer, x.T @ y)


@dataclass
class SurrogateModel:
    name: str
    predictor: Any
    cv_error: float = math.inf

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.predictor(x), dtype=np.float64)


@dataclass
class SurrogateEnsemble:
    output_dim: int
    input_dim: int
    models: list[SurrogateModel] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    training_count: int = 0

    def fit(self, observations: Sequence[Mapping[str, Any]], trust_radius: float) -> None:
        self.models = []
        self.weights = {}
        self.training_count = len(observations)
        if not observations:
            return
        x = np.asarray([row["x"] if "x" in row else row["d44_model_x"] for row in observations], dtype=np.float64)
        y = np.asarray([row["delta"] if "delta" in row else row["d44_model_delta"] for row in observations], dtype=np.float64)
        scale = np.maximum(np.std(y, axis=0), 1.0e-10)

        def linear_design(values: np.ndarray) -> np.ndarray:
            return np.column_stack((np.ones(len(values)), values))

        def make_linear(train_x: np.ndarray, train_y: np.ndarray) -> Any:
            coef = ridge_fit(linear_design(train_x), train_y / scale)
            return lambda value: linear_design(np.asarray(value, dtype=np.float64).reshape(1, -1)) @ coef * scale

        def make_quadratic(train_x: np.ndarray, train_y: np.ndarray) -> Any:
            design = np.vstack([quadratic_features(value) for value in train_x])
            coef = ridge_fit(design, train_y / scale, ridge=1.0e-4)
            return lambda value: quadratic_features(np.asarray(value, dtype=np.float64)) @ coef * scale

        def make_rbf(train_x: np.ndarray, train_y: np.ndarray) -> Any:
            length = max(float(trust_radius), 0.25)

            def predict(value: np.ndarray) -> np.ndarray:
                distances = np.linalg.norm(train_x - np.asarray(value, dtype=np.float64)[None, :], axis=1)
                kernel = np.exp(-0.5 * (distances / length) ** 2)
                if float(kernel.sum()) <= 1.0e-12:
                    return np.mean(train_y, axis=0)
                return (kernel[:, None] * train_y).sum(axis=0) / kernel.sum()

            return predict

        constructors: list[tuple[str, Any]] = [("local_linear", make_linear), ("rbf", make_rbf)]
        if len(observations) >= max(4, self.input_dim + 1):
            constructors.insert(1, ("local_quadratic", make_quadratic))
        for name, constructor in constructors:
            self.models.append(SurrogateModel(name, constructor(x, y)))
        for model in self.models:
            if len(observations) >= 3:
                errors = []
                for index in range(len(observations)):
                    train_indices = [j for j in range(len(observations)) if j != index]
                    train_x, train_y = x[train_indices], y[train_indices]
                    if model.name == "local_linear":
                        trial = make_linear(train_x, train_y)
                    elif model.name == "local_quadratic":
                        trial = make_quadratic(train_x, train_y)
                    else:
                        trial = make_rbf(train_x, train_y)
                    prediction = np.asarray(trial(x[index]), dtype=np.float64).reshape(-1)
                    errors.append(float(np.mean(np.abs((prediction - y[index]) / scale))))
                model.cv_error = float(np.mean(errors))
            else:
                model.cv_error = 1.0
            self.weights[model.name] = 1.0 / max(model.cv_error, 1.0e-6)
        total = sum(self.weights.values())
        if total > 0.0:
            self.weights = {name: value / total for name, value in self.weights.items()}

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if not self.models:
            zeros = np.zeros(self.output_dim, dtype=np.float64)
            return zeros, zeros, {}
        predictions = {model.name: model.predict(x).reshape(-1) for model in self.models}
        weights = self.weights or {name: 1.0 / len(predictions) for name in predictions}
        mean = sum(float(weights.get(name, 0.0)) * value for name, value in predictions.items())
        spread = np.sqrt(sum(float(weights.get(name, 0.0)) * (value - mean) ** 2 for name, value in predictions.items()))
        return mean, spread, {model.name: float(model.cv_error) for model in self.models}


def deterministic_candidates(dimension: int, radius: float, seed: int, attempt: int) -> list[tuple[np.ndarray, float, str]]:
    """Generate a positive-spanning, curved, and restoration-aware proposal set."""
    unit_vectors = [
        np.eye(dimension, dtype=np.float64)[0],
        np.eye(dimension, dtype=np.float64)[1],
        np.eye(dimension, dtype=np.float64)[2],
        np.eye(dimension, dtype=np.float64)[3],
        np.eye(dimension, dtype=np.float64)[4],
        -np.eye(dimension, dtype=np.float64)[0],
        -np.eye(dimension, dtype=np.float64)[1],
    ]
    combos = [
        np.asarray([1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.75, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0]),
        np.asarray([1.0, 0.5, 0.0, 0.0, 0.0, -0.25, 0.25, 0.0]),
    ]
    rng = np.random.default_rng(int(seed) + int(attempt) * 7919)
    random_points = [rng.normal(0.0, 0.55, dimension) for _ in range(8)]
    candidates: list[tuple[np.ndarray, float, str]] = []
    for index, direction in enumerate(unit_vectors + combos + random_points):
        padded = np.zeros(dimension, dtype=np.float64)
        padded[: min(dimension, len(direction))] = np.asarray(direction)[:dimension]
        length = max(float(np.linalg.norm(padded)), 1.0e-12)
        z = padded / length * min(float(radius), 1.0)
        alpha = ALPHA_LEVELS[(index + attempt) % len(ALPHA_LEVELS)]
        candidates.append((z, alpha, f"poll_{index:02d}"))
    return candidates


def basis_from_state(track: Mapping[str, Any], h1: Mapping[str, torch.Tensor], h32: Mapping[str, torch.Tensor], postclip: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Mapping[str, Any]], references: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
    """Build an adaptive basis from gradients, routes, and realized motion."""
    atoms: list[dict[str, torch.Tensor]] = [d43.unit(h1), d43.unit(h32), d43.unit(postclip)]
    labels = ["H1_descent_generator", "H32_restoration_generator", "postclip_optimizer_generator"]
    current_delta = {name: before[name] - track["canonical_model"][name] for name in PARAMETER_NAMES}
    if vector_norm(current_delta) > 1.0e-12:
        atoms.append(d43.unit({name: -current_delta[name] for name in PARAMETER_NAMES}))
        labels.append("canonical_route_return")
    else:
        atoms.append({name: torch.zeros_like(h1[name]) for name in PARAMETER_NAMES})
        labels.append("canonical_route_return_unavailable")
    # Keep the two authenticated safe-root routes in the active eight-vector
    # subspace.  Retaining route_toward_C while truncating the realized-motion
    # reversal was an accidental prioritization of the unsafe basin; the
    # reversal is the more useful history-dependent restoration atom.
    for ref_name in ("A", "B"):
        delta = references[ref_name]
        direction = {name: -(before[name] - delta[name]) for name in PARAMETER_NAMES}
        if vector_norm(direction) > 1.0e-12:
            atoms.append(d43.unit(direction))
            labels.append(f"route_toward_{ref_name}")
        else:
            atoms.append({name: torch.zeros_like(h1[name]) for name in PARAMETER_NAMES})
            labels.append(f"route_toward_{ref_name}_unavailable")
    named_momentum = {name: before_optimizer[name].get("exp_avg", torch.zeros_like(h1[name])).detach().cpu() for name in PARAMETER_NAMES}
    atoms.append(d43.unit(named_momentum) if vector_norm(named_momentum) > 1.0e-12 else {name: torch.zeros_like(h1[name]) for name in PARAMETER_NAMES})
    labels.append("inherited_adam_momentum")
    if track.get("history_deltas"):
        delta = track["history_deltas"][-1]
        atoms.append(d43.unit({name: -delta[name] for name in PARAMETER_NAMES}))
        labels.append("last_realized_motion_reversal")
    else:
        atoms.append({name: torch.zeros_like(h1[name]) for name in PARAMETER_NAMES})
        labels.append("last_realized_motion_unavailable")
    return atoms[:8], labels[:8]


def run_realized_trial(runtime: Mapping[str, Any], track: Mapping[str, Any], references: Mapping[str, Mapping[str, torch.Tensor]], coefficients: Sequence[float], alpha: float) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    spec = {
        "family": "D44_REALIZED_RESPONSE_TR",
        "base": "H1",
        "alpha": float(alpha),
        "target_norm": 1.0,
        "coefficients": [float(value) for value in coefficients],
    }
    original = d42.d32.build_presented_gradient
    holder: dict[str, Any] = {}

    def builder(inner_spec: Mapping[str, Any], canonical_postclip: Mapping[str, torch.Tensor], h1_gradients: Mapping[str, torch.Tensor], h32_gradients: Mapping[str, torch.Tensor], before_parameters: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Mapping[str, Any]], effective_lr: float) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        atoms, labels = basis_from_state(track, h1_gradients, h32_gradients, canonical_postclip, before_parameters, before_optimizer, references)
        direction = {name: torch.zeros_like(h1_gradients[name]) for name in PARAMETER_NAMES}
        for index, coefficient in enumerate(inner_spec["coefficients"]):
            if index >= len(atoms):
                break
            add_scaled(direction, atoms[index], float(coefficient))
        if vector_norm(direction) <= 1.0e-12:
            direction = d43.unit(h1_gradients)
            labels = ["H1_descent_generator_fallback"]
        presented = d43.unit(direction)
        holder["basis_labels"] = labels
        holder["basis_cosines"] = {
            "presented_h1": d43.cosine(presented, h1_gradients),
            "presented_h32": d43.cosine(presented, h32_gradients),
        }
        return presented, {
            "generator": "realized_response_adaptive_subspace",
            "basis_labels": labels,
            "basis_dimension": len(atoms),
            "basis_cosines": holder["basis_cosines"],
            "coefficients": [float(value) for value in inner_spec["coefficients"]],
        }

    d42.d32.build_presented_gradient = builder
    try:
        parent_update = int(track["payload"]["completed_optimizer_step"])
        record, candidate_model, candidate_optimizer, candidate_rng = d42.d32.run_trial(
            runtime,
            track["model"], track["optimizer"], track["rng"], track["canonical"],
            track["identity"], parent_update, parent_update + 1,
            float(track["parent_horizons"]["1"]), float(track["parent_horizons"]["32"]), spec,
        )
    finally:
        d42.d32.build_presented_gradient = original
    record["d44_basis_labels"] = holder.get("basis_labels", [])
    record["d44_basis_cosines"] = holder.get("basis_cosines", {})
    return record, candidate_model, candidate_optimizer, candidate_rng


def full_candidate_record(record: dict[str, Any], candidate_model: Any, runtime: Mapping[str, Any], track: Mapping[str, Any], coefficients: Sequence[float], alpha: float, predicted: np.ndarray, uncertainty: np.ndarray, model_errors: Mapping[str, float], trust_radius: float, funnel_limit: float, attempt: int) -> dict[str, Any]:
    actual = evaluate_horizons(candidate_model, runtime)
    parent = track["parent_horizons"]
    delta = np.asarray([float(actual[str(h)]) - float(parent[str(h)]) for h in HORIZONS], dtype=np.float64)
    predicted_horizons = {str(h): float(parent[str(h)] + predicted[h - 1]) for h in HORIZONS}
    actual_violation = relative_violation(actual, parent, track.get("funnel_reference"))
    predicted_violation = relative_violation(predicted_horizons, parent, track.get("funnel_reference"))
    record.update({
        "d44_attempt": int(attempt),
        "d44_coefficients": [float(value) for value in coefficients],
        "d44_alpha": float(alpha),
        "d44_trust_radius": float(trust_radius),
        "d44_funnel_limit": float(funnel_limit),
        "d44_predicted_response": {str(h): float(predicted[h - 1]) for h in HORIZONS},
        "d44_predicted_horizons": predicted_horizons,
        "d44_response_uncertainty": {str(h): float(uncertainty[h - 1]) for h in HORIZONS},
        "d44_model_cv_error": dict(model_errors),
        "d44_actual_horizons": actual,
        "d44_actual_response": {str(h): float(delta[h - 1]) for h in HORIZONS},
        "d44_predicted_feasibility_violation": float(predicted_violation),
        "d44_actual_feasibility_violation": float(actual_violation),
        "d44_prediction_error_l2": float(np.linalg.norm(delta - predicted)),
        "d44_prediction_error_h1": float(delta[0] - predicted[0]),
        "d44_prediction_error_h32": float(delta[-1] - predicted[-1]),
        "robot_metrics": {
            "evaluated": False,
            "end_effector_tracking_error": None,
            "terminal_error": None,
            "collision_clearance": None,
            "self_collision_clearance": None,
            "joint_limit_margin": None,
            "joint_velocity_margin": None,
            "joint_acceleration_margin": None,
            "jerk": None,
            "torque_effort_margin": None,
            "singularity_margin": None,
            "trajectory_smoothness": None,
            "rollout_stability": None,
            "disturbance_sensitivity": None,
            "recoverability": None,
            "safety_filter_intervention": None,
            "dynamic_feasibility": None,
            "replay_validity": bool(record.get("deterministic_replay_consistency_pass", False)),
            "availability_reason": "D44 neural candidate changed causal predictor state; no candidate-specific MoveIt/FK/dynamics trajectory adapter was exposed",
        },
    })
    record["d44_objective_h1"] = float(actual["1"])
    record["d44_filter_class"] = "RESTORATION" if float(track["current_violation"]) > 1.0e-12 else "OPTIMALITY"
    record["d44_actual_h32_safe"] = bool(float(actual["32"]) <= H32_LIMIT)
    record["d44_actual_h32_reserve_pass"] = bool(H32_LIMIT - float(actual["32"]) >= PROMOTION_H32_RESERVE)
    record["d44_full_retention_neutral"] = bool(all(float(actual[str(h)]) <= float(parent[str(h)]) + 1.0e-10 for h in range(2, 32)))
    return record


def candidate_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return (float(row.get("d44_actual_feasibility_violation", math.inf)), float(row.get("candidate_H1", math.inf)), float(row.get("candidate_H32", math.inf)))


def accept_filter(row: Mapping[str, Any], track: Mapping[str, Any]) -> tuple[bool, str]:
    actual_violation = float(row["d44_actual_feasibility_violation"])
    current_violation = float(track["current_violation"])
    h1 = float(row["candidate_H1"])
    current_h1 = float(track["parent_horizons"]["1"])
    h32 = float(row["candidate_H32"])
    if not bool(row.get("deterministic_replay_consistency_pass")):
        return False, "REPLAY_FAILURE"
    if not bool(row.get("finite_state_pass")):
        return False, "FINITE_STATE_FAILURE"
    if current_violation > 1.0e-12:
        if actual_violation < current_violation - 1.0e-10 or h32 < float(track["parent_horizons"]["32"]) - 1.0e-8:
            return True, "FEASIBILITY_RESTORATION"
        return False, "RESTORATION_NO_PROGRESS"
    if h32 <= H32_LIMIT and h1 < current_h1 - 1.0e-10 and actual_violation <= float(track["funnel_limit"]) + 1.0e-12:
        return True, "FILTER_OPTIMALITY"
    if h32 <= H32_LIMIT and h1 < current_h1 - 1.0e-10 and bool(row.get("d44_full_retention_neutral")):
        return True, "RETENTION_NEUTRAL_OPTIMALITY"
    return False, "FILTER_REJECTED"


def save_shadow(output: Path, row: Mapping[str, Any], model: Any, optimizer: Any, rng: Mapping[str, Any], parent_checkpoint: str, canonical: Sequence[float], label: str) -> Path:
    path = output / "shadow_checkpoints" / f"{label}.pt"
    payload = d42.d32.checkpoint_payload(model, optimizer, rng, dict(row), parent_checkpoint, canonical)
    payload.update({"scientific_state": "D44_REALIZED_RESPONSE_SHADOW", "promotion_state": "NOT_PROMOTED", "resumable": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def serial_track(track: Mapping[str, Any]) -> dict[str, Any]:
    payload = track["payload"]
    return {
        "root": track["root"],
        "checkpoint": track["checkpoint"],
        "completed_optimizer_step": int(payload["completed_optimizer_step"]),
        "H1": float(track["parent_horizons"]["1"]),
        "H32": float(track["parent_horizons"]["32"]),
        "H32_margin": H32_LIMIT - float(track["parent_horizons"]["32"]),
        "current_violation": float(track["current_violation"]),
        "trust_radius": float(track["trust_radius"]),
        "accepted_count": int(track["accepted_count"]),
    }


def run_track(runtime: Mapping[str, Any], track: dict[str, Any], references: Mapping[str, Mapping[str, torch.Tensor]], output: Path, max_evals: int, seed: int, existing_rows: Sequence[Mapping[str, Any]] = (), attempt_start: int = 0) -> list[dict[str, Any]]:
    dimension = 8
    ensemble = SurrogateEnsemble(len(HORIZONS), dimension + 1)
    rows: list[dict[str, Any]] = [dict(row) for row in existing_rows]
    if not existing_rows:
        track["funnel_reference"] = dict(track["parent_horizons"])
        track["current_violation"] = relative_violation(track["parent_horizons"], track["parent_horizons"], track["funnel_reference"])
        track["funnel_limit"] = max(1.0e-4, track["current_violation"] * 1.25)
        track["trust_radius"] = 1.0
    warmup = [
        (np.asarray([1.0, 0, 0, 0, 0, 0, 0, 0]), ALPHA_LEVELS[1], "H1_small"),
        (np.asarray([0, 1.0, 0, 0, 0, 0, 0, 0]), ALPHA_LEVELS[0], "H32_restore"),
        (np.asarray([1.0, 0.5, 0, 0, 0, 0, 0, 0]), ALPHA_LEVELS[1], "curved_H1_H32"),
        (np.asarray([1.0, 1.0, 0, 0, 0, 0, 0, 0]), ALPHA_LEVELS[2], "curved_equal"),
        (np.asarray([1.0, 0.75, 0, 0, 0, 0, 0.25, 0]), ALPHA_LEVELS[2], "momentum_compensated"),
        (np.asarray([0, 1.0, 0, 0.5, 0, 0, 0, 0]), ALPHA_LEVELS[1], "route_restoration"),
    ]
    for attempt in range(int(attempt_start), int(attempt_start) + int(max_evals)):
        if attempt < len(warmup):
            z, alpha, label = warmup[attempt]
        else:
            ensemble.fit(rows, track["trust_radius"])
            proposals = deterministic_candidates(dimension, track["trust_radius"], seed, attempt)
            if track["root"] == "C":
                # The first C campaign never directly polled the route atoms
                # that point from the unsafe state to the authenticated safe
                # roots.  Probe those curved-restoration chords explicitly;
                # they are parameter-space hypotheses and still require the
                # real transition/evaluator to validate them.
                for index, label in ((4, "route_C_to_A"), (5, "route_C_to_B")):
                    z = np.zeros(dimension, dtype=np.float64)
                    z[index] = min(float(track["trust_radius"]), 1.0)
                    proposals.append((z, ALPHA_LEVELS[0], label))
                z = np.zeros(dimension, dtype=np.float64)
                z[4], z[5] = 0.5 * min(float(track["trust_radius"]), 1.0), 0.5 * min(float(track["trust_radius"]), 1.0)
                proposals.append((z, ALPHA_LEVELS[1], "curved_C_to_A_to_B"))
            ranked: list[tuple[float, np.ndarray, float, str, np.ndarray, np.ndarray, dict[str, float]]] = []
            for proposal_z, proposal_alpha, proposal_label in proposals:
                x = model_features(proposal_z, proposal_alpha, dimension)
                predicted, uncertainty, model_errors = ensemble.predict(x)
                parent = track["parent_horizons"]
                projected = {str(h): float(parent[str(h)] + predicted[h - 1]) for h in HORIZONS}
                projected_violation = relative_violation(projected, parent, track["funnel_reference"])
                predicted_h1 = float(projected["1"])
                if track["current_violation"] > 1.0e-12:
                    score = projected_violation + 1.0e-4 * max(0.0, predicted_h1 - float(parent["1"])) + 0.25 * float(np.mean(uncertainty))
                else:
                    score = predicted_h1 + 0.02 * projected_violation + 0.05 * float(np.mean(uncertainty))
                ranked.append((score, proposal_z, proposal_alpha, proposal_label, predicted, uncertainty, model_errors))
            ranked.sort(key=lambda item: item[0])
            _, z, alpha, label, predicted, uncertainty, model_errors = ranked[0]
            # The first explicit route probes were only offered to the model
            # ranker and were never selected.  Force one complete, authenticated
            # realization of each route hypothesis on the next three C-track
            # attempts; the measured response still controls acceptance.
            if track["root"] == "C" and 13 <= attempt <= 15:
                route_index = attempt - 13
                route_specs = [
                    (4, 0.0, ALPHA_LEVELS[0], "route_C_to_A_forced"),
                    (5, 0.0, ALPHA_LEVELS[0], "route_C_to_B_forced"),
                    (4, 5, ALPHA_LEVELS[1], "curved_C_to_A_to_B_forced"),
                ]
                spec = route_specs[route_index]
                z = np.zeros(dimension, dtype=np.float64)
                if route_index < 2:
                    z[int(spec[0])] = min(float(track["trust_radius"]), 1.0)
                else:
                    z[4] = z[5] = 0.5 * min(float(track["trust_radius"]), 1.0)
                alpha = float(spec[2])
                label = str(spec[3])
                forced_x = model_features(z, alpha, dimension)
                predicted, uncertainty, model_errors = ensemble.predict(forced_x)
        if attempt < len(warmup):
            x = model_features(z, alpha, dimension)
            predicted, uncertainty, model_errors = ensemble.predict(x)
        record, candidate_model, candidate_optimizer, candidate_rng = run_realized_trial(runtime, track, references, z.tolist(), alpha)
        replay_record, replay_model, replay_optimizer, replay_rng = run_realized_trial(runtime, track, references, z.tolist(), alpha)
        replay = d42.d32.d27.replay_compare(record, replay_record, candidate_model, replay_model, candidate_optimizer, replay_optimizer, candidate_rng, replay_rng)
        record["d44_replay"] = replay
        record["deterministic_replay_consistency_pass"] = bool(replay["pass"])
        record["deterministic_replay_status"] = "PASS" if replay["pass"] else "FAIL"
        record["deterministic_replay_candidate_record_digest"] = replay["candidate_record_digest"]
        record["deterministic_replay_record_digest"] = replay["replay_record_digest"]
        record["replay_pass"] = bool(replay["pass"])
        full_candidate_record(record, candidate_model, runtime, track, z.tolist(), alpha, predicted, uncertainty, model_errors, track["trust_radius"], track["funnel_limit"], attempt)
        record["d44_proposal_label"] = label
        record["d44_track"] = track["root"]
        record["d44_parent_violation"] = float(track["current_violation"])
        record["d44_model_x"] = model_features(z, alpha, dimension).tolist()
        record["d44_model_delta"] = [float(value) for value in record["d44_actual_response"].values()]
        accepted, disposition = accept_filter(record, track)
        record["d44_filter_disposition"] = disposition
        record["d44_accepted_for_shadow_progression"] = bool(accepted)
        record["d44_trust_region_update"] = "unchanged"
        actual_reduction = -float(record["delta_H1"])
        predicted_reduction = -float(predicted[0])
        if attempt >= len(warmup) and abs(predicted_reduction) > 1.0e-12:
            rho = actual_reduction / predicted_reduction
            record["d44_rho"] = float(rho)
            if rho < 0.25:
                track["trust_radius"] = max(0.125, track["trust_radius"] * 0.5)
                record["d44_trust_region_update"] = "contracted_model_miss"
            elif rho > 0.75 and accepted:
                track["trust_radius"] = min(1.5, track["trust_radius"] * 1.5)
                record["d44_trust_region_update"] = "expanded_model_agreement"
        else:
            record["d44_rho"] = None
        if accepted:
            parent_model = track["model"]
            motion = {name: parameters(candidate_model)[name] - parameters(parent_model)[name] for name in PARAMETER_NAMES}
            track["history_deltas"].append(motion)
            accepted_path = save_shadow(output, record, candidate_model, candidate_optimizer, candidate_rng, track["checkpoint"], track["canonical"], f"{track['root']}_accepted_{attempt:03d}")
            record["shadow_checkpoint"] = str(accepted_path.resolve())
            track["checkpoint"] = str(accepted_path.resolve())
            track["model"], track["optimizer"], track["rng"] = candidate_model, candidate_optimizer, candidate_rng
            track["identity"] = d42.d32.semantic_parent_identity(candidate_model, candidate_optimizer, candidate_rng)
            track["payload"] = {
                "completed_optimizer_step": int(record["candidate_update"]),
                "H1": float(record["candidate_H1"]),
                "H32": float(record["candidate_H32"]),
            }
            track["parent_horizons"] = dict(record["d44_actual_horizons"])
            track["current_violation"] = float(record["d44_actual_feasibility_violation"])
            track["funnel_reference"] = dict(track["parent_horizons"])
            track["funnel_limit"] = max(1.0e-4, track["current_violation"] * 1.25)
            track["accepted_count"] += 1
        else:
            # The candidate is still scientifically valuable, but does not
            # become the next parent of this shadow track.
            if float(record.get("candidate_H1", math.inf)) < CANONICAL_H1:
                raw_path = save_shadow(output, record, candidate_model, candidate_optimizer, candidate_rng, track["checkpoint"], track["canonical"], f"{track['root']}_raw_{attempt:03d}")
                record["shadow_checkpoint"] = str(raw_path.resolve())
        record["d44_track_after"] = serial_track(track)
        serial = {key: value for key, value in record.items() if key not in {"candidate_model", "candidate_optimizer", "candidate_rng"}}
        append_jsonl(output / "D44_REALIZED_RESPONSE_LEDGER.jsonl", serial)
        rows.append(serial)
        print(f"D44 track={track['root']} attempt={attempt} label={label} H1={record.get('candidate_H1')} H32={record.get('candidate_H32')} class={record.get('d44_filter_disposition')} accepted={accepted} rho={record.get('d44_rho')}", flush=True)
    return rows


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resume_track(runtime: Mapping[str, Any], key: str, roots: Mapping[str, Mapping[str, Any]], existing_rows: Sequence[Mapping[str, Any]], canonical_model: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Reconstruct the last accepted state without replaying completed trials."""
    rows = [row for row in existing_rows if row.get("d44_track") == key]
    accepted = [row for row in rows if row.get("d44_accepted_for_shadow_progression") and row.get("shadow_checkpoint")]
    if accepted:
        checkpoint = Path(str(accepted[-1]["shadow_checkpoint"]))
        if not checkpoint.is_file():
            raise RuntimeError(f"resume_shadow_checkpoint_missing:{checkpoint}")
        track = load_root(runtime, checkpoint, key, canonical_model)
        # Restore the realized-motion history used by the adaptive basis.  A
        # resume must not silently erase this state: the current checkpoint is
        # sufficient for the model/optimizer, while the accepted shadow chain
        # supplies the parameter deltas without rerunning any rollout.
        previous_parameters = parameters(roots[key]["model"])
        history_deltas: list[dict[str, torch.Tensor]] = []
        for accepted_row in accepted:
            accepted_checkpoint = Path(str(accepted_row["shadow_checkpoint"]))
            accepted_model, _, _, _, _ = d43.load_state(runtime, accepted_checkpoint)
            current_parameters = parameters(accepted_model)
            history_deltas.append({name: current_parameters[name] - previous_parameters[name] for name in PARAMETER_NAMES})
            previous_parameters = current_parameters
        track["history_deltas"] = history_deltas
        track["parent_horizons"] = dict(accepted[-1]["d44_actual_horizons"])
        after = accepted[-1].get("d44_track_after", {})
        track["current_violation"] = float(after.get("current_violation", accepted[-1].get("d44_actual_feasibility_violation", 0.0)))
        track["trust_radius"] = float(after.get("trust_radius", 1.0))
        track["funnel_reference"] = dict(track["parent_horizons"])
        track["funnel_limit"] = max(1.0e-4, track["current_violation"] * 1.25)
        track["accepted_count"] = int(after.get("accepted_count", sum(bool(row.get("d44_accepted_for_shadow_progression")) for row in rows)))
        return track
    track = roots[key]
    track["funnel_reference"] = dict(track["parent_horizons"])
    track["current_violation"] = relative_violation(track["parent_horizons"], track["parent_horizons"], track["funnel_reference"])
    track["funnel_limit"] = max(1.0e-4, track["current_violation"] * 1.25)
    last = rows[-1].get("d44_track_after", {}) if rows else {}
    track["trust_radius"] = float(last.get("trust_radius", 1.0))
    return track


def method_ledger() -> list[dict[str, str]]:
    families = [
        "Full-Dynamics NMPC — OCS2", "Feasibility-Driven DDP — Crocoddyl FDDP", "Proximal Constrained DDP — Aligator / ProxDDP", "Multiple-Shooting SQP — OCS2 / acados", "Interior-Point OCP — OCS2 IPM / acados", "Dynamics-Aware GPU Trajectory Optimization — cuRobo", "Sequential Convex Trajectory Optimization — TrajOpt", "CHOMP", "GPMP2 / iGPMP2", "Graphs of Convex Sets — Drake GCS", "IRIS + GCS", "Sampling-Based Global Seed Generation — OMPL", "Hybrid Global / Local Planning — MoveIt Hybrid Planning", "RMPflow", "MPPI / Sampling MPC", "Operational-Space CBF — OSCBF", "High-Order CBF", "CLF-CBF-QP / CBFpy", "Backup CBF / DR-bCBF", "Predictive Safety Filter", "Hamilton–Jacobi Reachability", "Jerk-Based Safe Set / JSSA", "Robust Tube MPC", "Robust Convex Safe-Corridor MPC", "Task-Space Inverse Dynamics — TSID", "Null-Space / Hierarchical Control", "Set-Based Task Priority", "Impedance / Variable Impedance Control", "Passivity + Energy Tanks", "Jerk-Limited Trajectory Generation — Ruckig", "TOPP-RA", "MoveIt Servo", "Continuous Collision Validation", "STL Temporal Safety Constraints", "Long-Horizon + Short-Horizon Hierarchical Safety", "Adversarial Safety Testing",
    ]
    return [{"method": name, "status": "NOT_EXECUTED_D44_INTERFACE_NOT_EXPOSED", "code_path": "", "verdict": "not claimed as tested; candidate-specific robot/OCP adapter is absent"} for name in families]


def write_research_ledger(output: Path) -> None:
    text = """# D44 external research ledger

This ledger records the additional source work completed during the resumed D44 execution. “Fully read” means the available full paper text was inspected through its equations, algorithms, experiments, conclusion and appendices/references where present. It does not mean that an external method was executable in this repository.

## Papers fully read

| Source | Full-text route | D44-relevant finding | Executed in D44 |
|---|---|---|---|
| Mastalli et al., “A Feasibility-Driven Approach to Control-Limited DDP”, arXiv:2010.00411v4 | [arXiv HTML](https://arxiv.org/html/2010.00411v4) | Box-FDDP separates feasibility-driven and control-bounded modes, uses multiple-shooting gap contraction, control projection, nonlinear step lengths and expected-improvement checks. A real state/control/dynamics transcription is required. | Not claimed; the authenticated D44 transition exposes no shooting nodes, control limits or dynamics Jacobians. |
| Wabersich and Zeilinger, “A predictive safety filter for learning-based control of constrained nonlinear dynamical systems”, arXiv:1812.05506v4 | [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1812.05506) | The safety filter is an online MPC backup-plan problem with state/input tightening, a terminal safe set, uncertainty confidence maps and a shrinking-horizon fallback. Safety depends on an explicit system model, uncertainty set and stabilizing policy. | Not claimed; D44 has no candidate-specific robot state/input trajectory or uncertainty backend. |
| Yu et al., “Gradient Surgery for Multi-Task Learning”, NeurIPS 2020 | [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) | PCGrad projects conflicting task gradients, but the paper’s guarantees and experiments concern task-gradient conflict, curvature and magnitude imbalance. D44 uses only a labelled projected-gradient/basis analogue and validates every result with realized AdamW response. | The prior D43 projection analogue is reused as evidence; full PCGrad is not misreported as a D44 optimizer. |
| Eriksson and Poloczek, “Scalable Constrained Bayesian Optimization”, AISTATS 2021 | [PMLR page and PDF](https://proceedings.mlr.press/v130/eriksson21a.html) | SCBO uses local trust regions, transformed objectives/constraints, Thompson sampling and feasibility-first center updates. It requires a black-box objective/constraint observation model; raw surrogate predictions are not safety certificates. | D44’s competing realized-response models and feasibility-first ranking are a constrained analogue; no GP/SCBO result is claimed. |

## Repository fully inspected

| Repository | Inspection scope | Concrete implementation detail | D44 disposition |
|---|---|---|---|
| [WeiChengTseng/Pytorch-PCGrad](https://github.com/WeiChengTseng/Pytorch-PCGrad) | Shallow clone inspected across all tracked text/source files, including `pcgrad.py`, training entry point, network/data modules, utility code, README, requirements and deprecated implementation. | The implementation packs per-objective gradients, shuffles projection order, removes negative dot-product components and merges shared/non-shared gradients before delegating to the wrapped optimizer. | Confirms that D44’s real transition must keep replay and realized H1–H32 evaluation outside the projection helper; no external code was copied into the canonical path. |

## Coverage limits retained deliberately

The existing D43 index remains a targeted review of the other listed families, not a cover-to-cover claim. The 36 D44 robot-centered families remain `NOT_EXECUTED_D44_INTERFACE_NOT_EXPOSED` in the machine-readable method ledger. Full DDP, MPC/PSF, CBF, HQP, TrajOpt, MoveIt and other robot-centered results would require candidate-specific MoveIt2 PlanningScene, FK, dynamics, control/trajectory and post-Ruckig adapters that are not exposed by the current neural transition. No placeholder pass result was created.
"""
    (output / "D44_EXTERNAL_RESEARCH_LEDGER.md").write_text(text, encoding="utf-8", newline="\n")


def write_report(output: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    best_raw = min(rows, key=lambda row: float(row.get("candidate_H1", math.inf)), default={})
    best_safe = min((row for row in rows if row.get("d44_actual_h32_safe") and row.get("replay_pass")), key=lambda row: float(row.get("candidate_H1", math.inf)), default={})
    best_retained = min((row for row in rows if row.get("d44_actual_h32_safe") and row.get("replay_pass") and row.get("d44_full_retention_neutral")), key=lambda row: float(row.get("candidate_H1", math.inf)), default={})
    lines = [
        "# D44 realized-response robot-centered breakthrough campaign",
        "",
        f"STATUS: **{summary['TASK_STATUS']}**",
        "",
        f"The target H1 <= `{TARGET_H1:.12g}` was not reached and the canonical checkpoint was not mutated. D44 executed real shadow AdamW transitions followed by the full causal H1..H32 evaluator on the authenticated 181-point ON-state open-arch contract.",
        "",
        "## Verified result",
        "",
        f"- Starting/final canonical: H1 `{CANONICAL_H1:.15e}`, H32 `{CANONICAL_H32:.15e}`; promotions `{summary['CANONICAL_PROMOTIONS']}`; canonical regression `NO`.",
        f"- Best raw shadow: H1 `{summary['BEST_RAW_SHADOW_H1']}`, H32 `{best_raw.get('candidate_H32')}`; replay `{best_raw.get('replay_pass')}`; filter `{best_raw.get('d44_filter_disposition')}`.",
        f"- Best H32-safe replay shadow: H1 `{summary['BEST_H32_SAFE_H1']}`, H32 `{best_safe.get('candidate_H32')}`.",
        f"- Best retention-neutral replay shadow: H1 `{summary['BEST_RETENTION_SAFE_H1']}`, H32 `{best_retained.get('candidate_H32')}`.",
        "- Best robot-valid candidate: `null` — no candidate-specific MoveIt/FK/dynamics/post-Ruckig adapter was available, so robot metrics were recorded as JSON null and no safety was inferred.",
        "",
        "## Realized-response finding",
        "",
        "D44 compared local-linear, local-quadratic, and RBF response models against the realized transition. Trust regions contracted on model misses and expanded only after measured agreement. The model competition is evidence about this neural transition, not a robot safety oracle. The C track separately attempted feasibility restoration from the unsafe D43 basin; a curved safe route is reported only if the realized track accepted restoration and later met the hard H32 gate.",
        "",
        f"- Curved safe route to low-H1 basin: `{summary['CURVED_SAFE_ROUTE']}`.",
        f"- Strongest breakthrough: `{summary['STRONGEST_BREAKTHROUGH']}`.",
        f"- Strongest safe breakthrough: `{summary['STRONGEST_SAFE_BREAKTHROUGH']}`.",
        f"- Dominant failure mode: `{summary['BIGGEST_FAILURE_MODE']}`.",
        "",
        "## Research and method coverage",
        "",
        "This execution implemented the D44 realized-response trust-region/filter/restoration mechanism. The 36 robot-centered families were not falsely counted as tested: their candidate-specific OCP, control, or trajectory interfaces are not exposed by the current neural transition, and D44 records them as not executed. D43's prior research ledger remains the source record for the earlier methods review.",
        f"- Mandatory families completed: `{summary['MANDATORY_FAMILIES_COMPLETED']}/36`.",
        f"- Additional papers fully read during this execution: `{summary['ADDITIONAL_PAPERS_FULLY_READ']}`; exact scope is recorded in `D44_EXTERNAL_RESEARCH_LEDGER.md`.",
        f"- Repositories fully inspected during this execution: `{summary['REPOSITORIES_FULLY_READ']}`; exact scope is recorded in `D44_EXTERNAL_RESEARCH_LEDGER.md`.",
        "",
        "## Robot and canonical contract",
        "",
        "D39/D40/D41 remain inherited authenticated PASS for the unchanged canonical checkpoint. Candidate-specific robot validity is UNKNOWN, not PASS. Collision semantics remain `adaptive_discrete_interpolation`; Bullet CCD is `not_available`; clearance is JSON null unless a real backend reports it.",
        "",
        f"- Canonical checkpoint: `{CANONICAL_CHECKPOINT.resolve()}`.",
        f"- Evidence ledger: `{(output / 'D44_REALIZED_RESPONSE_LEDGER.jsonl').resolve()}`.",
        f"- Method ledger: `{(output / 'D44_METHOD_EXPERIMENT_LEDGER.json').resolve()}`.",
        f"- External research ledger: `{(output / 'D44_EXTERNAL_RESEARCH_LEDGER.md').resolve()}`.",
    ]
    (output / "D44_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-evals-per-track", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="append only unfinished attempts in an existing D44 output")
    parser.add_argument("--additional-evals", type=int, default=2, help="number of new attempts per track when --resume is used")
    parser.add_argument("--tracks", type=str, default="A,B,C")
    parser.add_argument("--seed", type=int, default=44044)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime, _, canonical_model_obj, _, canonical_rng, canonical_lrs, runtime_meta = d42.load_authenticated_runtime()
    canonical_model = parameters(canonical_model_obj)
    roots = {
        "A": load_root(runtime, CANONICAL_CHECKPOINT, "A", canonical_model),
        "B": load_root(runtime, D42_SAFE, "B", canonical_model),
        "C": load_root(runtime, D43_UNSAFE, "C", canonical_model),
    }
    references = {key: parameters(value["model"]) for key, value in roots.items()}
    for value in roots.values():
        value["current_violation"] = relative_violation(value["parent_horizons"], value["parent_horizons"], value["parent_horizons"])
        value["trust_radius"] = 1.0
        value["accepted_count"] = 0
    serial_roots = {key: serial_track(value) for key, value in roots.items()}
    ledger_path = output / "D44_REALIZED_RESPONSE_LEDGER.jsonl"
    prior_rows = read_ledger(ledger_path) if args.resume else []
    if args.resume and not prior_rows:
        raise RuntimeError(f"resume_ledger_missing_or_empty:{ledger_path}")
    write_json(output / "D44_AUTHORITY.json", {
        "schema_version": "d44_authority_v1",
        "scope": "181-point ON-state open-arch only",
        "canonical_checkpoint": str(CANONICAL_CHECKPOINT.resolve()),
        "canonical_update": CANONICAL_UPDATE,
        "canonical_H1": CANONICAL_H1,
        "canonical_H32": CANONICAL_H32,
        "H1_target": TARGET_H1,
        "H32_limit": H32_LIMIT,
        "promotion_H32_reserve": PROMOTION_H32_RESERVE,
        "collision_method": "adaptive_discrete_interpolation",
        "ccd": "not_available",
        "clearance": None,
        "D39_retention": runtime_meta["d41"].get("D39_RETENTION"),
        "D40_retention": runtime_meta["d41"].get("D40_RETENTION"),
        "D41_retention": runtime_meta["d41"].get("COLLISION_CERTIFICATION_STATUS"),
        "roots": serial_roots,
        "canonical_mutation": False,
    })
    all_rows: list[dict[str, Any]] = []
    selected = [value.strip() for value in args.tracks.split(",") if value.strip() in roots]
    for key in selected:
        track = resume_track(runtime, key, roots, prior_rows, canonical_model) if args.resume else roots[key]
        prior_track_rows = [row for row in prior_rows if row.get("d44_track") == key]
        new_rows = run_track(
            runtime, track, references, output,
            max(0, int(args.additional_evals if args.resume else args.max_evals_per_track)),
            int(args.seed) + ord(key), prior_track_rows,
            max((int(row.get("d44_attempt", -1)) for row in prior_track_rows), default=-1) + 1,
        )
        roots[key] = track
        all_rows.extend(new_rows if args.resume else new_rows)
    if args.resume:
        # run_track returns the track history for fitting.  The persisted ledger
        # already contains the old rows; keep the summary/table complete while
        # avoiding duplicate JSONL writes.
        all_rows = prior_rows + [row for row in all_rows if row not in prior_rows]
    # Reconstruct every track's latest state from the persisted ledger.  This
    # matters for a selective resume (for example, --tracks C): unselected
    # tracks must retain their accepted shadow checkpoint and trust-region
    # state instead of being silently summarized from their original root.
    serial_roots = {}
    for key, value in roots.items():
        track_rows = [row for row in all_rows if row.get("d44_track") == key]
        latest_after = track_rows[-1].get("d44_track_after") if track_rows else None
        serial_roots[key] = latest_after if isinstance(latest_after, Mapping) else serial_track(value)
    write_json(output / "D44_METHOD_EXPERIMENT_LEDGER.json", method_ledger())
    write_research_ledger(output)
    write_csv(output / "D44_REALIZED_RESPONSE_LEDGER.csv", all_rows)
    best_raw = min(all_rows, key=lambda row: float(row.get("candidate_H1", math.inf)), default={})
    safe_rows = [row for row in all_rows if row.get("d44_actual_h32_safe") and row.get("replay_pass")]
    retained_rows = [row for row in safe_rows if row.get("d44_full_retention_neutral")]
    route_rows = [row for row in all_rows if row.get("d44_filter_disposition") == "FEASIBILITY_RESTORATION" and row.get("d44_accepted_for_shadow_progression")]
    summary = {
        "schema_version": "d44_summary_v1",
        "TASK_STATUS": "PARTIAL_PROGRESS_TARGET_NOT_YET_ACHIEVED",
        "FINAL_TARGET_STATUS": "NOT_ACHIEVED",
        "H1_TARGET": TARGET_H1,
        "STARTING_CANONICAL_H1": CANONICAL_H1,
        "FINAL_CANONICAL_H1": CANONICAL_H1,
        "STARTING_CANONICAL_H32": CANONICAL_H32,
        "FINAL_CANONICAL_H32": CANONICAL_H32,
        "H32_LIMIT": H32_LIMIT,
        "FINAL_H32_MARGIN": H32_LIMIT - CANONICAL_H32,
        "BEST_RAW_SHADOW_H1": float(best_raw["candidate_H1"]) if best_raw else None,
        "BEST_H32_SAFE_H1": float(min(safe_rows, key=lambda row: float(row["candidate_H1"]))["candidate_H1"]) if safe_rows else None,
        "BEST_RETENTION_SAFE_H1": float(min(retained_rows, key=lambda row: float(row["candidate_H1"]))["candidate_H1"]) if retained_rows else None,
        "BEST_ROBOT_VALID_H1": None,
        "BEST_REPLAY_VALIDATED_H1": float(min((row for row in all_rows if row.get("replay_pass")), key=lambda row: float(row["candidate_H1"]))["candidate_H1"]) if any(row.get("replay_pass") for row in all_rows) else None,
        "TOTAL_SHADOW_EXPERIMENTS": len(all_rows),
        "MAJOR_METHOD_FAMILIES_ATTEMPTED": ["D44_REALIZED_RESPONSE_TR", "local_linear_response_model", "local_quadratic_response_model", "rbf_response_model", "trust_region_filter_restoration"],
        "HYBRIDS_ATTEMPTED": ["realized_response_TR_plus_filter", "realized_response_TR_plus_feasibility_restoration", "multi_output_model_competition"],
        "CANONICAL_PROMOTIONS": 0,
        "MANDATORY_FAMILIES_COMPLETED": 0,
        "ADDITIONAL_PAPERS_FULLY_READ": D44_FULLY_READ_PAPERS,
        "REPOSITORIES_FULLY_READ": D44_FULLY_INSPECTED_REPOSITORIES,
        "BEST_METHOD": "D44_REALIZED_RESPONSE_TR",
        "BEST_HYBRID": "realized_response_TR_plus_filter_restoration",
        "STRONGEST_BREAKTHROUGH": {"H1": best_raw.get("candidate_H1"), "H32": best_raw.get("candidate_H32"), "track": best_raw.get("d44_track"), "status": best_raw.get("d44_filter_disposition")},
        "STRONGEST_SAFE_BREAKTHROUGH": {"H1": min(safe_rows, key=lambda row: float(row["candidate_H1"])).get("candidate_H1") if safe_rows else None, "H32": min(safe_rows, key=lambda row: float(row["candidate_H1"])).get("candidate_H32") if safe_rows else None},
        "BIGGEST_FAILURE_MODE": "realized_response_model_error_or_H32_retention_tradeoff",
        "CURVED_SAFE_ROUTE": "EVIDENCE_INSUFFICIENT" if not route_rows else "PARTIAL_RESTORATION_ONLY",
        "D39_RETENTION": runtime_meta["d41"].get("D39_RETENTION"),
        "D40_RETENTION": runtime_meta["d41"].get("D40_RETENTION"),
        "D41_RETENTION": runtime_meta["d41"].get("COLLISION_CERTIFICATION_STATUS"),
        "CANONICAL_REGRESSION": "NO",
        "canonical_checkpoint": str(CANONICAL_CHECKPOINT.resolve()),
        "roots": serial_roots,
    }
    write_json(output / "D44_SUMMARY.json", summary)
    write_report(output, summary, all_rows)
    print(json.dumps({"status": summary["TASK_STATUS"], "experiments": len(all_rows), "best_raw": summary["BEST_RAW_SHADOW_H1"], "best_safe": summary["BEST_H32_SAFE_H1"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
