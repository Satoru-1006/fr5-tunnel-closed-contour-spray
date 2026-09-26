"""Pure aggregation and integrity helpers for the Stage 3 H13 R8E-B audit."""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


SEEDS = (13086, 218309645, 258275761, 1638377564, 1727279519)
R6_H32 = 0.0232112820732007
H32_THRESHOLD = 0.01856902565856056
H1_THRESHOLD = 0.00008597914317990215


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_seed_manifest(payload: Mapping[str, Any]) -> None:
    seeds = tuple(int(seed) for seed in payload.get("seeds", ()))
    if seeds != SEEDS or len(set(seeds)) != len(SEEDS):
        raise RuntimeError("seed_manifest_not_exact")
    if payload.get("mutation_policy") != "FORBIDDEN_AFTER_CREATION":
        raise RuntimeError("seed_manifest_immutability_not_declared")


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    seeds = tuple(int(record["seed"]) for record in records)
    if seeds != SEEDS or len(set(seeds)) != len(SEEDS):
        raise RuntimeError("all_five_seeds_not_represented_exactly_once")


def distribution(values: Sequence[float]) -> dict[str, float]:
    finite = [float(value) for value in values]
    if not finite or not all(math.isfinite(value) for value in finite):
        raise RuntimeError("aggregate_requires_finite_values")
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }


def improvement_percent(value: float) -> float:
    return 100.0 * (R6_H32 - float(value)) / R6_H32


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_records(records)
    h1_values = [float(record["metrics"]["1"]) for record in records]
    h32_values = [float(record["metrics"]["32"]) for record in records]
    h1 = distribution(h1_values)
    h32 = distribution(h32_values)
    h1_pass_count = sum(value <= H1_THRESHOLD for value in h1_values)
    h32_pass_count = sum(value <= H32_THRESHOLD for value in h32_values)
    beating_r6 = sum(value < R6_H32 for value in h32_values)
    result = {
        "h1": {**h1, "pass_count": h1_pass_count, "fail_count": len(records) - h1_pass_count},
        "h32": {
            **h32,
            "mean_improvement_percent": improvement_percent(h32["mean"]),
            "median_improvement_percent": improvement_percent(h32["median"]),
            "seeds_beating_r6": beating_r6,
            "seeds_reaching_20_percent": h32_pass_count,
        },
    }
    result["gates"] = {
        "h1_4_of_5_pass": h1_pass_count >= 4,
        "h32_4_of_5_pass": h32_pass_count >= 4,
        "median_20_percent_pass": h32["median"] <= H32_THRESHOLD,
    }
    return result


def aggregate_hidden(records: Sequence[Mapping[str, Any]], r6: Mapping[str, Any]) -> dict[str, Any]:
    validate_records(records)
    result: dict[str, Any] = {"r6": dict(r6), "horizons": {}}
    for horizon in (2, 4, 8, 16, 32):
        key = str(horizon)
        values = [float(record["hidden_divergence"][key]["relative_l2_median"]) for record in records]
        layers = {}
        for layer in (1, 2):
            layer_key = f"layer_{layer}_relative_l2_median"
            layers[f"layer_{layer}"] = distribution([float(record["hidden_divergence"][key][layer_key]) for record in records])
        r6_value = float(r6[key]["relative_l2_median"])
        result["horizons"][key] = {
            **distribution(values),
            "r6_relative_l2_median": r6_value,
            "seeds_lower_than_r6": sum(value < r6_value for value in values),
            "per_layer": layers,
        }
    return result


def compute_gates(
    aggregate: Mapping[str, Any],
    *,
    integrity_pass: bool,
    tests_pass: bool,
    replay_pass: bool,
) -> dict[str, bool]:
    gates = {
        "integrity_pass": bool(integrity_pass),
        "h1_4_of_5_pass": bool(aggregate["gates"]["h1_4_of_5_pass"]),
        "h32_4_of_5_pass": bool(aggregate["gates"]["h32_4_of_5_pass"]),
        "median_20_percent_pass": bool(aggregate["gates"]["median_20_percent_pass"]),
        "tests_pass": bool(tests_pass),
        "replay_pass": bool(replay_pass),
    }
    gates["r8e_b_final_pass"] = all(gates.values())
    return gates


def assert_allowed_paths(paths: Sequence[Path]) -> None:
    """Fail closed if a configured path could address the sealed evaluation set."""
    forbidden_fragments = ("frozen" + "20", "frozen_" + "20")
    for path in paths:
        lowered = str(path).lower()
        if any(fragment in lowered for fragment in forbidden_fragments):
            raise RuntimeError("sealed_evaluation_path_excluded")
