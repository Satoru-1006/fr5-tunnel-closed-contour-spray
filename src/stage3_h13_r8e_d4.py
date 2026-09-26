"""Pure D4 canonical-batch-schedule helpers.

The execution runner keeps the R8E-A1 training/evaluation implementation
unchanged and uses this module only for the single precommitted schedule
intervention plus fail-closed aggregation predicates.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "STAGE_3_H13_R8E_D4_A1_ORDER_ROBUSTNESS_V1"
SCHEDULE_RULE = (
    "For each zero-based schedule cycle c, sort frozen TRAIN window IDs by "
    "SHA-256(EXPERIMENT_ID|cycle|c|window_id), with integer window_id as the "
    "deterministic tie-break; consume contiguous batches of 256. At a cycle "
    "boundary concatenate the remaining tail with the next canonical-cycle "
    "prefix. Generate exactly the 512 ordered batches before the first run."
)
BATCH_SIZE = 256
UPDATES_PER_PHASE = 128
TOTAL_UPDATES = 512
H1_THRESHOLD = 0.0000859791431235184
H32_THRESHOLD = 0.01856902565856056
CHAMPION_MEDIAN_H32_LIMIT = 0.015847526627505683
SPREAD_ABSOLUTE_TOLERANCE = 5.0e-9
SPREAD_RELATIVE_TOLERANCE = 5.0e-6
FROZEN_SEEDS = (3830844401, 1842510982, 3932772377, 2312436631, 4214033242)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_frozen_seeds(seeds: Sequence[int]) -> dict[str, Any]:
    observed = [int(seed) for seed in seeds]
    expected = list(FROZEN_SEEDS)
    if len(observed) != len(expected):
        raise ValueError("seed_count_mismatch")
    if len(set(observed)) != len(observed):
        raise ValueError("seed_duplicates")
    if set(observed) != set(expected):
        raise ValueError("seed_set_mismatch")
    if any(seed < 0 or seed > 0xFFFFFFFF for seed in observed):
        raise ValueError("seed_out_of_uint32_range")
    return {
        "seeds": observed,
        "seed_count": len(observed),
        "unique_seed_count": len(set(observed)),
        "all_seeds_frozen_before_first_training": "YES",
        "failed_seed_replacement_allowed": "NO",
        "extra_rescue_seed_allowed": "NO",
        "lucky_seed_selection_allowed": "NO",
    }


def _cycle_order(window_ids: Sequence[int], cycle: int, experiment_id: str) -> list[int]:
    keyed = []
    for raw_id in window_ids:
        window_id = int(raw_id)
        key = f"{experiment_id}|cycle|{int(cycle)}|{window_id}"
        keyed.append((sha256_text(key), window_id))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[1] for item in keyed]


def build_canonical_schedule(
    window_ids: Iterable[int],
    *,
    batch_size: int = BATCH_SIZE,
    update_count: int = TOTAL_UPDATES,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    """Build the complete deterministic ordered-batch manifest.

    This function has no seed parameter and uses only frozen window IDs plus
    the precommitted experiment identifier and cycle index.
    """

    ids = [int(item) for item in window_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("train_window_ids_empty_or_duplicate")
    if int(batch_size) < 1 or int(update_count) < 1:
        raise ValueError("schedule_dimensions_invalid")

    cycle_orders: dict[int, list[int]] = {}

    def get_order(cycle: int) -> list[int]:
        if cycle not in cycle_orders:
            cycle_orders[cycle] = _cycle_order(ids, cycle, experiment_id)
        return cycle_orders[cycle]

    cycle = 0
    order = get_order(cycle)
    cursor = 0
    batches: list[dict[str, Any]] = []
    while len(batches) < int(update_count):
        remaining = order[cursor:]
        if len(remaining) >= int(batch_size):
            batch_ids = remaining[: int(batch_size)]
            cursor += int(batch_size)
            spans = [{"cycle": cycle, "count": int(batch_size)}]
        else:
            tail = list(remaining)
            next_cycle = cycle + 1
            next_order = get_order(next_cycle)
            needed = int(batch_size) - len(tail)
            batch_ids = tail + next_order[:needed]
            spans = []
            if tail:
                spans.append({"cycle": cycle, "count": len(tail)})
            spans.append({"cycle": next_cycle, "count": needed})
            cycle = next_cycle
            order = next_order
            cursor = needed
        if len(batch_ids) != int(batch_size):
            raise AssertionError("canonical_batch_size_internal_failure")
        batches.append(
            {
                "update_index": len(batches),
                "window_ids": [int(item) for item in batch_ids],
                "source_cycle_spans": spans,
            }
        )

    cycle_records = [
        {
            "cycle": int(index),
            "window_count": len(cycle_orders[index]),
            "ordered_window_ids_sha256": canonical_json_sha256(cycle_orders[index]),
            "first_window_ids": cycle_orders[index][:8],
            "last_window_ids": cycle_orders[index][-8:],
        }
        for index in sorted(cycle_orders)
    ]
    return {
        "schema_version": "stage3_h13_r8e_d4_canonical_schedule_manifest_v1",
        "experiment_identifier": experiment_id,
        "generation_rule": SCHEDULE_RULE,
        "schedule_cycle_index_base": 0,
        "batch_size": int(batch_size),
        "number_of_training_updates": int(update_count),
        "number_of_batches": len(batches),
        "number_of_windows": len(ids),
        "train_window_ids_sha256": canonical_json_sha256(ids),
        "cycle_count_used": len(cycle_records),
        "cycle_order_records": cycle_records,
        "ordered_batches": batches,
        "ordered_batch_manifest_sha256": canonical_json_sha256(batches),
        "canonical_schedule_seed_independent": "YES",
        "prospective_seed_consumed": "NO",
    }


def schedule_indices(manifest: Mapping[str, Any], available_window_ids: Sequence[int]) -> np.ndarray:
    """Resolve the manifest's window IDs to array positions without reshuffling."""

    available = [int(item) for item in available_window_ids]
    if len(available) != len(set(available)):
        raise ValueError("available_window_ids_duplicate")
    position = {window_id: index for index, window_id in enumerate(available)}
    rows = []
    for expected_update, batch in enumerate(manifest.get("ordered_batches", ())):
        if int(batch.get("update_index", -1)) != expected_update:
            raise ValueError("schedule_update_index_mismatch")
        ids = [int(item) for item in batch.get("window_ids", ())]
        if len(ids) != int(manifest["batch_size"]):
            raise ValueError("schedule_batch_size_mismatch")
        try:
            rows.append([position[item] for item in ids])
        except KeyError as exc:
            raise ValueError("schedule_window_id_not_available") from exc
    result = np.asarray(rows, dtype=np.int64)
    if result.shape != (int(manifest["number_of_batches"]), int(manifest["batch_size"])):
        raise ValueError("schedule_shape_mismatch")
    return result


def metric_spread(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        return {
            "minimum": None,
            "maximum": None,
            "spread": None,
            "allowed_bound": None,
            "pass": False,
            "finite": False,
            "count": len(numbers),
        }
    minimum = min(numbers)
    maximum = max(numbers)
    spread = maximum - minimum
    bound = SPREAD_ABSOLUTE_TOLERANCE + SPREAD_RELATIVE_TOLERANCE * max(abs(value) for value in numbers)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "spread": spread,
        "allowed_bound": bound,
        "pass": bool(spread <= bound),
        "finite": True,
        "count": len(numbers),
    }


def aggregate_metric(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        return {"count": len(numbers), "finite": False, "minimum": None, "median": None, "mean": None, "population_std": None, "maximum": None, "range": None}
    array = np.asarray(numbers, dtype=np.float64)
    return {
        "count": len(numbers),
        "finite": True,
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array, ddof=0)),
        "maximum": float(np.max(array)),
        "range": float(np.max(array) - np.min(array)),
    }


def h1_pass(value: float) -> bool:
    return bool(math.isfinite(float(value)) and float(value) <= H1_THRESHOLD)


def h32_pass(value: float) -> bool:
    return bool(math.isfinite(float(value)) and float(value) <= H32_THRESHOLD)


def champion_replacement_pass(
    *,
    completed_count: int,
    valid_count: int,
    h1_pass_count: int,
    h32_pass_count: int,
    h16_all_finite: bool,
    h1_spread_pass: bool,
    h16_spread_pass: bool,
    h32_spread_pass: bool,
    median_h32: float | None,
    integrity_pass: bool,
    leakage_pass: bool,
    schedule_pass: bool,
    checkpoint_pass: bool,
    no_seed_replacement: bool,
    no_post_hoc_change: bool,
    frozen20_excluded: bool,
) -> bool:
    return bool(
        completed_count == 5
        and valid_count == 5
        and h1_pass_count == 5
        and h32_pass_count == 5
        and h16_all_finite
        and h1_spread_pass
        and h16_spread_pass
        and h32_spread_pass
        and median_h32 is not None
        and math.isfinite(float(median_h32))
        and float(median_h32) <= CHAMPION_MEDIAN_H32_LIMIT
        and integrity_pass
        and leakage_pass
        and schedule_pass
        and checkpoint_pass
        and no_seed_replacement
        and no_post_hoc_change
        and frozen20_excluded
    )


def assert_frozen20_excluded(paths: Iterable[str]) -> None:
    forbidden = ("frozen20", "frozen_20", "frozen-20", "stage3_h13_r3_frozen")
    for raw_path in paths:
        lowered = str(raw_path).replace("\\", "/").lower()
        if any(token in lowered for token in forbidden):
            raise ValueError("frozen20_path_access_forbidden")
