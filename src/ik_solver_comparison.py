"""Stage 1.9.2 result flattening and evidence-bound classification."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return [dict(row) for row in data.get("records", [])]
    rows: list[dict[str, Any]] = []
    for child in sorted(path.glob("*.json")):
        rows.extend(load_records(child))
    return rows


def solution_hash_set(records: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("solution_quantized_sha256", row.get("solution_raw_sha256", "NO_SOLUTION"))) for row in records})


def summarize(records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    success = [bool(row.get("solver_success", row.get("direct_kdl_success", False))) for row in records]
    input_keys = [tuple(row.get(key) for key in ("target_transform_raw_sha256", "seed_vector_raw_sha256", "all_robot_variables_raw_sha256", "solver_options_raw_sha256")) for row in records if "target_transform_raw_sha256" in row]
    return {
        "record_count": len(records), "expected_count": expected_count, "count_match": len(records) == expected_count,
        "success_count": sum(success), "success_values": sorted(set(success)), "all_success_status_identical": len(set(success)) <= 1,
        "solution_hashes": solution_hash_set(records), "raw_solution_hashes": sorted({str(row.get("solution_raw_sha256", "NO_SOLUTION")) for row in records}),
        "input_hashes_identical": len(set(input_keys)) <= 1, "input_hash_values": sorted({str(item) for item in set(input_keys)}),
        "fk_position_errors_m": [row.get("fk_position_error_m") for row in records if row.get("fk_position_error_m") is not None and float(row.get("fk_position_error_m")) >= 0],
        "teardown_status": "tracked_separately",
    }


def classification(moveitpy: dict[str, Any], cpp: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    if not all(item.get("count_match") for item in (moveitpy, cpp, direct)):
        return {"first_unstable_layer": None, "root_cause_classification": "undetermined", "reason": "one or more replay groups incomplete"}
    # A fully deterministic failure is not an IK nondeterminism proof.
    if all(item.get("all_success_status_identical") and len(item.get("solution_hashes", [])) <= 1 for item in (moveitpy, cpp, direct)):
        return {"first_unstable_layer": None, "root_cause_classification": "undetermined", "reason": "all completed layers were deterministic; no nondeterministic layer reproduced"}
    if not moveitpy.get("all_success_status_identical") and cpp.get("all_success_status_identical") and direct.get("all_success_status_identical"):
        return {"first_unstable_layer": "moveitpy", "root_cause_classification": "moveitpy_binding_or_lifecycle", "reason": "only MoveItPy status or output varied under identical binary inputs"}
    if not cpp.get("all_success_status_identical") and direct.get("all_success_status_identical"):
        return {"first_unstable_layer": "cpp_moveit_plugin", "root_cause_classification": "moveit_kdl_adapter_layer", "reason": "C++ plugin varied while direct KDL did not"}
    if not direct.get("all_success_status_identical"):
        return {"first_unstable_layer": "direct_orocos_kdl", "root_cause_classification": "orocos_kdl_numeric_runtime", "reason": "direct KDL status varied"}
    return {"first_unstable_layer": None, "root_cause_classification": "undetermined", "reason": "evidence does not isolate a layer"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
