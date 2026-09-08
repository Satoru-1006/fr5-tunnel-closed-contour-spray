#!/usr/bin/env python3
"""Minimum-bottleneck DP over a revalidated native branch pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def wrapped_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(b - a), np.cos(b - a))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in read_rows(args.attempts) if row.get("reason") == "VALID_NATIVE_IK" and row.get("collision_free") == "true"]
    layers: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        point = int(row["point_id"])
        q = np.asarray([float(row[f"q{i}_rad"]) for i in range(1, 7)], dtype=float)
        layers.setdefault(point, []).append({"row": row, "q": q})
    points = sorted(layers)
    if not points or points != list(range(points[0], points[-1] + 1)):
        raise RuntimeError(f"native branch layers are not contiguous: {points}")
    costs: list[np.ndarray] = [np.zeros(len(layers[points[0]]))]
    predecessors: list[np.ndarray] = [np.full(len(layers[points[0]]), -1, dtype=int)]
    for layer_index in range(1, len(points)):
        previous = layers[points[layer_index - 1]]
        current = layers[points[layer_index]]
        prev_cost = costs[-1]
        current_cost = np.full(len(current), math.inf)
        current_prev = np.full(len(current), -1, dtype=int)
        for j, candidate in enumerate(current):
            edge = np.asarray([float(np.max(np.abs(wrapped_delta(item["q"], candidate["q"])))) for item in previous])
            total = np.maximum(prev_cost, edge)
            index = int(np.argmin(total))
            current_cost[j] = total[index]
            current_prev[j] = index
        costs.append(current_cost)
        predecessors.append(current_prev)
    selected_indices = [int(np.argmin(costs[-1]))]
    for layer_index in range(len(points) - 1, 0, -1):
        selected_indices.append(int(predecessors[layer_index][selected_indices[-1]]))
    selected_indices.reverse()
    selected_rows: list[dict[str, object]] = []
    jumps: list[float] = []
    for layer_index, point in enumerate(points):
        item = layers[point][selected_indices[layer_index]]
        row = item["row"]
        selected_rows.append({"waypoint_id": point, "seed_id": row.get("seed_id", ""), **{f"q{i}_rad": row[f"q{i}_rad"] for i in range(1, 7)}})
        if layer_index:
            previous_q = layers[points[layer_index - 1]][selected_indices[layer_index - 1]]["q"]
            jumps.append(float(np.max(np.abs(wrapped_delta(previous_q, item["q"])))))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "BRANCH_DP_SELECTED.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(selected_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(selected_rows)
    maximum = max(jumps, default=0.0)
    max_index = int(np.argmax(jumps)) if jumps else -1
    summary = {
        "schema_version": "post_stage_d5_native_branch_dp_v1",
        "point_range": [points[0], points[-1]],
        "layer_count": len(points),
        "native_valid_solution_count": len(rows),
        "native_valid_counts_by_waypoint": {str(point): len(layers[point]) for point in points},
        "selected_path_established": True,
        "max_wrapped_jump_rad": maximum,
        "max_wrapped_jump_deg": math.degrees(maximum),
        "max_jump_transition": [points[max_index], points[max_index + 1]] if max_index >= 0 else None,
        "continuity_threshold_deg": 20.0,
        "continuity_violations_over_20deg": sum(value > math.radians(20.0) for value in jumps),
    }
    (args.output / "BRANCH_DP_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
