from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


JOINT_COLUMNS = ["j1_q", "j2_q", "j3_q", "j4_q", "j5_q", "j6_q"]


def circular_delta(delta: np.ndarray) -> np.ndarray:
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def read_joint_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not set(JOINT_COLUMNS).issubset(fieldnames):
            alternate = [f"q{i}" for i in range(1, 7)]
            if set(alternate).issubset(fieldnames):
                columns = alternate
            else:
                raise ValueError(f"{path} is missing joint columns {JOINT_COLUMNS}.")
        else:
            columns = JOINT_COLUMNS
        rows = [[float(row[column]) for column in columns] for row in reader]
    if not rows:
        raise ValueError(f"{path} contains no joint rows.")
    return np.asarray(rows, dtype=float)


def max_adjacent_step_deg(q: np.ndarray) -> dict[str, float | int]:
    if len(q) < 2:
        return {"max_step_deg": 0.0, "from_index": -1, "to_index": -1, "joint_index": -1}
    steps = np.rad2deg(np.abs(circular_delta(np.diff(q, axis=0))))
    flat_index = int(np.argmax(steps))
    from_index, joint_index = np.unravel_index(flat_index, steps.shape)
    return {
        "max_step_deg": float(steps[from_index, joint_index]),
        "from_index": int(from_index),
        "to_index": int(from_index + 1),
        "joint_index": int(joint_index),
    }


def analyze_splice(
    full_csv: Path,
    local_csv: Path,
    local_source_start_index: int,
    limit_deg: float = 20.0,
) -> dict[str, object]:
    full = read_joint_csv(full_csv)
    local = read_joint_csv(local_csv)
    local_start = int(local_source_start_index)
    local_end = local_start + len(local) - 1
    if local_start < 0 or local_end >= len(full):
        raise ValueError(
            f"Local source range {local_start}-{local_end} is outside full trajectory length {len(full)}."
        )

    direct_differences = []
    for local_index, full_index in enumerate(range(local_start, local_end + 1)):
        diff_deg = np.rad2deg(np.abs(circular_delta(local[local_index] - full[full_index])))
        joint_index = int(np.argmax(diff_deg))
        direct_differences.append(
            {
                "full_index": int(full_index),
                "local_index": int(local_index),
                "max_difference_deg": float(diff_deg[joint_index]),
                "joint": f"j{joint_index + 1}",
            }
        )

    splice_trials = []
    for splice_start in range(local_start, local_end + 1):
        local_offset = splice_start - local_start
        patched = full.copy()
        patched[splice_start : local_end + 1] = local[local_offset:]
        step = max_adjacent_step_deg(patched)
        splice_trials.append(
            {
                "splice_start_index": int(splice_start),
                "splice_end_index": int(local_end),
                "max_step_deg": step["max_step_deg"],
                "max_step_from_index": step["from_index"],
                "max_step_to_index": step["to_index"],
                "max_step_joint": f"j{int(step['joint_index']) + 1}" if int(step["joint_index"]) >= 0 else "",
                "status": "pass" if float(step["max_step_deg"]) <= float(limit_deg) else "fail",
            }
        )
    best_trial = min(splice_trials, key=lambda row: float(row["max_step_deg"]))
    return {
        "full_csv": str(full_csv),
        "local_csv": str(local_csv),
        "local_source_start_index": int(local_start),
        "local_source_end_index": int(local_end),
        "limit_deg": float(limit_deg),
        "direct_difference_max_deg": max(float(row["max_difference_deg"]) for row in direct_differences),
        "direct_differences": direct_differences,
        "best_splice_trial": best_trial,
        "splice_trials": splice_trials,
        "interpretation": (
            "pass" if best_trial["status"] == "pass"
            else "No tested direct splice of the local IK branch satisfies the joint-step limit; "
            "the local branch is a diagnostic island unless a longer branch transition is found."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether a local IK branch can be spliced into a full trajectory.")
    parser.add_argument("--full-csv", type=Path, required=True)
    parser.add_argument("--local-csv", type=Path, required=True)
    parser.add_argument("--local-source-start-index", type=int, required=True)
    parser.add_argument("--limit-deg", type=float, default=20.0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    payload = analyze_splice(args.full_csv, args.local_csv, args.local_source_start_index, args.limit_deg)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
