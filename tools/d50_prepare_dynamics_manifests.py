"""Create Windows-path manifests for the isolated D50 Pinocchio audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def entries(root: Path) -> list[tuple[str, Path, str]]:
    result = []
    for path in sorted((root / "trajectories").glob("*.csv")):
        prefix = path.stem.rsplit("_", 1)[0].upper()
        family = "COLLISION_SENSITIVE" if prefix == "COLLISION" else prefix
        result.append((path.stem, path.resolve(), family))
    return result


def write_manifest(path: Path, rows: list[tuple[str, Path, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = entries(args.baseline_root)
    candidate_rows = entries(args.candidate_root)
    if [item[0] for item in baseline_rows] != [item[0] for item in candidate_rows]:
        raise SystemExit("baseline_candidate_case_order_mismatch")
    write_manifest(args.output_dir / "baseline_manifest.csv", baseline_rows)
    write_manifest(args.output_dir / "candidate_manifest.csv", candidate_rows)
    summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    with (args.output_dir / "baseline_durations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_duration_s"])
        for item in summary["cases"]:
            writer.writerow([item["case_id"], item["duration_s"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
