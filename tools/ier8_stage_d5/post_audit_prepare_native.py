#!/usr/bin/env python3
"""Prepare seed files for post-D5 native discriminating experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SEED_FIELDS = ["point_id", "seed_id", "seed_source", *(f"q{i}_rad" for i in range(1, 7))]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_seeds(rows: list[dict[str, str]], output: Path, source: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SEED_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            point = int(row["point_id"])
            writer.writerow({
                "point_id": point,
                "seed_id": f"POST_AUDIT_{source}_WP{point:03d}_C{index:03d}",
                "seed_source": source,
                **{f"q{i}_rad": row[f"q{i}_rad"] for i in range(1, 7)},
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="post_audit_independent_bounded_branch_pool")
    args = parser.parse_args()
    rows = read_rows(args.candidates)
    write_seeds(rows, args.output, args.source)
    print(f"wrote {len(rows)} seeds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
