"""Create an explicit shadow source-q variant with selected local IK endpoints.

This never edits a canonical trajectory.  It is used only to test whether a
different, recorded IK family can be spliced to the unchanged trajectory on
either side of the Stage 5B branch cut.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--waypoint", type=int, required=True)
    parser.add_argument("--q", type=float, nargs=6, required=True)
    args = parser.parse_args()
    with args.source_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= args.waypoint < len(rows):
        raise SystemExit("waypoint_out_of_range")
    for joint, value in enumerate(args.q, start=1):
        field = f"j{joint}_q"
        if field not in rows[args.waypoint]:
            raise SystemExit(f"missing_field:{field}")
        rows[args.waypoint][field] = repr(float(value))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
