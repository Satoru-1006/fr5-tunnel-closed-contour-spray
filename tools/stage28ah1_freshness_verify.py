#!/usr/bin/env python3
"""Recompute Stage 2.8A-H1 freshness from an immutable raw telemetry CSV.

This command has no SDK, network, ROS, controller-manager, or hardware
dependency.  It is intentionally a separate process from the C++ probe so
the formal result cannot be copied from the probe's own summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28ah0_freshness import verify_raw_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--sample-count-min", type=int, default=100)
    args = parser.parse_args()

    result = verify_raw_csv(args.raw_csv, sample_count_min=args.sample_count_min)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["fresh_samples_proven"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
