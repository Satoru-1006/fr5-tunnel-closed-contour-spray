#!/usr/bin/env python3
"""Build the immutable Stage 1.9.1 IK call corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frozen_replay_diagnostics import build_frozen_corpus, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage191_determinism_diagnostics.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ik_graph_stage191/frozen_ik_call_corpus.parquet")
    args = parser.parse_args()
    summary = build_frozen_corpus(args.config.resolve(), args.output.resolve())
    write_json(args.output.with_name("frozen_ik_call_corpus_summary.json"), summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
