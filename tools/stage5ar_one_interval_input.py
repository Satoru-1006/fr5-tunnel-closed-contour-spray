"""Project the first frozen D65 interval into a one-row probe input."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = args.source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise RuntimeError("source interval CSV has no data row")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    print(f"wrote_one_interval={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
