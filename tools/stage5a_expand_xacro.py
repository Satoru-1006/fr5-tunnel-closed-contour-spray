"""Expand and persist the Stage 5A model used by a runtime instance."""

from __future__ import annotations

import argparse
from pathlib import Path

from xacro import process_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xacro", type=Path, required=True)
    parser.add_argument("--initial-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = process_file(str(args.xacro), mappings={
        "initial_positions_file": str(args.initial_positions.resolve()),
        "tool_tcp_xyz": "0 0 0.150",
        "tool_tcp_rpy": "0 0 0",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document.toxml() + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
