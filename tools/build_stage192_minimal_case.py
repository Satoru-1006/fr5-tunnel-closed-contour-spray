#!/usr/bin/env python3
"""Build the Stage 1.9.2 bytewise snapshot from the frozen Stage 1.9.1 case."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.ik_solver_binary_snapshot import build_snapshot, sha256_file


PROTECTED = ["outputs/ik_graph", "outputs/ik_graph_diagnostics", "outputs/ik_feasibility_ablation_stage1_6", "outputs/ik_graph_stage17", "outputs/ik_graph_stage18", "outputs/ik_graph_stage19", "outputs/ik_graph_stage191"]


def protected_hashes() -> dict:
    files = {}
    for rel in PROTECTED:
        directory = ROOT / rel
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"schema_version": "1.0", "files": files}


def main() -> int:
    output = ROOT / "outputs/ik_graph_stage192"
    output.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    (output / "protected_output_hashes_before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    result = build_snapshot(ROOT, output / "minimal_case")
    print(json.dumps({"call_id": result["target_call_id"], "corpus_match": result["input_manifest"]["corpus_match"], "protected_file_count": len(before["files"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
