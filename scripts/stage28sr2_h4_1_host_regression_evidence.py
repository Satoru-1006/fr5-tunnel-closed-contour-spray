"""Run fresh H4/H4.1 host regression and emit runner-compatible evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOCUSED_TESTS = [
    "tests/test_stage28sr2_h4_1_identity_migration.py",
    "tests/test_stage28sr2_h4_final_boundary.py",
    "tests/test_stage28sr2_production_integration.py",
    "tests/test_stage28sr2_r2_formal_runner.py",
    "tests/test_stage28sr2_r2_zero_goal_hardening.py",
]


def _run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    raw = completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else "")
    return completed.returncode, raw


def _counts(raw: str) -> dict[str, int | None]:
    def number(pattern: str) -> int:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    collected_match = re.search(r"(?P<count>\d+)\s+tests?\s+collected\b", raw, flags=re.IGNORECASE)
    passed = number(r"\b(\d+)\s+passed\b")
    failed = number(r"\b(\d+)\s+failed\b")
    errors = number(r"\b(\d+)\s+errors?\b")
    skipped = number(r"\b(\d+)\s+skipped\b")
    collected = int(collected_match.group("count")) if collected_match else passed + failed + errors + skipped
    return {"collected": collected, "passed": passed, "failed": failed, "errors": errors, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2.8S-R2 H4.1 host regression evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    focused_code, focused_raw = _run([sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS])
    full_code, full_raw = _run([sys.executable, "-m", "pytest", "-q"])
    evidence = "===== focused =====\n" + focused_raw + "\n===== full =====\n" + full_raw
    evidence_path = output / "h4_1_host_regression_evidence.txt"
    evidence_path.write_text(evidence, encoding="utf-8")
    summary = {
        "focused": {**_counts(focused_raw), "exit_code": focused_code, "passed_gate": focused_code == 0},
        "full": {**_counts(full_raw), "exit_code": full_code, "passed_gate": full_code == 0},
        "evidence_path": str(evidence_path),
    }
    (output / "h4_1_host_regression_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["focused"]["passed_gate"] and summary["full"]["passed_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
