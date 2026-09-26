"""Run the Stage 2.7R dense Bullet probe with an extended orchestration timeout."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage26 as stage26
import scripts.stage27r_run as run


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: stage27r_long_bullet.py OUTPUT_DIR [NATIVE_DIR_NAME]")
    output = Path(sys.argv[1]).resolve()
    interval_csv = output / "stage27r_bullet_intervals.csv"
    native_dir = output / (sys.argv[2] if len(sys.argv) == 3 else "native_bullet_longrun")
    stage26.PARTS = run.PARTS_DIR
    result = stage26.run_native(native_dir, interval_csv, "bullet", 1, False, False)
    rows_path = native_dir / "stage26_continuous_robot_world_intervals.jsonl"
    rows = []
    malformed = []
    if rows_path.exists():
        for line_number, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed.append(line_number)
    expected = sum(1 for _ in csv.DictReader(interval_csv.open(encoding="utf-8", newline="")))
    collisions = [row for row in rows if row.get("continuous_collision") or row.get("endpoint_collision")]
    skipped = [row for row in rows if row.get("ccd_api_called") is not True]
    native_summary = run.read_json(native_dir / "stage26_native_summary.json") if (native_dir / "stage26_native_summary.json").exists() else {}
    evidence = {
        "schema_version": "stage27r-bullet-dense-validation-v1",
        "backend": "native Bullet robot-world CCD",
        "validation_complete": bool(result.get("exit_code") == 0 and len(rows) == expected and not malformed and not skipped),
        "checked_intervals": len(rows),
        "expected_intervals": expected,
        "collisions": len(collisions),
        "first_collision": collisions[0] if collisions else None,
        "skipped": len(skipped),
        "malformed_raw_lines": malformed,
        "native_runner_exit_code": result.get("exit_code"),
        "native_summary": native_summary,
        "raw_directory": str(native_dir.resolve()),
        "supersedes_partial_native_directory": str((output / "native_bullet").resolve()),
    }
    evidence["status"] = "passed" if evidence["validation_complete"] and evidence["collisions"] == 0 else "blocked"
    run.write_json(output / "stage27r_bullet_dense_validation.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False))
    return 0 if evidence["validation_complete"] and evidence["collisions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
