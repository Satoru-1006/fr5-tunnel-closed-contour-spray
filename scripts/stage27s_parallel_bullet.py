"""Run Stage 2.7S candidate collision intervals in isolated ROS domains."""

from __future__ import annotations

import concurrent.futures
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage26 as stage26


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: stage27s_parallel_bullet.py OUTPUT_DIR [WORKERS]")
    output = Path(sys.argv[1]).resolve()
    workers = int(sys.argv[2]) if len(sys.argv) == 3 else 8
    if workers < 2 or workers > 8:
        raise SystemExit("workers must be between 2 and 8")
    interval_csv = output / "stage27s_bullet_intervals.csv"
    rows = list(csv.DictReader(interval_csv.open(encoding="utf-8", newline="")))
    root = output / "stage27s_native_bullet_parallel"
    root.mkdir(exist_ok=False)
    chunks = []
    base, remainder = divmod(len(rows), workers)
    start = 0
    for worker in range(workers):
        size = base + (1 if worker < remainder else 0)
        subset = rows[start:start + size]
        worker_dir = root / f"worker_{worker:02d}"
        worker_dir.mkdir()
        chunk_csv = worker_dir / "intervals.csv"
        with chunk_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(subset)
        chunks.append((worker, worker_dir, chunk_csv, start, start + size))
        start += size

    def run_one(item: tuple[int, Path, Path, int, int]) -> dict[str, object]:
        worker, worker_dir, chunk_csv, begin, end = item
        result = stage26.run_native(
            worker_dir, chunk_csv, "bullet", worker + 1, False, False,
            ros_domain_id=70 + worker,
        )
        return {
            "worker": worker,
            "begin": begin,
            "end_exclusive": end,
            "output_dir": str(worker_dir.resolve()),
            "result": result,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, chunks))

    all_rows = []
    malformed = []
    worker_summaries = []
    for item in results:
        worker = int(item["worker"])
        worker_dir = Path(str(item["output_dir"]))
        path = worker_dir / "stage26_continuous_robot_world_intervals.jsonl"
        worker_rows = []
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    worker_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed.append({"worker": worker, "line": line_number})
        all_rows.extend(worker_rows)
        worker_summaries.append({
            "worker": worker,
            "expected": int(item["end_exclusive"]) - int(item["begin"]),
            "observed": len(worker_rows),
            "exit_code": item["result"].get("exit_code"),
            "raw": str(path.resolve()),
        })
    all_rows.sort(key=lambda row: int(row.get("interval_index", -1)))
    aggregate_raw = output / "stage27s_bullet_robot_world_intervals.jsonl"
    with aggregate_raw.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    expected = len(rows)
    indices = [int(row.get("interval_index", -1)) for row in all_rows]
    collisions = [
        row for row in all_rows
        if row.get("continuous_collision") or row.get("endpoint_collision")
    ]
    skipped = [row for row in all_rows if row.get("ccd_api_called") is not True]
    complete = bool(
        len(all_rows) == expected
        and len(set(indices)) == expected
        and set(indices) == set(range(expected))
        and not malformed
        and not skipped
        and all(item["result"].get("exit_code") == 0 for item in results)
    )
    evidence = {
        "schema_version": "stage27s-bullet-validation-v2",
        "backend": "native Bullet robot-world collision API",
        "collision_method": "adaptive_discrete_interpolation",
        "strict_continuous_collision_detection": "not_available",
        "execution_strategy": "non-overlapping interval chunks in independent ROS domains",
        "workers": workers,
        "worker_summaries": worker_summaries,
        "validation_complete": complete,
        "checked_intervals": len(all_rows),
        "expected_intervals": expected,
        "unique_interval_indices": len(set(indices)),
        "missing_interval_count": len(set(range(expected)) - set(indices)),
        "malformed_raw_lines": malformed,
        "skipped": len(skipped),
        "collisions": len(collisions),
        "first_collision": collisions[0] if collisions else None,
        "aggregate_raw": str(aggregate_raw.resolve()),
    }
    evidence["passed"] = bool(complete and not collisions)
    evidence["status"] = "passed" if evidence["passed"] else "blocked"
    write_json(output / "stage27s_bullet_validation.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False))
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
