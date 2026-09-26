from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _float_or_none(value: str) -> float | None:
    value = str(value).strip()
    if not value:
        return None
    return float(value)


def _int_or_zero(value: str) -> int:
    value = str(value).strip()
    return int(value) if value else 0


def summarize_diagnostics(csv_path: Path) -> dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "source": str(csv_path),
            "status": "empty",
            "row_count": 0,
            "last_index": -1,
            "last_status": "",
        }

    elapsed = [_float_or_none(row.get("elapsed_s", "")) or 0.0 for row in rows]
    elapsed_step = [elapsed[0], *[max(0.0, elapsed[i] - elapsed[i - 1]) for i in range(1, len(elapsed))]]
    slowest_index = max(range(len(rows)), key=lambda i: elapsed_step[i])
    failure_rows = [row for row in rows if row.get("status") not in {"ok", ""}]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    bottleneck_values = [
        value
        for row in ok_rows
        if (value := _float_or_none(row.get("best_bottleneck_step_deg", ""))) is not None
    ]
    candidate_failure_counts = [_int_or_zero(row.get("candidate_failure_count", "")) for row in rows]
    colliding_counts = [_int_or_zero(row.get("colliding_candidate_count", "")) for row in rows]
    last = rows[-1]
    return {
        "source": str(csv_path),
        "status": "fail" if failure_rows else "pass",
        "row_count": len(rows),
        "completed_ok_count": len(ok_rows),
        "last_index": _int_or_zero(last.get("index", "")),
        "last_status": last.get("status", ""),
        "last_detail": last.get("detail", ""),
        "elapsed_s": elapsed[-1],
        "slowest_index": _int_or_zero(rows[slowest_index].get("index", "")),
        "slowest_elapsed_step_s": elapsed_step[slowest_index],
        "max_candidate_failure_count": max(candidate_failure_counts, default=0),
        "max_colliding_candidate_count": max(colliding_counts, default=0),
        "best_bottleneck_step_deg_last_ok": bottleneck_values[-1] if bottleneck_values else None,
        "best_bottleneck_step_deg_max_ok": max(bottleneck_values, default=None),
        "failure_rows": [
            {
                "index": _int_or_zero(row.get("index", "")),
                "status": row.get("status", ""),
                "detail": row.get("detail", ""),
                "elapsed_s": _float_or_none(row.get("elapsed_s", "")),
            }
            for row in failure_rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MoveIt2 IK search diagnostics CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    payload = summarize_diagnostics(args.csv_path)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote IK search diagnostics summary to {args.out_json}")


if __name__ == "__main__":
    main()
