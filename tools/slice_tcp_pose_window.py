from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_COLUMNS = ["x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz"]


def _wrapped_index(index: int, loop_size: int) -> int:
    return int(index) % int(loop_size)


def slice_window(
    source_csv: Path,
    out_csv: Path,
    center_phase: int,
    before: int,
    after: int,
    loop_size: int,
    close_window: bool = True,
    out_json: Path | None = None,
) -> dict[str, object]:
    with source_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = set(REQUIRED_COLUMNS) - set(fieldnames)
        if missing:
            raise ValueError(f"{source_csv} is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < loop_size:
        raise ValueError(f"{source_csv} has {len(rows)} rows, fewer than loop_size={loop_size}.")

    start_phase = int(center_phase) - int(before)
    end_phase = int(center_phase) + int(after)
    phases = [_wrapped_index(phase, loop_size) for phase in range(start_phase, end_phase + 1)]
    out_rows = []
    for local_index, phase in enumerate(phases):
        row = {column: rows[phase][column] for column in REQUIRED_COLUMNS}
        row["local_index"] = str(local_index)
        row["source_index"] = str(phase)
        row["source_phase_index"] = str(phase)
        row["is_closure_duplicate"] = "false"
        out_rows.append(row)
    if close_window and out_rows:
        closure = out_rows[0].copy()
        closure["local_index"] = str(len(out_rows))
        closure["is_closure_duplicate"] = "true"
        out_rows.append(closure)

    output_fields = [*REQUIRED_COLUMNS, "local_index", "source_index", "source_phase_index", "is_closure_duplicate"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    if close_window:
        interpretation = (
            "This is a closed local window for targeted IK diagnostics. The final duplicate row intentionally "
            "connects the window tail back to its head, so any failure at that closure row is a local-window "
            "artifact unless the same transition exists in the full contour."
        )
    else:
        interpretation = (
            "This is an open local window for targeted IK diagnostics. It preserves only adjacent source "
            "phases and does not add an artificial closure transition."
        )

    payload = {
        "source_csv": str(source_csv),
        "out_csv": str(out_csv),
        "center_phase": int(center_phase),
        "before": int(before),
        "after": int(after),
        "loop_size": int(loop_size),
        "close_window": bool(close_window),
        "samples_per_loop_for_validation": len(phases) if close_window else 0,
        "row_count": len(out_rows),
        "phase_indices": phases,
        "interpretation": f"{interpretation} It is not a formal full-contour production validation input.",
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice a closed local TCP-pose window around a contour phase.")
    parser.add_argument("--source-csv", type=Path, default=Path("outputs/tcp_poses_base_link.csv"))
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--center-phase", type=int, required=True)
    parser.add_argument("--before", type=int, default=8)
    parser.add_argument("--after", type=int, default=8)
    parser.add_argument("--loop-size", type=int, default=240)
    parser.add_argument("--no-close-window", action="store_true")
    args = parser.parse_args()

    payload = slice_window(
        args.source_csv,
        args.out_csv,
        args.center_phase,
        args.before,
        args.after,
        args.loop_size,
        close_window=not args.no_close_window,
        out_json=args.out_json,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
