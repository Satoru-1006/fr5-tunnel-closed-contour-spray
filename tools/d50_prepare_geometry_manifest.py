"""Prepare an isolated q-only manifest for the native D41 geometry replay."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    return f"/mnt/{resolved.drive.rstrip(':').lower()}{resolved.as_posix()[2:]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    for source in sorted((args.candidate_root / "native_postprocess" / "trajectories").glob("*.csv")):
        case_id = source.stem
        target = args.output_root / "geometry_inputs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            with target.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, lineterminator="\n")
                writer.writerow(["waypoint", *[f"j{i}_q" for i in range(1, 7)]])
                for index, row in enumerate(reader):
                    writer.writerow([index, *[float(row[f"j{i}_q"]) for i in range(1, 7)]])
        family = case_id.rsplit("_", 1)[0].upper()
        if family == "COLLISION":
            family = "COLLISION_SENSITIVE"
        rows.append((case_id, wsl_path(target), family))
    manifest = args.output_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        stream.write("case_id,trajectory_csv,family\n")
        stream.writelines(f"{case_id},{trajectory},{family}\n" for case_id, trajectory, family in rows)
    if len(rows) != 12:
        raise SystemExit(f"expected 12 candidate trajectories, got {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
