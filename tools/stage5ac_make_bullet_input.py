"""Build isolated native Bullet interval input and translated world parts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("nominal_parts", type=Path)
    parser.add_argument("environment_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.trajectory_csv.open(encoding="utf-8-sig", newline="")))
    if len(rows) < 2:
        raise RuntimeError("stage5ac_bullet_input_trajectory_too_short")
    env = json.loads(args.environment_json.read_text(encoding="utf-8"))
    variant = env.get("stage5ac_variant", {})
    translation = [float(x) for x in variant.get("translation_m", [0.0, 0.0, 0.0])]
    if abs(float(variant.get("yaw_rad", 0.0))) > 1.0e-12:
        raise RuntimeError("stage5ac_bullet_input_yaw_variant_requires_rotated_stl_materializer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    interval = args.output_dir / "STAGE5AC_BULLET_INTERVALS.csv"
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary"] + [f"q0_{i}" for i in range(6)] + [f"q1_{i}" for i in range(6)]
    with interval.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, (a, b) in enumerate(zip(rows[:-1], rows[1:])):
            row = {"interval_index": i, "time_start_s": float(a["t"]), "time_end_s": float(b["t"]), "process_order_index": i, "spray_state": "STAGE5AC_SHADOW", "process_kind": "stage5ac_shadow", "segment_id": str(i), "transition_id": "", "source_boundary": "stage5ac_waypoint_or_ruckig_sample"}
            row.update({f"q0_{j}": float(a[f"j{j+1}_q"]) for j in range(6)})
            row.update({f"q1_{j}": float(b[f"j{j+1}_q"]) for j in range(6)})
            writer.writerow(row)
    parts = args.output_dir / "STAGE5AC_BULLET_PARTS"
    parts.mkdir(exist_ok=True)
    for source in sorted(args.nominal_parts.glob("*.stl")):
        target = parts / source.name
        with source.open(encoding="ascii") as inp, target.open("w", encoding="ascii", newline="\n") as out:
            for line in inp:
                stripped = line.strip()
                if stripped.lower().startswith("vertex "):
                    tokens = stripped.split()
                    x, y, z = (float(tokens[k]) for k in (1, 2, 3))
                    out.write(f"      vertex {x + translation[0]:.17g} {y + translation[1]:.17g} {z + translation[2]:.17g}\n")
                else:
                    out.write(line)
    manifest = {"schema_version": "stage5ac-bullet-input-v1", "trajectory_csv": str(args.trajectory_csv.resolve()), "state_count": len(rows), "interval_count": len(rows) - 1, "environment_json": str(args.environment_json.resolve()), "translation_m": translation, "nominal_parts": str(args.nominal_parts.resolve()), "parts_dir": str(parts.resolve()), "interval_csv": str(interval.resolve()), "source_environment_unchanged": True}
    (args.output_dir / "STAGE5AC_BULLET_INPUT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
