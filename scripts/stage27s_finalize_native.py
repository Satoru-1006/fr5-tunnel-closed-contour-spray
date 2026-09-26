"""Finalize an already completed Stage 2.7S native execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage27s_native_spline_remediation as stage27s


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage27s_finalize_native.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    trajectory = stage27s.read_trajectory(output / "stage27s_candidate_trajectory.csv")
    certificate = read_json(output / "stage27s_exact_jtc_certificate.json")
    coefficients = [
        np.asarray(row["coefficients_c0_to_c5_by_joint"], dtype=float)
        for row in (
            json.loads(line)
            for line in (output / "stage27s_exact_jtc_coefficients.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    exact = {"summary": certificate, "coefficients": coefficients}
    phase_a = read_json(output / "stage27s_phase_a_report.json")
    geometry = read_json(output / "stage27s_process_geometry.json")
    bullet = read_json(output / "stage27s_bullet_validation.json")
    native = read_json(output / "stage27r_clean_action_execution.json")
    native_oracle = stage27s.compare_native(output, trajectory, exact, native)
    determinism = stage27s.build_determinism(output, trajectory, exact, geometry, bullet, "pending")
    source_hash = stage27s.sha256(stage27s.R2_TRAJECTORY)
    gate = stage27s.build_gate(
        output, source_hash, phase_a, exact, geometry, bullet, native,
        native_oracle, determinism,
    )
    determinism = stage27s.build_determinism(
        output, trajectory, exact, geometry, bullet,
        gate["Stage_2_7"]["status"],
    )
    gate = stage27s.build_gate(
        output, source_hash, phase_a, exact, geometry, bullet, native,
        native_oracle, determinism,
    )
    stage27s.write_sums(output)
    print(json.dumps({
        "output_root": str(output),
        "Stage_2_7": gate["Stage_2_7"]["status"],
        "first_real_blocker": gate.get("first_real_blocker"),
        "native_oracle": native_oracle.get("summary", {}),
    }, ensure_ascii=False, indent=2))
    return 0 if gate["Stage_2_7"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
