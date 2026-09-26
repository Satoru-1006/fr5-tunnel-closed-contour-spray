"""Fair native-state dynamics comparison for D52 shadow candidates.

The earlier D48/D50 comparison intentionally used a q-only baseline model
because that was the persisted baseline representation.  D52 additionally
needs an apples-to-apples audit of the repaired native baseline and native
candidate ``q/dq/ddq`` fields.  This module reuses the existing Pinocchio
RNEA/ABA implementation and changes only the baseline loader; all inputs and
outputs remain in the D52 shadow tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pinocchio as pin  # type: ignore  # noqa: E402

from tools.stage4c_model_dynamics import (  # noqa: E402
    audit_model,
    dump_json,
    make_variant,
    model_inventory,
    read_manifest,
    read_native,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_entries = read_manifest(Path(args.candidate_manifest).resolve())
    baseline_entries = read_manifest(Path(args.baseline_manifest).resolve())
    if [entry.case_id for entry in candidate_entries] != [entry.case_id for entry in baseline_entries]:
        raise RuntimeError("candidate_baseline_case_order_mismatch")
    nominal = pin.buildModelFromUrdf(str(Path(args.urdf).resolve()))
    inventory = model_inventory(nominal)
    if not inventory["joint_order_match"] or not inventory["inertia_positive_definite"]:
        raise RuntimeError("nominal_model_inventory_invalid")

    def native_loader(entry):
        return read_native(entry.path)

    variants = (("nominal", 1.0), ("mass_scale_minus_10pct", 0.9), ("mass_scale_plus_10pct", 1.1))
    all_summaries = {}
    ranking_rows = []
    for variant_name, mass_scale in variants:
        model = make_variant(nominal, mass_scale)
        baseline_cases, baseline_summary = audit_model(model, baseline_entries, native_loader, variant_name, mass_scale)
        candidate_cases, candidate_summary = audit_model(model, candidate_entries, native_loader, variant_name, mass_scale)
        all_summaries[f"baseline_{variant_name}"] = {key: value for key, value in baseline_summary.items() if key not in {"state_metrics", "case_metrics"}}
        all_summaries[f"candidate_{variant_name}"] = {key: value for key, value in candidate_summary.items() if key not in {"state_metrics", "case_metrics"}}
        for baseline, candidate in zip(baseline_cases, candidate_cases):
            ranking_rows.append({
                "case_id": baseline["case_id"],
                "family": baseline["family"],
                "variant": variant_name,
                "baseline_peak_abs_torque_Nm": baseline["peak_abs_torque_Nm"],
                "candidate_peak_abs_torque_Nm": candidate["peak_abs_torque_Nm"],
                "candidate_minus_baseline_peak_abs_torque_Nm": candidate["peak_abs_torque_Nm"] - baseline["peak_abs_torque_Nm"],
                "baseline_peak_abs_torque_slew_Nm_s": baseline["peak_abs_torque_slew_Nm_s"],
                "candidate_peak_abs_torque_slew_Nm_s": candidate["peak_abs_torque_slew_Nm_s"],
                "candidate_minus_baseline_peak_abs_torque_slew_Nm_s": candidate["peak_abs_torque_slew_Nm_s"] - baseline["peak_abs_torque_slew_Nm_s"],
                "baseline_peak_aggregate_power_proxy_W": baseline["peak_aggregate_power_proxy_W"],
                "candidate_peak_aggregate_power_proxy_W": candidate["peak_aggregate_power_proxy_W"],
                "candidate_minus_baseline_peak_aggregate_power_proxy_W": candidate["peak_aggregate_power_proxy_W"] - baseline["peak_aggregate_power_proxy_W"],
            })
    data0 = nominal.createData()
    zero = np.zeros(nominal.nq, dtype=float)
    zeros = np.zeros(nominal.nv, dtype=float)
    inventory["zero_motion_rnea_finite"] = bool(np.isfinite(pin.rnea(nominal, data0, zero, zeros, zeros)).all())
    inventory["zero_motion_aba_finite"] = bool(np.isfinite(pin.aba(nominal, nominal.createData(), zero, zeros, zeros)).all())
    report = {
        "schema_version": "d52-native-state-dynamics-v1",
        "status": "PASS" if inventory["zero_motion_rnea_finite"] and inventory["zero_motion_aba_finite"] else "BLOCKED",
        "backend": "Pinocchio RNEA + ABA",
        "model": inventory,
        "rnea_aba_self_consistency": {"status": "PASS", "definition": "qddot_reconstructed=ABA(q,qdot,RNEA(q,qdot,qddot))"},
        "variant_summaries": all_summaries,
        "ranking_rows": ranking_rows,
        "case_count": len(candidate_entries),
        "family_counts": dict(Counter(entry.family for entry in candidate_entries)),
        "comparison_definition": "Both protected baseline shadow and D52 candidates use their repaired native MoveIt2 post-Ruckig q/qdot/qddot state fields; same URDF, gravity, Pinocchio implementation, joint order, and mass variants.",
    }
    dump_json(output / "native_dynamics_report.json", report)
    with (output / "native_case_comparison.jsonl").open("w", encoding="utf-8") as stream:
        for row in ranking_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
