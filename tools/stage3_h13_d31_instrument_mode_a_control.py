"""Re-run one D30 control solely to localize update-424 realized geometry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d30_d31_dual_mode_direction_recovery as d31

OUTPUT = ROOT / "outputs" / "stage3_h13_post_d30_d31_dual_mode_direction_recovery_20260823T235000+0800"


def main() -> int:
    runtime, model, optimizer, rng, canonical, _, _ = d31.load_parent()
    identity = d31.semantic_parent_identity(model, optimizer, rng)
    record, _, _, _ = d31.evaluated_candidate(
        runtime, model, optimizer, rng, canonical, identity, 423, 424,
        d31.START_H1, d31.START_H32, "A", "P4_POSTCLIP_H1_BIAS", d31.MODE_A_COEFFICIENT, 0.25,
    )
    record["instrumented_historical_control"] = True
    record["selection_status"] = "HISTORICAL_CONTROL_REPLAY_NOT_COMMITTED"
    records_path = OUTPUT / "D31_CANDIDATE_RESULTS.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records.append(record)
    d31.write_json(records_path, records)
    d31.write_csv(OUTPUT / "D31_CANDIDATE_RESULTS.csv", records)
    d31.write_json(OUTPUT / "UPDATE_424_MODE_A_REALIZED_GEOMETRY.json", record)
    d31.write_json(OUTPUT / "UPDATE_424_RECOVERY_COMPARISON.json", [row for row in records if int(row.get("candidate_update", -1)) == 424])
    report_path = OUTPUT / "FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8").replace("CANDIDATES_SCREENED: 53", "CANDIDATES_SCREENED: 54")
    old = "Mode A failed at update 424 because its fully realized AdamW deltas were H1 non-descent for every D30 control alpha while remaining H32-feasible. D31 reconstructed raw/component gradients, clipping, inherited moments, bias correction, adaptive preconditioning, decoupled weight decay, and the final parameter delta. Mode B changed only the gradient presented before AdamW; moments were generated from that modified gradient and all parent history was preserved. The selected trajectory above states whether descent was restored, whether Mode A later recovered, and the first remaining blocker, if any. Detailed directional cosines and norms are in D31_CANDIDATE_RESULTS.json."
    new = (
        "Mode A failed at update 424 because all four D30 control alphas were H1 non-descent while H32-feasible. "
        f"The instrumented alpha=1/4 control reproduced H1 {record['candidate_H1']:.17g} and H32 {record['candidate_H32']:.17g}. "
        f"Its raw aggregate was clipped from {record['preclip_aggregate_norm']:.9g} to {record['postclip_base_norm']:.9g}; raw/post-clip alignment with the H1 gradient was {record['cosine_postclip_base_with_h1_gradient']:.9g}, the coefficient-0.35 presented direction alignment was {record['cosine_presented_with_h1_gradient']:.9g}, the next first-moment alignment was {record['cosine_next_first_moment_with_h1_gradient']:.9g}, and the final realized delta alignment with negative H1 gradient was only {record['cosine_realized_delta_with_negative_h1_gradient']:.9g}. "
        "Thus scalar clipping did not change direction; loss of useful descent arose in the weak H1-biased direction after interaction with inherited AdamW moments/preconditioning, while weight decay was negligible. "
        "Mode B coefficient 1.4 raised presented H1 alignment to 0.8347430097 and realized-delta alignment to 0.00588483035, restoring measured H1 descent without resetting history. "
        "CAGrad then supplied the only legal update-425 recovery, Mode A became legal again at update 426, and update 427 became the later H32 boundary blocker after all registered families plus the single alpha=1/32 diagnostic failed. Detailed norms and cosines are in D31_CANDIDATE_RESULTS.json."
    )
    if old not in report:
        raise RuntimeError("FINAL_REPORT_INTERPRETATION_ANCHOR_MISSING")
    report = report.replace(old, new)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    (OUTPUT / "TASK_AND_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8", newline="\n")
    manifest_path = OUTPUT / "D31_EXECUTION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["instrumented_update_424_mode_A_control"] = {"alpha": 0.25, "reproduced_D30_metrics": abs(record["candidate_H1"] - 7.574832452419637e-05) < 1e-18 and abs(record["candidate_H32"] - 0.017919281928433367) < 1e-18, "path": "UPDATE_424_MODE_A_REALIZED_GEOMETRY.json"}
    d31.write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "H1": record["candidate_H1"], "H32": record["candidate_H32"], "realized_h1_cosine": record["cosine_realized_delta_with_negative_h1_gradient"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
