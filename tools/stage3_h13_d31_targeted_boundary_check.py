"""One authorized targeted alpha diagnostic at the D31 update-427 boundary."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d30_d31_dual_mode_direction_recovery as d31

OUTPUT = ROOT / "outputs" / "stage3_h13_post_d30_d31_dual_mode_direction_recovery_20260823T235000+0800"
CHECKPOINT = OUTPUT / "checkpoints" / "committed_update_426.pt"


def main() -> int:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    d31.require(int(payload["completed_optimizer_step"]) == 426, "TARGETED_PARENT_UPDATE_MISMATCH")
    runtime = d31.d24.d17.preflight_runtime()
    model, optimizer = d31.d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = copy.deepcopy(payload["rng_state"])
    canonical = [float(value) for value in payload["canonical_base_lr"]]
    identity = d31.semantic_parent_identity(model, optimizer, rng)
    record, _, _, _ = d31.evaluated_candidate(
        runtime, model, optimizer, rng, canonical, identity, 426, 427,
        float(payload["H1"]), float(payload["H32"]), "B", "CAGRAD_H1_VS_REST", 0.4, 0.03125,
    )
    record["targeted_boundary_diagnostic"] = True
    record["diagnostic_justification"] = "single smaller-alpha local-boundary check after registered Mode-A/Mode-B families failed H32"
    records_path = OUTPUT / "D31_CANDIDATE_RESULTS.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records.append(record)
    d31.write_json(records_path, records)
    d31.write_csv(OUTPUT / "D31_CANDIDATE_RESULTS.csv", records)
    d31.write_json(OUTPUT / "TARGETED_UPDATE_427_BOUNDARY_DIAGNOSTIC.json", record)
    report_path = OUTPUT / "FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8").replace("CANDIDATES_SCREENED: 52", "CANDIDATES_SCREENED: 53")
    report += f"\n## TARGETED UPDATE-427 BOUNDARY CHECK\n\nCAGrad c=0.4, alpha=1/32: H1 {record['candidate_H1']:.17g}, delta H1 {record['delta_H1']:.17g}, H32 {record['candidate_H32']:.17g}, H32 margin {record['H32_margin']:.17g}, legal {record['candidate_valid']}. This was the single smaller-alpha diagnostic permitted to determine the local boundary direction.\n"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    (OUTPUT / "TASK_AND_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8", newline="\n")
    manifest_path = OUTPUT / "D31_EXECUTION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["targeted_update_427_boundary_diagnostic"] = {"family": "CAGRAD_H1_VS_REST", "c": 0.4, "alpha": 0.03125, "candidate_valid": record["candidate_valid"], "H1": record["candidate_H1"], "H32": record["candidate_H32"], "H32_margin": record["H32_margin"]}
    d31.write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "candidate_valid": record["candidate_valid"], "H1": record["candidate_H1"], "H32": record["candidate_H32"], "H32_margin": record["H32_margin"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
