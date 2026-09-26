from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h12_r2_process_manifold_post_ruckig import (
    H10_ROOT,
    H11_ROOT,
    H12_R_ROOT,
    ROLES,
    build_contexts,
    build_repair_and_units,
    constant_velocity_prediction,
    frozen_e2_model,
    global_by_id,
    load_h10_segments_fast,
    load_json,
    metric_payload,
    predict_residual_model,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    windows = load_json(H11_ROOT / "h11_window_manifest.json")["windows"]
    h11_stats = load_json(H11_ROOT / "normalization_stats.json")
    contexts = build_contexts(H10_ROOT, H11_ROOT, windows, h11_stats)
    selected = load_json(H12_R_ROOT / "selected_model_manifest.json")
    checkpoint = Path(str(selected["checkpoint"]))
    if sha256_file(checkpoint) != str(selected.get("checkpoint_sha256")):
        raise RuntimeError("replay_checkpoint_hash_mismatch")
    model = frozen_e2_model(checkpoint)
    raw_by_role: dict[str, np.ndarray] = {}
    cv_by_role: dict[str, np.ndarray] = {}
    for role in ROLES:
        arrays = contexts[role]["arrays"]
        cv_by_role[role] = constant_velocity_prediction(arrays)
        raw_by_role[role], _ = predict_residual_model(model, arrays, cv_by_role[role], batch_size=2048)

    summary, windows, repaired, cv_global = build_repair_and_units(
        out.parent,
        contexts,
        windows,
        raw_by_role,
        cv_by_role,
        load_h10_segments_fast(H10_ROOT),
        persist=False,
    )
    target_by_role = {role: contexts[role]["arrays"].target_positions for role in ROLES}
    raw_metrics: dict[str, dict] = {}
    repaired_metrics: dict[str, dict] = {}
    for role in ROLES:
        ids = [int(row["window_id"]) for row in contexts[role]["window_metadata"]]
        raw_metrics[role] = metric_payload(raw_by_role[role], target_by_role[role])
        repaired_metrics[role] = metric_payload(repaired[ids], target_by_role[role])
    payload = {
        "schema_version": "stage3_h12_r2_replay_worker_v1",
        "TOTAL_WINDOWS": len(windows),
        "checkpoint_sha256": sha256_file(checkpoint),
        "repaired_semantic_sha256": summary["repaired_semantic_sha256"],
        "raw_semantic_sha256": summary["raw_semantic_sha256"],
        "tier_counts": summary["tier_counts"],
        "metrics": {
            "CV_BASELINE_RMSE": float(metric_payload(cv_by_role["TEST"], target_by_role["TEST"])["joint_position_rmse_rad"]),
            "E2_RAW_NEURAL_RMSE": float(raw_metrics["TEST"]["joint_position_rmse_rad"]),
            "PRE_RUCKIG_REPAIRED_RMSE": float(repaired_metrics["TEST"]["joint_position_rmse_rad"]),
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
