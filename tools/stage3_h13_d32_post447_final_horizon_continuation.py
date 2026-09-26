"""Resume the interrupted D32 tail exactly from committed update 447."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_d32_post430_h1_bias_recovery_continuation as base


SOURCE_ROOT = ROOT / "outputs" / "stage3_h13_d32_post430_h1_bias_recovery_20260824T110000+0800"
PARENT = SOURCE_ROOT / "checkpoints" / "committed_update_447.pt"
EXPECTED_PARENT_SHA256 = "3973815d64dc4d0ecc8a26ddda1f708fddd05bb9bedfee47a3ddd88a28cded13"
EXPECTED_PARENT_PARENT_SHA256 = "a9c54ea70f4683fc350e844ecc523ebaf31786c67762b477f00484bb994a05e9"
START_UPDATE = 447
START_H1 = 7.288555643116757e-05
START_H32 = 0.018031493201067752


def load_parent() -> tuple[dict[str, Any], Any, Any, dict[str, Any], list[float], dict[str, Any], list[dict[str, Any]]]:
    base.require(PARENT.is_file(), f"PARENT_MISSING:{PARENT}")
    base.require(base.d32.d31.sha256_file(PARENT) == EXPECTED_PARENT_SHA256, "UPDATE_447_CHECKPOINT_SHA256_MISMATCH")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    base.require(int(payload["completed_optimizer_step"]) == START_UPDATE, "PARENT_COMPLETED_STEP_MISMATCH")
    base.require(int(payload["next_schedule_index"]) == START_UPDATE, "PARENT_SCHEDULE_INDEX_MISMATCH")
    base.require(all(int(state.get("step", -1)) == START_UPDATE for state in payload["optimizer_state_dict"]["state"].values()), "PARENT_OPTIMIZER_STEP_MISMATCH")
    base.require(int(payload["parent_update"]) == START_UPDATE - 1, "PARENT_CHAIN_UPDATE_MISMATCH")
    base.require(str(payload["parent_checkpoint_sha256"]) == EXPECTED_PARENT_PARENT_SHA256, "PARENT_CHAIN_SHA256_MISMATCH")
    base.require(float(payload["H1"]) == START_H1 and float(payload["H32"]) == START_H32, "PARENT_METRIC_MISMATCH")
    runtime = base.d32.d24.d17.preflight_runtime()
    model, optimizer = base.d32.d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = copy.deepcopy(payload["rng_state"])
    canonical = [float(value) for value in payload["canonical_base_lr"]]
    base.require(canonical == [base.d32.BASE_LR], "CANONICAL_LR_MISMATCH")

    post429_root = ROOT / "outputs" / "stage3_h13_d32_post429_tailor_forward_20260824T083000+0800"
    combined = list(json.loads((post429_root / "D32_CONTINUATION_TRAJECTORY.json").read_text(encoding="utf-8"))["combined_post426_trajectory"])
    for update in range(431, START_UPDATE + 1):
        metric_path = SOURCE_ROOT / "metrics" / f"update_{update}.json"
        base.require(metric_path.is_file(), f"MISSING_COMMITTED_METRIC:{update}")
        combined.append(json.loads(metric_path.read_text(encoding="utf-8"))["trajectory_row"])
    base.require([int(row["update"]) for row in combined] == list(range(427, START_UPDATE + 1)), "RECONSTRUCTED_TRAJECTORY_NOT_CONTIGUOUS")
    previous_update = 426
    previous_sha = base.d32.EXPECTED_PARENT_SHA256
    previous_h1 = base.d32.START_H1
    previous_h32 = base.d32.START_H32
    for row in combined:
        update = int(row["update"])
        base.require(int(row["parent"]) == previous_update, f"RECONSTRUCTED_PARENT_UPDATE_MISMATCH:{update}")
        base.require(str(row["parent_checkpoint_sha256"]) == previous_sha, f"RECONSTRUCTED_PARENT_SHA256_MISMATCH:{update}")
        base.require(float(row["H1_before"]) == previous_h1, f"RECONSTRUCTED_H1_LINK_MISMATCH:{update}")
        base.require(float(row["H32_before"]) == previous_h32, f"RECONSTRUCTED_H32_LINK_MISMATCH:{update}")
        previous_update = update
        previous_sha = str(row["checkpoint_sha256"])
        previous_h1 = float(row["H1"])
        previous_h32 = float(row["H32"])
    base.require(str(combined[-1]["checkpoint_sha256"]) == EXPECTED_PARENT_SHA256, "RECONSTRUCTED_TRAJECTORY_TAIL_SHA256_MISMATCH")
    d31_prior = json.loads((base.d32.D31_ROOT / "D31_CANONICAL_TRAJECTORY.json").read_text(encoding="utf-8"))["trajectory"]
    return runtime, model, optimizer, rng, canonical, {"trajectory": list(d31_prior) + list(combined)}, combined


def configure_base() -> None:
    base.SOURCE_ROOT = SOURCE_ROOT
    base.PARENT = PARENT
    base.EXPECTED_PARENT_SHA256 = EXPECTED_PARENT_SHA256
    base.EXPECTED_PARENT_PARENT_SHA256 = EXPECTED_PARENT_PARENT_SHA256
    base.START_UPDATE = START_UPDATE
    base.START_H1 = START_H1
    base.START_H32 = START_H32
    base.INITIAL_PREVIOUS_ALPHA = 0.015625
    base.INITIAL_PREVIOUS_BETA = 0.35
    base.load_parent = load_parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-update", type=int, default=450)
    args = parser.parse_args()
    configure_base()
    try:
        result = base.run(args.output.resolve(), args.max_update)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
