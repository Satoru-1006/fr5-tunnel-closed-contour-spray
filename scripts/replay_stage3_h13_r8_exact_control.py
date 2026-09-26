#!/usr/bin/env python3
"""Replay the historical R8 chunked candidate with its original batch policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_model import set_deterministic  # noqa: E402
from src.stage3_h13_r8 import bounded_direct_prediction, rollout_rmse, train_reference_anchored_model  # noqa: E402
from scripts.stage3_h13_r8_autoregressive_state_semantics import (  # noqa: E402
    H11_R2_CHECKPOINT,
    MAX_HORIZON,
    SEED,
    initial_model,
    load_train_validation,
    train_derived_residual_bound,
)


def main() -> int:
    import torch

    torch.set_num_threads(max(2, min(8, int(torch.get_num_threads()))))
    set_deterministic(SEED + 2)
    _, stats, train, validation, _ = load_train_validation()
    bound = train_derived_residual_bound(train)["bound_rad"]
    config = {
        "batch_size": 512,
        "eval_batch_size": 4096,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-5,
        "max_epochs": 2,
        "patience": 1,
        "w_direct": 1.0,
        "w_rollout": 0.75,
        "rollout_horizon": MAX_HORIZON,
        "huber_beta": 1.0e-3,
        "gradient_clip_norm": 1.0,
        "training_rollout_horizon": MAX_HORIZON,
    }
    model = initial_model(H11_R2_CHECKPOINT)
    model, history, best_epoch, _ = train_reference_anchored_model(
        model,
        train,
        validation,
        stats["channels"],
        config,
        semantics="chunked_multihorizon",
        residual_bound=bound,
        device="cpu",
    )
    _, direct = bounded_direct_prediction(model, validation, bound, device="cpu", batch_size=4096)
    rollout = rollout_rmse(model, validation, stats["channels"], semantics="chunked_multihorizon", residual_bound=bound, horizons=(4, 8, 16, 32), device="cpu", batch_size=4096)
    print(json.dumps({"best_epoch": best_epoch, "history": history, "direct_rmse": direct["joint_position_rmse_rad"], "rollout": rollout, "batch_size": 512, "threads": torch.get_num_threads()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
