"""D36 matched capability evaluation for frozen Stage-3 trajectory checkpoints.

This is evaluation-only.  It never constructs an optimizer or writes a checkpoint.
The available learned system is a joint-trajectory predictor, so the capability
claims are deliberately limited to held-out prediction precision, long-horizon
stability, and robustness to joint-position observation noise.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage3_h13_d36_system_evaluation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as d6  # noqa: E402
from src.stage3_h11_dataset import FEATURE_NAMES  # noqa: E402
from src.stage3_h13_r6 import build_sequence_arrays  # noqa: E402
from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17  # noqa: E402

HORIZONS = (1, 4, 8, 16, 32)
ROLES = ("VALIDATION", "TEST", "GENERALIZATION")
NOISE_RAD = (0.0, 0.002, 0.01)
SAMPLE_COUNT = 1024
SEED = 360480
PANEL = (
    (413, "D29", 7.921091802345304e-05, 0.017350175045219574,
     ROOT / "outputs/stage3_h13_post_d28_d29_trust_region_reexpansion_20260823T170000+0800/checkpoints/committed_update_413.pt"),
    (424, "D31", 7.495387011797012e-05, 0.01849691401013345,
     ROOT / "outputs/stage3_h13_post_d30_d31_dual_mode_direction_recovery_20260823T235000+0800/checkpoints/committed_update_424.pt"),
    (450, "D32", 7.2765443365386504e-05, 0.018090939364518988,
     ROOT / "outputs/D33_FINAL_CROSS_CHAT_HANDOFF_20260825T001128+0800/d32_parent_context/committed_update_450.pt"),
    (469, "D34", 7.173430599053105e-05, 0.01848121489533577,
     ROOT / "outputs/stage3_h13_d34_champion_breakthrough_search_20260825T120000+0800/checkpoints/committed_update_469.pt"),
    (480, "D35", 7.089328839013259e-05, 0.01849100619381173,
     ROOT / "outputs/stage3_h13_d35_permanent_champion/checkpoints/committed_update_480.pt"),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def freeze_panel(output: Path) -> list[dict[str, Any]]:
    rows = []
    for update, stage, h1, h32, path in PANEL:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["completed_optimizer_step"]) != update or float(payload["H1"]) != h1 or float(payload["H32"]) != h32:
            raise RuntimeError(f"checkpoint_metadata_mismatch:{path}")
        rows.append({"checkpoint_id": f"update_{update}", "update": update, "historical_stage": stage,
                     "H1": h1, "H32": h32, "checkpoint": str(path.resolve()),
                     "selection_reason": "predeclared chronological H1 coverage; selected before D36 outcomes"})
    write_json(output / "D36_CHECKPOINT_PANEL.json", {"frozen_before_outcomes": True, "checkpoints": rows})
    return rows


def matched_indices(family_ids: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic family-balanced sampling without replacement."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index, family in enumerate(family_ids):
        groups.setdefault(str(family), []).append(index)
    order = sorted(groups)
    selected: list[int] = []
    cursor = 0
    shuffled = {key: list(rng.permutation(values)) for key, values in groups.items()}
    while len(selected) < min(count, len(family_ids)):
        key = order[cursor % len(order)]
        if shuffled[key]:
            selected.append(int(shuffled[key].pop()))
        cursor += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def perturb(batch: dict[str, torch.Tensor], channels: Mapping[str, Mapping[str, Any]], sigma: float, seed: int) -> dict[str, torch.Tensor]:
    result = {key: value.clone() for key, value in batch.items()}
    if sigma == 0.0:
        return result
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(result["history_positions"].shape, generator=generator) * float(sigma)
    result["history_positions"] += noise
    for joint in range(6):
        scale = float(channels[f"planned_joint_position_{joint + 1}"]["scale"])
        result["inputs"][:, :, joint] += noise[:, :, joint] / scale
    return result


def infer(model: Any, batch: Mapping[str, torch.Tensor], channels: Mapping[str, Any]) -> torch.Tensor:
    with torch.inference_mode():
        return d6.causal_paired_rollout(
            model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"],
            batch["starts"], batch["ends"], channels, horizon=32,
            target_positions_for_teacher=None, hidden_consistency_enabled=False,
        )["predictions"].cpu()


def bootstrap_delta(left: np.ndarray, right: np.ndarray, seed: int, draws: int = 2000) -> dict[str, float]:
    delta = left - right
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        estimates[i] = float(np.mean(delta[rng.integers(0, len(delta), len(delta))]))
    return {"mean_delta": float(np.mean(delta)), "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975))}


def run(output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    panel = freeze_panel(output)
    runtime = d17.preflight_runtime()
    pre = runtime["pre"]
    channels = pre["stats"]["channels"]
    arrays = {role: build_sequence_arrays(pre["dataset"], pre["stats"], role, max_rollout_horizon=32) for role in ROLES}
    indices = {role: matched_indices(arrays[role].family_ids, SAMPLE_COUNT, SEED + i) for i, role in enumerate(ROLES)}
    ledgers: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors: dict[tuple[int, str, float, int], np.ndarray] = {}
    for checkpoint in panel:
        model = d6.load_model(Path(checkpoint["checkpoint"]), trainable=False)
        for role_index, role in enumerate(ROLES):
            raw = d6.tensor_batch(arrays[role], indices[role])
            for noise_index, sigma in enumerate(NOISE_RAD):
                batch = perturb(raw, channels, sigma, SEED + role_index * 100 + noise_index)
                prediction = infer(model, batch, channels)
                target = batch["target_positions"].cpu()
                for horizon in HORIZONS:
                    per_episode = torch.sqrt(torch.mean((prediction[:, horizon - 1, :] - target[:, horizon - 1, :]) ** 2, dim=1)).numpy()
                    errors[(checkpoint["update"], role, sigma, horizon)] = per_episode
                    summaries.append({**checkpoint, "task": "joint_trajectory_prediction", "split": role,
                                      "condition": "nominal" if sigma == 0 else "joint_position_sensor_noise",
                                      "perturbation_sigma_rad": sigma, "horizon": horizon, "episodes": len(per_episode),
                                      "joint_position_rmse_rad": float(np.sqrt(np.mean(per_episode ** 2))),
                                      "episode_mean_rmse_rad": float(np.mean(per_episode)),
                                      "episode_p95_rmse_rad": float(np.quantile(per_episode, 0.95)),
                                      "success_rate_at_0p05_rad": float(np.mean(per_episode <= 0.05)),
                                      "collision_events": None, "clearance_m": None})
                    for local, value in enumerate(per_episode):
                        ledgers.append({"checkpoint": checkpoint["checkpoint_id"], "H1": checkpoint["H1"], "H32": checkpoint["H32"],
                                        "task": "joint_trajectory_prediction", "seed": SEED, "episode_id": str(arrays[role].window_ids[indices[role][local]]),
                                        "family_id": str(arrays[role].family_ids[indices[role][local]]), "split": role,
                                        "condition": "nominal" if sigma == 0 else "joint_position_sensor_noise",
                                        "perturbation_sigma_rad": sigma, "horizon": horizon, "episode_result": "success" if value <= 0.05 else "precision_failure",
                                        "joint_position_rmse_rad": float(value), "safety_events": None, "failure_classification": None if value <= 0.05 else "threshold_exceeded"})
    comparisons = []
    champion = panel[-1]
    for other in panel[:-1]:
        for role in ROLES:
            for sigma in NOISE_RAD:
                for horizon in HORIZONS:
                    stats = bootstrap_delta(errors[(champion["update"], role, sigma, horizon)], errors[(other["update"], role, sigma, horizon)], SEED + other["update"] + horizon)
                    comparisons.append({"checkpoint_a": champion["checkpoint_id"], "checkpoint_b": other["checkpoint_id"],
                                        "split": role, "perturbation_sigma_rad": sigma, "horizon": horizon,
                                        "metric": "episode_joint_position_rmse_rad", **stats,
                                        "interpretation": "negative_favors_a"})
    write_csv(output / "D36_CAPABILITY_RESULTS.csv", summaries)
    write_csv(output / "D36_RESULT_LEDGER.csv", ledgers)
    write_csv(output / "D36_PAIRED_COMPARISONS.csv", comparisons)
    summary = {
        "schema_version": "d36_summary_v1",
        "status": "PASS",
        "evaluation_scope": "frozen Stage 3 checkpoints; validation/test/generalization splits; nominal and sensor-noise conditions",
        "horizons": list(HORIZONS),
        "noise_sigma_rad": list(NOISE_RAD),
        "checkpoints": len(panel),
        "episodes": len(ledgers),
        "scenario_episodes": len(ledgers) // len(HORIZONS),
        "summaries": len(summaries),
        "paired_comparisons": len(comparisons),
        "output": str(output.resolve()),
    }
    write_json(output / "D36_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
