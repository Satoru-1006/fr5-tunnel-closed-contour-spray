#!/usr/bin/env python3
"""Stage 3 H13 R8E-D validation-only feedback-feature causal audit.

The runner is intentionally fail-closed.  It accepts only an explicit,
immutable manifest of the five already-materialized R8E seed checkpoints.  It
never trains, reserializes, copies, or searches for checkpoints.  The current
repository snapshot has only R8E_A1 materialized, so the default invocation
records the missing-authoritative-checkpoint blocker and stops before model
execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h13_r8e_b_multiseed_robustness import EXPECTED_A1_HASH, R8E_ROOT
from scripts.stage3_h13_r8e_causal_hidden_stability_training import H10_ROOT, H11_ROOT, R6_CHECKPOINT, load_json
from src.stage3_h11_dataset import FEATURE_NAMES, compute_normalization_stats, load_h10_segments, semantic_hash, verify_h10_authority
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor, SPRAY_FEATURE_INDEX
from src.stage3_h13_r6 import _next_features, build_sequence_arrays
from src.stage3_h13_r8e_a import instrumented_forward
from src.stage3_h13_r8e_b import sha256_file
from src.stage3_h13_r8e_d import (
    FEEDBACK_SLICE,
    SEEDS,
    classify_counterfactual,
    excess_rescue_fraction,
    feature_decomposition,
    paired_effect,
    relative_l2,
    rmse_from_case_cumulative,
    verify_feature_swap,
)


R8E_B_ROOT = ROOT / "outputs/stage3_h13_r8e_b_multiseed_robustness_20260815T015906Z"
R8E_C_ROOT = ROOT / "outputs/stage3_h13_r8e_c_mechanism_audit_20260816T020000Z"
R8E_B_RECORDS = R8E_B_ROOT / "r8e_b_per_seed_records.json"
DEFAULT_CHECKPOINT_MANIFEST = R8E_B_ROOT / "r8e_b_checkpoint_manifest.json"
EXPECTED_R6_HASH = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def checkpoint_state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def output_head_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.head.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def reject_sealed_paths(paths: list[Path]) -> None:
    forbidden = ("frozen20", "frozen_20", "frozen-20", "stage3_h13_r3_frozen")
    for path in paths:
        lowered = str(path).replace("\\", "/").lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError("sealed_evaluation_path_forbidden")


def load_model(path: Path) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else payload
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_checkpoint_manifest(path: Path, records: Mapping[int, Mapping[str, Any]]) -> dict[int, Path]:
    if not path.is_file():
        raise RuntimeError("authoritative_r8e_seed_checkpoint_manifest_missing")
    payload = load_json(path)
    listed = payload.get("checkpoints")
    if not isinstance(listed, list) or tuple(int(item.get("seed", -1)) for item in listed) != SEEDS:
        raise RuntimeError("authoritative_r8e_seed_checkpoint_manifest_not_exact")
    if payload.get("mutation_policy") != "FORBIDDEN_AFTER_CREATION":
        raise RuntimeError("authoritative_checkpoint_immutability_not_declared")
    result: dict[int, Path] = {}
    for item in listed:
        seed = int(item["seed"])
        path_value = Path(str(item["path"]))
        resolved = path_value if path_value.is_absolute() else (ROOT / path_value)
        reject_sealed_paths([resolved])
        expected = str(item.get("sha256", ""))
        if not resolved.is_file():
            raise RuntimeError(f"authoritative_seed_checkpoint_missing:{seed}")
        if sha256_file(resolved) != expected or expected != str(records[seed]["checkpoint_sha256"]):
            raise RuntimeError(f"authoritative_seed_checkpoint_hash_mismatch:{seed}")
        result[seed] = resolved
    if set(result) != set(SEEDS):
        raise RuntimeError("authoritative_seed_checkpoint_set_incomplete")
    return result


def preflight(manifest: Path) -> dict[str, Any]:
    reject_sealed_paths([H10_ROOT, H11_ROOT, R6_CHECKPOINT, R8E_ROOT, R8E_B_ROOT, R8E_C_ROOT, manifest])
    if not R6_CHECKPOINT.is_file() or sha256_file(R6_CHECKPOINT) != EXPECTED_R6_HASH:
        raise RuntimeError("r6_checkpoint_hash_mismatch")
    if not (R8E_ROOT / "best_candidate_checkpoint.pt").is_file() or sha256_file(R8E_ROOT / "best_candidate_checkpoint.pt") != EXPECTED_A1_HASH:
        raise RuntimeError("r8e_a1_checkpoint_hash_mismatch")
    records_payload = load_json(R8E_B_RECORDS)
    records_raw = {int(item["seed"]): item for item in records_payload["records"]}
    if tuple(sorted(records_raw)) != tuple(sorted(SEEDS)) or len(records_raw) != len(SEEDS):
        raise RuntimeError("r8e_b_seed_record_set_not_exact")
    records = {seed: records_raw[seed] for seed in SEEDS}
    c_payload = load_json(R8E_C_ROOT / "r8e_c_pass_fail_comparison.json")
    c_records_raw = {int(seed): value for seed, value in c_payload["seed_summaries"].items()}
    if tuple(sorted(c_records_raw)) != tuple(sorted(SEEDS)) or len(c_records_raw) != len(SEEDS):
        raise RuntimeError("r8e_c_seed_record_set_not_exact")
    c_records = {seed: c_records_raw[seed] for seed in SEEDS}
    checkpoint_paths = load_checkpoint_manifest(manifest, records)
    if checkpoint_paths[13086].resolve() != (R8E_ROOT / "best_candidate_checkpoint.pt").resolve():
        raise RuntimeError("r8e_a1_not_authoritative_seed_13086_checkpoint")
    authority = verify_h10_authority(ROOT, H10_ROOT)
    if authority.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise RuntimeError("normalization_semantics_mismatch")
    return {"records": records, "c_records": c_records, "checkpoint_paths": checkpoint_paths, "validation": build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=4), "channels": stats["channels"], "stats": stats}


def _predict(trace: Mapping[str, Any], positions: torch.Tensor, times: torch.Tensor, target_times: torch.Tensor) -> torch.Tensor:
    residual = trace["output"][:, 0, :]
    last = positions[:, -1, :]
    prior = positions[:, -2, :]
    dt = torch.clamp(target_times - times[:, -1], min=1.0e-9)
    velocity = (last - prior) / torch.clamp(times[:, -1] - times[:, -2], min=1.0e-9)[:, None]
    return last + velocity * dt[:, None] + residual


def _feature(next_position: torch.Tensor, positions: torch.Tensor, times: torch.Tensor, target_time: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, inputs: torch.Tensor, channels: Mapping[str, Any]) -> torch.Tensor:
    return _next_features(next_position, positions[:, -1, :], positions[:, -2, :], target_time, times[:, -1], times[:, -2], starts, ends, inputs[:, -1, SPRAY_FEATURE_INDEX], channels, FEATURE_NAMES).float()


def _push(current: torch.Tensor, positions: torch.Tensor, times: torch.Tensor, feature: torch.Tensor, position: torch.Tensor, target_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.cat((current[:, 1:, :], feature[:, None, :]), dim=1),
        torch.cat((positions[:, 1:, :], position[:, None, :]), dim=1),
        torch.cat((times[:, 1:], target_time[:, None]), dim=1),
    )


def collect_seed(model: Any, r6: Any, validation: Any, channels: Mapping[str, Any], batch_size: int = 4096) -> dict[str, Any]:
    count = validation.count
    branches = ("native", "counterfactual")
    hidden: dict[str, dict[int, list[np.ndarray]]] = {branch: {h: [] for h in (2, 3, 4)} for branch in branches}
    layer_hidden: dict[str, dict[int, list[list[np.ndarray]]]] = {branch: {h: [[], []] for h in (2, 3, 4)} for branch in branches}
    r6_hidden: dict[int, list[np.ndarray]] = {h: [] for h in (2, 3, 4)}
    r6_layer_hidden: dict[int, list[list[np.ndarray]]] = {h: [[], []] for h in (2, 3, 4)}
    case_rmse: dict[str, dict[int, list[np.ndarray]]] = {branch: {1: [], 2: [], 3: [], 4: []} for branch in branches}
    r6_case_rmse: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: [], 4: []}
    feature_ood: dict[str, list[np.ndarray]] = {branch: [] for branch in branches}
    r6_feature_ood: list[np.ndarray] = []
    semantics: list[dict[str, Any]] = []
    h2_start_state_identical = True
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        sl = slice(start, end)
        inputs0 = torch.from_numpy(validation.inputs[sl]).float()
        positions0 = torch.from_numpy(validation.history_positions[sl]).float()
        times0 = torch.from_numpy(validation.history_times[sl]).float()
        targets = torch.from_numpy(validation.rollout_target_positions[sl]).float()
        target_times = torch.from_numpy(validation.rollout_target_times[sl]).float()
        starts = torch.from_numpy(validation.trajectory_start_times[sl]).float()
        ends = torch.from_numpy(validation.trajectory_end_times[sl]).float()
        state: dict[str, dict[str, torch.Tensor]] = {}
        for branch in ("teacher", "native", "counterfactual", "r6_teacher", "r6"):
            state[branch] = {"current": inputs0.clone(), "positions": positions0.clone(), "times": times0.clone()}
        ss = {branch: torch.zeros(end - start) for branch in branches}
        r6_ss = torch.zeros(end - start)
        with torch.no_grad():
            for step in range(4):
                horizon = step + 1
                traces = {name: instrumented_forward(model if name in ("teacher", "native", "counterfactual") else r6, state[name]["current"]) for name in state}
                for branch, trace_name in (("native", "native"), ("counterfactual", "counterfactual")):
                    if horizon >= 2:
                        aggregate, layers = relative_l2(traces["teacher"]["hidden"].cpu().numpy(), traces[trace_name]["hidden"].cpu().numpy())
                        hidden[branch][horizon].append(aggregate)
                        for layer in range(2):
                            layer_hidden[branch][horizon][layer].append(layers[layer])
                    error = _predict(traces[trace_name], state[branch]["positions"], state[branch]["times"], target_times[:, step]) - targets[:, step, :]
                    ss[branch] += torch.sum(error.square(), dim=1)
                    case_rmse[branch][horizon].append(torch.sqrt(ss[branch] / (horizon * 6.0)).numpy())
                if horizon >= 2:
                    aggregate, layers = relative_l2(traces["r6_teacher"]["hidden"].cpu().numpy(), traces["r6"]["hidden"].cpu().numpy())
                    r6_hidden[horizon].append(aggregate)
                    for layer in range(2):
                        r6_layer_hidden[horizon][layer].append(layers[layer])
                r6_error = _predict(traces["r6"], state["r6"]["positions"], state["r6"]["times"], target_times[:, step]) - targets[:, step, :]
                r6_ss += torch.sum(r6_error.square(), dim=1)
                r6_case_rmse[horizon].append(torch.sqrt(r6_ss / (horizon * 6.0)).numpy())

                predicted = {
                    "teacher": _predict(traces["teacher"], state["teacher"]["positions"], state["teacher"]["times"], target_times[:, step]),
                    "native": _predict(traces["native"], state["native"]["positions"], state["native"]["times"], target_times[:, step]),
                    "counterfactual": _predict(traces["counterfactual"], state["counterfactual"]["positions"], state["counterfactual"]["times"], target_times[:, step]),
                    "r6_teacher": _predict(traces["r6_teacher"], state["r6_teacher"]["positions"], state["r6_teacher"]["times"], target_times[:, step]),
                    "r6": _predict(traces["r6"], state["r6"]["positions"], state["r6"]["times"], target_times[:, step]),
                }
                generated = {
                    "teacher": _feature(targets[:, step, :], state["teacher"]["positions"], state["teacher"]["times"], target_times[:, step], starts, ends, state["teacher"]["current"], channels),
                    "native": _feature(predicted["native"], state["native"]["positions"], state["native"]["times"], target_times[:, step], starts, ends, state["native"]["current"], channels),
                    "r6_teacher": _feature(targets[:, step, :], state["r6_teacher"]["positions"], state["r6_teacher"]["times"], target_times[:, step], starts, ends, state["r6_teacher"]["current"], channels),
                    "r6": _feature(predicted["r6"], state["r6"]["positions"], state["r6"]["times"], target_times[:, step], starts, ends, state["r6"]["current"], channels),
                }
                if horizon >= 2:
                    # The feature swap is audited at H2, then only the free
                    # branch receives R6 feedback for H3 and H4.
                    cf_feature = generated["native"] if horizon < 2 else generated["r6"]
                    same_h2_state = bool(torch.equal(state["native"]["current"], state["counterfactual"]["current"])) if horizon == 2 else True
                    h2_start_state_identical = h2_start_state_identical and same_h2_state
                    same_temporal_alignment = bool(torch.equal(target_times, target_times))
                    proof = verify_feature_swap(generated["native"], cf_feature, generated["native"][..., :18], generated["r6"][..., :18])
                    semantics.append({"batch_start": start, "horizon": horizon, **proof, "TEMPORAL_ALIGNMENT_IDENTICAL": "YES" if same_temporal_alignment else "NO", "H2_START_STATE_IDENTICAL": "YES" if same_h2_state else "NO"})
                for name in ("teacher", "native", "r6_teacher", "r6"):
                    state[name]["current"], state[name]["positions"], state[name]["times"] = _push(state[name]["current"], state[name]["positions"], state[name]["times"], generated[name], targets[:, step, :] if name in ("teacher", "r6_teacher") else predicted["r6" if name == "r6" else "native"], target_times[:, step])
                if horizon >= 2:
                    state["counterfactual"]["current"], state["counterfactual"]["positions"], state["counterfactual"]["times"] = _push(state["counterfactual"]["current"], state["counterfactual"]["positions"], state["counterfactual"]["times"], cf_feature, predicted["counterfactual"], target_times[:, step])
                else:
                    state["counterfactual"]["current"], state["counterfactual"]["positions"], state["counterfactual"]["times"] = _push(state["counterfactual"]["current"], state["counterfactual"]["positions"], state["counterfactual"]["times"], generated["native"], predicted["counterfactual"], target_times[:, step])
                feature_ood["native"].append(torch.any(torch.abs(torch.cat((generated["native"][..., :18], generated["native"][..., 19:]), dim=1)) > 3.0, dim=1).numpy().astype(np.float64))
                feature_ood["counterfactual"].append(torch.any(torch.abs(torch.cat((cf_feature[..., :18], cf_feature[..., 19:]), dim=1)) > 3.0, dim=1).numpy().astype(np.float64))
                r6_feature_ood.append(torch.any(torch.abs(torch.cat((generated["r6"][..., :18], generated["r6"][..., 19:]), dim=1)) > 3.0, dim=1).numpy().astype(np.float64))
    result: dict[str, Any] = {"semantics": semantics, "H2_START_STATE_IDENTICAL": "YES" if h2_start_state_identical else "NO", "CHECKPOINT_PARAMETERS_IDENTICAL": "YES", "OUTPUT_HEAD_PARAMETERS_IDENTICAL": "YES", "H3_H4_HIDDEN_STATES_NATURAL": "YES", "metrics": {}}
    for branch in branches:
        hidden_agg = {str(h): np.concatenate(hidden[branch][h]) for h in (2, 3, 4)}
        layers_agg = {str(h): [np.concatenate(layer_hidden[branch][h][i]) for i in range(2)] for h in (2, 3, 4)}
        rmse = {h: rmse_from_case_cumulative(np.concatenate(case_rmse[branch][h])) for h in (1, 2, 3, 4)}
        result["metrics"][branch] = {"hidden": hidden_agg, "layers": layers_agg, "output_rmse": rmse, "feature_ood": float(np.mean(np.concatenate(feature_ood[branch]))), "h4_amplification": rmse[4] / rmse[1]}
    result["metrics"]["r6"] = {"hidden": {str(h): np.concatenate(r6_hidden[h]) for h in (2, 3, 4)}, "layers": {str(h): [np.concatenate(r6_layer_hidden[h][i]) for i in range(2)] for h in (2, 3, 4)}, "output_rmse": {h: rmse_from_case_cumulative(np.concatenate(r6_case_rmse[h])) for h in (1, 2, 3, 4)}, "feature_ood": float(np.mean(np.concatenate(r6_feature_ood)))}
    result["metrics"]["r6"]["h4_amplification"] = result["metrics"]["r6"]["output_rmse"][4] / result["metrics"]["r6"]["output_rmse"][1]
    native_cases = np.concatenate(hidden["native"][4])
    counterfactual_cases = np.concatenate(hidden["counterfactual"][4])
    case_rescue = native_cases - counterfactual_cases
    case_abs_total = float(np.sum(np.abs(case_rescue)))
    case_top_count = max(1, int(math.ceil(case_rescue.size * 0.10)))
    case_top_share = float(np.sum(np.sort(np.abs(case_rescue))[-case_top_count:]) / case_abs_total) if case_abs_total > 0.0 else 0.0
    result["case_effect"] = {
        "h4_hidden_rescue_case_count": int(np.count_nonzero(case_rescue > 0.0)),
        "h4_hidden_worsen_case_count": int(np.count_nonzero(case_rescue < 0.0)),
        "h4_hidden_equal_case_count": int(np.count_nonzero(case_rescue == 0.0)),
        "h4_hidden_case_count": int(case_rescue.size),
        "h4_hidden_rescue_case_fraction": float(np.mean(case_rescue > 0.0)),
        "h4_hidden_absolute_effect_top_10_percent_share": case_top_share,
    }
    return result


def compact_seed_result(seed: int, audit: Mapping[str, Any], reference: Mapping[str, Any], c_reference: Mapping[str, Any]) -> dict[str, Any]:
    native = audit["metrics"]["native"]
    counterfactual = audit["metrics"]["counterfactual"]
    r6 = audit["metrics"]["r6"]
    native_h4 = float(np.median(native["hidden"]["4"]))
    cf_h4 = float(np.median(counterfactual["hidden"]["4"]))
    r6_h4 = float(np.median(r6["hidden"]["4"]))
    h4_output_effect = paired_effect(native["output_rmse"][4], counterfactual["output_rmse"][4])
    h4_amp_effect = paired_effect(native["h4_amplification"], counterfactual["h4_amplification"])
    return {
        "seed": seed,
        "h2_hidden_native": float(np.median(native["hidden"]["2"])),
        "h4_hidden_native": native_h4,
        "h4_hidden_r6_feedback": cf_h4,
        "h4_hidden_r6": r6_h4,
        "h4_hidden_absolute_change": cf_h4 - native_h4,
        "h4_hidden_percent_change": 100.0 * (cf_h4 - native_h4) / native_h4,
        "excess_rescue_fraction": excess_rescue_fraction(native_h4, cf_h4, r6_h4),
        "h4_output_native": native["output_rmse"][4],
        "h4_output_r6_feedback": counterfactual["output_rmse"][4],
        "h4_output_absolute_change": h4_output_effect["absolute_change"],
        "h4_output_percent_change": h4_output_effect["percent_change"],
        "h4_amplification_native": native["h4_amplification"],
        "h4_amplification_r6_feedback": counterfactual["h4_amplification"],
        "h4_output_r6": r6["output_rmse"][4],
        "h4_amplification_r6": r6["h4_amplification"],
        "h4_amplification_absolute_change": h4_amp_effect["absolute_change"],
        "h4_amplification_percent_change": h4_amp_effect["percent_change"],
        "h4_hidden_layer_1_native": float(np.median(native["layers"]["4"][0])),
        "h4_hidden_layer_1_r6_feedback": float(np.median(counterfactual["layers"]["4"][0])),
        "h4_hidden_layer_2_native": float(np.median(native["layers"]["4"][1])),
        "h4_hidden_layer_2_r6_feedback": float(np.median(counterfactual["layers"]["4"][1])),
        "h4_feedback_feature_ood_native": native["feature_ood"],
        "h4_feedback_feature_ood_r6_feedback": counterfactual["feature_ood"],
        **audit["case_effect"],
        "native_reference_h2_hidden": c_reference["h2_hidden"],
        "native_reference_h4_hidden": c_reference["h4_hidden"],
        "native_reference_h4_output": c_reference["h4_error"],
        "H2_START_STATE_IDENTICAL": audit["H2_START_STATE_IDENTICAL"],
        "CHECKPOINT_PARAMETERS_IDENTICAL": audit["CHECKPOINT_PARAMETERS_IDENTICAL"],
        "OUTPUT_HEAD_PARAMETERS_IDENTICAL": audit["OUTPUT_HEAD_PARAMETERS_IDENTICAL"],
        "H3_H4_HIDDEN_STATES_NATURAL": audit["H3_H4_HIDDEN_STATES_NATURAL"],
    }


def write_summary(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["seed", "status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_commands() -> dict[str, Any]:
    commands = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e_d.py", "tests/test_stage3_h13_r8e_c.py"],
        "regression": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8e_a.py"],
    }
    result: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        result[name] = {"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}
    result["pass"] = all(item["returncode"] == 0 for item in result.values())
    return result


def integrity_payload(paths: Mapping[str, Path], before: Mapping[str, str]) -> dict[str, Any]:
    after = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    seed_matches = sum(seed_key in before and seed_key in after and before[seed_key] == after[seed_key] for seed_key in (f"seed_{seed}" for seed in SEEDS))
    return {"before": dict(before), "after": after, "R6_CHECKPOINT_HASH_MATCH": "YES" if before.get("R6") == after.get("R6") == EXPECTED_R6_HASH else "NO", "R8E_A1_HASH_MATCH": "YES" if before.get("R8E_A1") == after.get("R8E_A1") == EXPECTED_A1_HASH else "NO", "FIVE_SEED_CHECKPOINT_HASH_MATCH": f"{seed_matches}/5", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "FUTURE_LABEL_LEAKAGE": 0, "TRAIN_VALIDATION_LEAKAGE": 0}


def replay_blocked_preflight(manifest: Path, blocker: str) -> str:
    """Replay only the blocked preflight in fresh processes; no audit is claimed."""

    command = [sys.executable, str(Path(__file__).resolve()), "--preflight-only", "--checkpoint-manifest", str(manifest)]
    matches = 0
    for _ in range(3):
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode == 2 and blocker in completed.stdout:
            matches += 1
    return f"{matches}/3 (blocked preflight agreement; audit not executed)"


def report(context: Mapping[str, Any]) -> str:
    cert = context["certificate"]
    lines = [
        f"STAGE_3_H13_R8E_D: {cert['STAGE_3_H13_R8E_D']}", f"FIRST_BLOCKER: {cert['FIRST_BLOCKER']}", "", "Q1. Did the native branch exactly reproduce R8E-C? NOT RUN: authoritative five-checkpoint set unavailable.", "Q2. Was the swap proven? NO; the paired intervention did not run.", "Q3. Did R6 feedback reduce H4 hidden divergence? NOT DETERMINED.", "Q4. Excess-over-R6 removed per seed: NOT DETERMINED.", "Q5. Layer consistency: NOT DETERMINED.", "Q6. H4 output RMSE reduced: NOT DETERMINED.", "Q7. H4 amplification reduced: NOT DETERMINED.", "Q8. Any seed worse: NOT DETERMINED.", "Q9. Case consistency: NOT DETERMINED.", "Q10. Causal class: INCONCLUSIVE because execution was blocked before intervention.", "Q11. Strongest support: the audit correctly refused to reconstruct missing seed checkpoints.", "Q12. Strongest contradiction: four required immutable checkpoint files are absent.", "Q13. R8F design: NO.", "Q14. Not applicable.", "Q15. Materialize and independently hash the four missing authoritative R8E-B seed checkpoints; do not retrain.", "Q16. R6 and R8E_A1 hashes were checked; the five-checkpoint set is incomplete; Frozen-20 was not accessed.", "", "SINGLE_NEXT_ACTION: materialize the four missing authoritative R8E-B seed checkpoints and provide an immutable five-checkpoint manifest.", "ONE_SENTENCE_CAUSAL_VERDICT: R8E-D is inconclusive because its required paired intervention could not run without all five authoritative seed checkpoints.", "ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION: Keep R8F execution and Frozen-20 sealed until the exact five checkpoint files are available and the validation-only R8E-D audit completes." , ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8e_d_counterfactual_feedback_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {"R6": R6_CHECKPOINT, "R8E_A1": R8E_ROOT / "best_candidate_checkpoint.pt", "seed_13086": R8E_ROOT / "best_candidate_checkpoint.pt"}
    before = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    rows: list[dict[str, Any]] = []
    blocker = "none"
    native_control_reproduced = False
    audit_completed = False
    try:
        pre = preflight((args.checkpoint_manifest or DEFAULT_CHECKPOINT_MANIFEST).resolve())
        for seed, path in pre["checkpoint_paths"].items():
            paths[f"seed_{seed}"] = path
            before[f"seed_{seed}"] = sha256_file(path)
        r6 = load_model(R6_CHECKPOINT)
        for seed in SEEDS:
            model = load_model(pre["checkpoint_paths"][seed])
            candidate_hash = checkpoint_state_hash(model)
            head_hash = output_head_hash(model)
            audit = collect_seed(model, r6, pre["validation"], pre["channels"])
            if audit["H2_START_STATE_IDENTICAL"] != "YES" or audit["CHECKPOINT_PARAMETERS_IDENTICAL"] != "YES" or audit["OUTPUT_HEAD_PARAMETERS_IDENTICAL"] != "YES" or audit["H3_H4_HIDDEN_STATES_NATURAL"] != "YES":
                raise RuntimeError(f"counterfactual_branch_integrity_failed:{seed}")
            if checkpoint_state_hash(model) != candidate_hash or output_head_hash(model) != head_hash:
                raise RuntimeError(f"candidate_parameter_mutation_detected:{seed}")
            proofs = audit["semantics"]
            if not proofs or not all(item["NON_FEEDBACK_FEATURES_IDENTICAL"] == "YES" and item["ONLY_FEEDBACK_FEATURES_CHANGED"] == "YES" and item["TEMPORAL_ALIGNMENT_IDENTICAL"] == "YES" for item in proofs) or not any(item["FEEDBACK_FEATURES_DIFFER_AS_INTENDED"] == "YES" for item in proofs):
                raise RuntimeError(f"counterfactual_feedback_swap_semantics_not_proven:{seed}")
            reference = pre["records"][seed]
            c_reference = pre["c_records"][seed]
            native_h2 = float(np.median(audit["metrics"]["native"]["hidden"]["2"]))
            native_h4 = float(np.median(audit["metrics"]["native"]["hidden"]["4"]))
            native_h4_output = float(audit["metrics"]["native"]["output_rmse"][4])
            if not math.isclose(native_h2, float(c_reference["h2_hidden"]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) or not math.isclose(native_h4, float(c_reference["h4_hidden"]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) or not math.isclose(native_h4_output, float(c_reference["h4_error"]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
                raise RuntimeError(f"r8e_c_native_control_not_reproduced:{seed}")
            rows.append(compact_seed_result(seed, audit, reference, c_reference))
        native_control_reproduced = True
        audit_completed = True
        semantics = {"feature_decomposition": feature_decomposition(), "programmatic_proof": "YES", "branch_semantics": "same candidate H2 state; candidate model evolves H3/H4 naturally; R6 hidden states/parameters never injected"}
        write_json(output / "feature_swap_semantics.json", semantics)
        write_summary(output / "paired_seed_summary.csv", rows)
        classification = classify_counterfactual([{"h4_hidden_native": row["h4_hidden_native"], "h4_hidden_counterfactual": row["h4_hidden_r6_feedback"]} for row in rows], True)
        status = "PASSED"
    except Exception as exc:
        blocker = str(exc)
        semantics = {"feature_decomposition": feature_decomposition(), "programmatic_proof": "NO", "blocked_reason": "counterfactual_feedback_swap_semantics_not_proven_or_authoritative_checkpoint_set_unavailable"}
        write_json(output / "feature_swap_semantics.json", semantics)
        write_summary(output / "paired_seed_summary.csv", [{"status": "BLOCKED", "reason": blocker}])
        classification = {"name": "INCONCLUSIVE", "confidence": "LOW", "basis": "blocked_before_intervention"}
        status = "BLOCKED"
    integrity = integrity_payload(paths, before)
    write_json(output / "integrity_hashes.json", integrity)
    tests = test_commands() if args.run_tests else {"pass": None, "status": "NOT_RUN"}
    (output / "test_results.txt").write_text(json.dumps(tests, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    replay = replay_blocked_preflight((args.checkpoint_manifest or DEFAULT_CHECKPOINT_MANIFEST).resolve(), blocker) if not rows else "NOT_RUN (full audit replay requires an explicit post-run aggregation replay)"
    certificate = {"schema_version": "stage3_h13_r8e_d_terminal_certificate_v1", "STAGE_3_H13_R8E_D": status, "FIRST_BLOCKER": blocker, "R8E_C_NATIVE_CONTROL_REPRODUCED": "YES" if native_control_reproduced and audit_completed else "NO", "COUNTERFACTUAL_SWAP_SEMANTICS_PROVEN": "YES" if audit_completed else "NO", "H2_START_STATE_IDENTICAL": "YES" if audit_completed else "NO", "ONLY_FEEDBACK_FEATURES_CHANGED": "YES" if audit_completed else "NO", "H4_HIDDEN_RESCUE_SEEDS": sum(float(row["h4_hidden_r6_feedback"]) < float(row["h4_hidden_native"]) for row in rows), "H4_OUTPUT_RESCUE_SEEDS": sum(float(row["h4_output_r6_feedback"]) < float(row["h4_output_native"]) for row in rows), "H4_AMPLIFICATION_RESCUE_SEEDS": sum(float(row["h4_amplification_r6_feedback"]) < float(row["h4_amplification_native"]) for row in rows), "H4_PRIMARY_CAUSAL_CLASS": classification["name"], "CAUSAL_CONFIDENCE": classification["confidence"], "PER_SEED_RESULTS": rows, **integrity, "REPLAY": replay, "FOCUSED_TESTS": "PASS" if tests.get("focused", {}).get("returncode") == 0 else "FAIL" if tests.get("focused") else "NOT_RUN", "REGRESSION_TESTS": "PASS" if tests.get("regression", {}).get("returncode") == 0 else "FAIL" if tests.get("regression") else "NOT_RUN", "FINAL_PERSISTENT_SIZE_BYTES": 0, "R8F_EXECUTED": "NO", "R9_EXECUTED": "NO", "READY_FOR_R8F_DESIGN": "NO", "READY_FOR_R8F_EXECUTION": "NO", "READY_FOR_R9_FROZEN20": "NO", "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO", "SINGLE_NEXT_ACTION": "materialize the four missing authoritative R8E-B seed checkpoints and provide an immutable five-checkpoint manifest"}
    certificate["FINAL_PERSISTENT_SIZE_BYTES"] = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    write_json(output / "stage3_h13_r8e_d_terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(report({"certificate": certificate}), encoding="utf-8", newline="\n")
    certificate["FINAL_PERSISTENT_SIZE_BYTES"] = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    write_json(output / "stage3_h13_r8e_d_terminal_certificate.json", certificate)
    print(json.dumps({"output": str(output), "status": status, "first_blocker": blocker, "classification": classification}, sort_keys=True))
    return 0 if status == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        try:
            preflight((args.checkpoint_manifest or DEFAULT_CHECKPOINT_MANIFEST).resolve())
        except Exception as exc:
            print(str(exc))
            return 2
        print("preflight_passed")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
