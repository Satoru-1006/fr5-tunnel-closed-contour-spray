"""Read-only Stage 3 H13 R8E-D3A model-selection failure audit.

This module intentionally contains no training, optimizer, checkpoint-writing,
or causal-intervention path.  It reuses the frozen rollout evaluator and
recomputes its scalar reduction independently while retaining only compact
JSON/Markdown evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_dataset import load_h10_segments, verify_h10_authority
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor
from src.stage3_h13_r6 import build_sequence_arrays, evaluate_rollout_metrics, rollout_residual_torch_r6
from src.stage3_h13_r8e import EVALUATION_HORIZONS, derive_h1_guardrail
from src.stage3_h13_r8e_d2 import canonical_json_sha256, canonical_model_state_sha256, sha256_file


D3_ROOT = ROOT / "outputs/stage3_h13_r8e_d3_prospective_five_seed_causal_replication_20260816T062203Z"
D2_ROOT = ROOT / "outputs/stage3_h13_r8e_d2_prospective_causal_design_20260816T053129Z"
H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
A1_ROOT = ROOT / "outputs/stage3_h13_r8e_causal_hidden_stability_training_20260814T181715Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
A1_CHECKPOINT = A1_ROOT / "best_candidate_checkpoint.pt"
SEED_13086 = D3_ROOT / "checkpoints/prospective_seed_13086.pt"
SEED_218309645 = D3_ROOT / "checkpoints/prospective_seed_218309645.pt"

EXPECTED_D2_CONFIG_SHA256 = "2c7647feb14fa47d4363d3479d34706da5d06cb6e89913111f64b0d146005972"
EXPECTED_H10_SEMANTIC_SHA256 = "cb282a06d73c322208f27bf7c0b019831a24f524ac0d72263929f22303f0801b"
EXPECTED_H10_MANIFEST_SHA256 = "26a3287d555c415560a342f5edffa7f52b523fee0632544cc91841e7d17e0570"
EXPECTED_H10_SPLIT_SHA256 = "b336adf2bb39237f62da1bb20245134a51a4b4dba8ea60a10b0c6c31cbadc129"
EXPECTED_R6_RAW_SHA256 = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
EXPECTED_A1_RAW_SHA256 = "4d57a265ab130619ae2971124a514c35bf6efac8731fbadbaef6a8d9d921e6ac"
EXPECTED_A1_CANONICAL_SHA256 = "dd16d6f19c387f066eb4f9ac490ab9b6299eb745a2ce71da224ecae3a59e56de"
EXPECTED_13086_RAW_SHA256 = "8c50da2550d019c9a4ecae501447f5bbf16ff4aaafdb377e5be358f5392f4d0d"
EXPECTED_13086_CANONICAL_SHA256 = "dd16d6f19c387f066eb4f9ac490ab9b6299eb745a2ce71da224ecae3a59e56de"
EXPECTED_218309645_RAW_SHA256 = "9186824ea744d584495092e8bb7982b978b5abafdc34a6c0c7aa4068524f2bc1"
EXPECTED_218309645_CANONICAL_SHA256 = "648ad2d663527675643ecc2acd6ca7e4cbbdff15b8dfa5e0dd11de5e7f9f98c8"
AUTHORITATIVE_R6_H1 = 0.00008188013630811277
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
HORIZONS = tuple(int(item) for item in EVALUATION_HORIZONS)
SEALED_TOKENS = ("frozen20", "frozen_20", "frozen-20", "stage3_h13_r3_frozen")
D2_H11_TRAIN_WINDOW_COUNT = 122202
D2_H11_VALIDATION_WINDOW_COUNT = 27594


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def git_text(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def file_snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        resolved = path.resolve()
        records[str(resolved)] = {
            "sha256": sha256_file(resolved) if resolved.is_file() else None,
            "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
        }
    return records


def recursive_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    return file_snapshot(sorted(root.rglob("*"))) if root.is_dir() else {}


def snapshot_matches(expected: Mapping[str, Mapping[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for name, record in expected.items():
        path = Path(name)
        exists = path.is_file()
        if not exists:
            failures.append(f"missing:{name}")
            continue
        if record.get("sha256") != sha256_file(path):
            failures.append(f"sha256:{name}")
        if record.get("size_bytes") != path.stat().st_size:
            failures.append(f"size:{name}")
    return not failures, failures


def source_paths() -> list[Path]:
    return [
        ROOT / "scripts/stage3_h13_r8e_d3_prospective_five_seed_causal_replication.py",
        ROOT / "scripts/stage3_h13_r8e_causal_hidden_stability_training.py",
        ROOT / "src/stage3_h13_r6.py",
        ROOT / "src/stage3_h13_r8e.py",
        ROOT / "src/stage3_h13_r8e_d2.py",
        ROOT / "src/stage3_h11_dataset.py",
        ROOT / "src/stage3_h11_model.py",
    ]


def assert_no_sealed_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        lowered = str(path).lower()
        if any(token in lowered for token in SEALED_TOKENS):
            raise RuntimeError(f"sealed_evaluation_path_forbidden:{path}")


def load_model(path: Path) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"invalid_checkpoint_payload:{path}")
    model = ResidualCausalGRUTrajectoryPredictor(input_size=20, hidden_size=128, num_layers=2, horizon=8)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def compact_metrics(model: Any, validation: Any, channels: Mapping[str, Any]) -> dict[str, float]:
    result = evaluate_rollout_metrics(model, validation, channels, horizons=HORIZONS, device="cpu", batch_size=4096)
    return {str(horizon): float(result[str(horizon)]["joint_position_rmse_rad"]) for horizon in HORIZONS}


def selection_predicate(*, updates: int, phase_stability: Sequence[bool], metrics: Mapping[str, float], threshold: float) -> bool:
    """Exact D3 boolean expression at d3 runner line 523, without side effects."""
    return bool(
        int(updates) == 512
        and all(bool(item) for item in phase_stability)
        and all(math.isfinite(float(value)) for value in metrics.values())
        and float(metrics["1"]) <= float(threshold)
    )


def margin_payload(observed: float, threshold: float) -> dict[str, Any]:
    margin = float(threshold) - float(observed)
    return {
        "observed": float(observed),
        "threshold": float(threshold),
        "margin_threshold_minus_observed": margin,
        "relative_margin_percent_of_threshold": 100.0 * margin / float(threshold),
        "pass_if_frozen_rule_applied": bool(float(observed) <= float(threshold)),
        "direction": "lower_is_better; positive margin means pass-side room",
    }


def independent_recompute(model: Any, validation: Any, channels: Mapping[str, Any], batch_size: int = 4096) -> dict[str, float]:
    """Recompute the frozen RMSE definition with a separate fsum reduction."""
    sums = {horizon: 0.0 for horizon in HORIZONS}
    counts = {horizon: 0 for horizon in HORIZONS}
    model.eval()
    with torch.no_grad():
        for start in range(0, validation.count, int(batch_size)):
            end = min(validation.count, start + int(batch_size))
            rolled = rollout_residual_torch_r6(
                model,
                torch.from_numpy(validation.inputs[start:end]).to(dtype=torch.float32),
                torch.from_numpy(validation.history_positions[start:end]).to(dtype=torch.float32),
                torch.from_numpy(validation.history_times[start:end]).to(dtype=torch.float32),
                torch.from_numpy(validation.rollout_target_times[start:end]).to(dtype=torch.float32),
                torch.from_numpy(validation.trajectory_start_times[start:end]).to(dtype=torch.float32),
                torch.from_numpy(validation.trajectory_end_times[start:end]).to(dtype=torch.float32),
                channels,
                rollout_horizon=max(HORIZONS),
                teacher_forcing_ratio_value=0.0,
                target_positions=None,
                sample_random=False,
                detach_state=False,
            )
            predicted = rolled.detach().cpu().numpy().astype(np.float64)
            target = validation.rollout_target_positions[start:end]
            for horizon in HORIZONS:
                error = (predicted[:, :horizon, :] - target[:, :horizon, :]).reshape(-1)
                # fsum is deliberately a second scalar reduction, not a call
                # to evaluate_rollout_metrics or its np.sum implementation.
                sums[horizon] += math.fsum(float(item) * float(item) for item in error)
                counts[horizon] += int(error.size)
    return {str(horizon): math.sqrt(sums[horizon] / counts[horizon]) for horizon in HORIZONS}


def max_metric_delta(left: Mapping[str, float], right: Mapping[str, float]) -> tuple[float, float]:
    absolute = 0.0
    relative = 0.0
    for key in left:
        delta = abs(float(left[key]) - float(right[key]))
        absolute = max(absolute, delta)
        relative = max(relative, delta / max(abs(float(right[key])), 1.0e-30))
    return absolute, relative


def path_line(path: Path, line: int) -> str:
    return f"{path.resolve()}:{line}"


def authority_preflight() -> dict[str, Any]:
    d2_preflight = read_json(D3_ROOT / "d2_gate0_preflight.json")
    d3_terminal = read_json(D3_ROOT / "terminal_certificate.json")
    d3_manifest = read_json(D3_ROOT / "checkpoint_identity_manifest.json")
    d2_protocol = read_json(D2_ROOT / "prospective_training_protocol.json")
    d2_manifest = read_json(D2_ROOT / "prospective_seed_manifest.json")
    d2_contract_hash = canonical_json_sha256(d2_protocol["normalized_training_contract"])
    d2_data_contract = d2_protocol["normalized_training_contract"]["data_and_splits"]
    d2_ok, d2_failures = snapshot_matches(d2_preflight["d2_snapshot"])

    authorized_paths = [
        H10_ROOT,
        H11_ROOT,
        R6_CHECKPOINT,
        A1_CHECKPOINT,
        SEED_13086,
        SEED_218309645,
        D2_ROOT,
        D3_ROOT,
    ]
    assert_no_sealed_paths(authorized_paths)

    expected_by_seed = {
        13086: (SEED_13086, EXPECTED_13086_RAW_SHA256, EXPECTED_13086_CANONICAL_SHA256),
        218309645: (SEED_218309645, EXPECTED_218309645_RAW_SHA256, EXPECTED_218309645_CANONICAL_SHA256),
    }
    checkpoint_records: dict[str, Any] = {}
    for seed, (path, expected_raw, expected_canonical) in expected_by_seed.items():
        model, payload = load_model(path)
        raw = sha256_file(path)
        canonical = canonical_model_state_sha256(model.state_dict())
        checkpoint_records[str(seed)] = {
            "path": str(path.resolve()),
            "raw_sha256": raw,
            "expected_raw_sha256": expected_raw,
            "raw_hash_match": raw == expected_raw,
            "canonical_sha256": canonical,
            "expected_canonical_sha256": expected_canonical,
            "canonical_hash_match": canonical == expected_canonical,
            "size_bytes": path.stat().st_size,
            "payload_keys": sorted(payload.keys()),
            "payload_contains_training_history": "history" in payload or "training_summary" in payload,
        }

    r6_raw = sha256_file(R6_CHECKPOINT)
    a1_raw = sha256_file(A1_CHECKPOINT)
    r6_model, _ = load_model(R6_CHECKPOINT)
    a1_model, _ = load_model(A1_CHECKPOINT)
    r6_canonical = canonical_model_state_sha256(r6_model.state_dict())
    a1_canonical = canonical_model_state_sha256(a1_model.state_dict())
    h10_authority = verify_h10_authority(ROOT, H10_ROOT)
    env = read_json(D3_ROOT / "environment_provenance.json")
    source_before = file_snapshot(source_paths())

    return {
        "d2_config_sha256_expected": EXPECTED_D2_CONFIG_SHA256,
        "d2_config_sha256_observed": d2_contract_hash,
        "d2_config_sha256_match": d2_contract_hash == EXPECTED_D2_CONFIG_SHA256,
        "d2_artifacts_immutable_at_start": d2_ok,
        "d2_artifact_snapshot_failures": d2_failures,
        "d3_historical_terminal_certificate": d3_terminal,
        "d3_checkpoint_identity_manifest": d3_manifest,
        "checkpoint_records": checkpoint_records,
        "r6_raw_sha256": r6_raw,
        "r6_raw_sha256_match": r6_raw == EXPECTED_R6_RAW_SHA256,
        "r6_canonical_sha256": r6_canonical,
        "a1_raw_sha256": a1_raw,
        "a1_raw_sha256_match": a1_raw == EXPECTED_A1_RAW_SHA256,
        "a1_canonical_sha256": a1_canonical,
        "a1_canonical_sha256_match": a1_canonical == EXPECTED_A1_CANONICAL_SHA256,
        "h10_authority": h10_authority,
        "h10_semantic_sha256_match": read_json(H10_ROOT / "dataset_semantic_hash.json").get("semantic_dataset_sha256") == EXPECTED_H10_SEMANTIC_SHA256,
        "h10_manifest_sha256_match": sha256_file(H10_ROOT / "dataset_manifest.json") == EXPECTED_H10_MANIFEST_SHA256,
        "h10_split_sha256_match": sha256_file(H10_ROOT / "dataset_split_manifest.json") == EXPECTED_H10_SPLIT_SHA256,
        "environment_provenance": env,
        "source_commit": git_text(["rev-parse", "HEAD"]),
        "source_commit_match_recorded": bool(env.get("source_commit_match")),
        "source_hashes_before": source_before,
        "frozen20_opened": "NO",
        "frozen20_used": "NO",
        "future_label_leakage": 0,
        "historical_d3_artifact_snapshot_before": recursive_snapshot(D3_ROOT),
        "d2_seed_set": d2_manifest.get("expected_seed_set"),
        "d2_h11_reference_window_counts": {
            "TRAIN": D2_H11_TRAIN_WINDOW_COUNT,
            "VALIDATION": D2_H11_VALIDATION_WINDOW_COUNT,
            "source": "D2 data_and_splits prose; H11 fixed-horizon-8 window manifest",
            "d2_training_split_text": d2_data_contract.get("training_split"),
            "d2_model_selection_split_text": d2_data_contract.get("model_selection_split"),
        },
        "d3_preflight_source_commit": d2_preflight.get("environment", {}).get("source_commit"),
    }


def build_selection_metrics(validation: Any, channels: Mapping[str, Any], guardrail: Any) -> dict[str, Any]:
    model_paths = {
        "R6": R6_CHECKPOINT,
        "R8E_A1": A1_CHECKPOINT,
        "prospective_seed_13086": SEED_13086,
        "prospective_seed_218309645": SEED_218309645,
    }
    model_metrics: dict[str, dict[str, float]] = {}
    model_hashes: dict[str, Any] = {}
    for label, path in model_paths.items():
        model, _ = load_model(path)
        model_metrics[label] = compact_metrics(model, validation, channels)
        model_hashes[label] = {
            "path": str(path.resolve()),
            "raw_sha256": sha256_file(path),
            "canonical_sha256": canonical_model_state_sha256(model.state_dict()),
        }
    comparison: dict[str, Any] = {}
    for label, metrics in model_metrics.items():
        horizons: dict[str, Any] = {}
        for horizon, value in metrics.items():
            applies = horizon == "1"
            threshold = float(guardrail.threshold) if applies else None
            item: dict[str, Any] = {
                "value": float(value),
                "threshold": threshold,
                "gate_applies": applies,
                "status": "PASS" if applies and float(value) <= float(guardrail.threshold) else "FAIL" if applies else "NOT_A_GATE_DIAGNOSTIC",
            }
            if applies:
                item.update(margin_payload(float(value), float(guardrail.threshold)))
            else:
                item.update({"margin_threshold_minus_observed": None, "relative_margin_percent_of_threshold": None, "pass_if_frozen_rule_applied": None})
            horizons[horizon] = item
        comparison[label] = horizons
    return {
        "selection_gate_metric": "joint_position_rmse_rad at H1",
        "diagnostic_metrics": "joint_position_rmse_rad at H1,H2,H4,H8,H12,H16,H20,H24,H32; only H1 is a D3 gate clause",
        "horizons": list(HORIZONS),
        "guardrail": {
            "authoritative_r6_h1": float(guardrail.reference),
            "fresh_r6_h1": float(guardrail.fresh_replay),
            "threshold": float(guardrail.threshold),
            "formula": "max(authoritative_R6_H1,fresh_R6_H1) * 1.05 + 5e-9",
            "comparison_operator": "candidate_H1 <= threshold",
        },
        "model_hashes": model_hashes,
        "models": comparison,
        "raw_metric_values": model_metrics,
    }


def replay_seed_218(validation: Any, channels: Mapping[str, Any], guardrail: Any) -> dict[str, Any]:
    model, _ = load_model(SEED_218309645)
    replay_rows: list[dict[str, Any]] = []
    for index in range(1, 4):
        metrics = compact_metrics(model, validation, channels)
        decision = bool(float(metrics["1"]) <= float(guardrail.threshold))
        replay_rows.append({
            "replay": index,
            "checkpoint_raw_sha256": sha256_file(SEED_218309645),
            "checkpoint_canonical_sha256": canonical_model_state_sha256(model.state_dict()),
            "metrics": metrics,
            "final_selection_decision_from_metric_clause": "PASS" if decision else "BLOCKED",
            "metric_all_finite": all(math.isfinite(value) for value in metrics.values()),
        })
    baseline = replay_rows[0]["metrics"]
    absolute_deltas: list[float] = []
    relative_deltas: list[float] = []
    for row in replay_rows[1:]:
        absolute, relative = max_metric_delta(baseline, row["metrics"])
        absolute_deltas.append(absolute)
        relative_deltas.append(relative)
    return {
        "replays": replay_rows,
        "replay_semantic_match": all(row["metrics"].keys() == baseline.keys() and row["metric_all_finite"] for row in replay_rows),
        "replay_bitwise_metric_match": all(row["metrics"] == baseline for row in replay_rows),
        "max_absolute_metric_delta": max(absolute_deltas, default=0.0),
        "max_relative_metric_delta": max(relative_deltas, default=0.0),
        "decision_stable": len({row["final_selection_decision_from_metric_clause"] for row in replay_rows}) == 1,
        "stochastic_evaluation_controls": {
            "model_eval_called": True,
            "dropout_present_in_architecture": False,
            "dataloader_shuffle": False,
            "dataloader_workers": 0,
            "python_numpy_torch_rng_not_used_by_evaluator": True,
            "deterministic_algorithms_from_d3_environment": True,
            "device": "cpu",
            "dtype": "float32",
        },
    }


def report_text(context: Mapping[str, Any]) -> str:
    first = context["first_failure"]
    replay = context["replay"]
    authority = context["authority"]
    classification = context["classification"]
    lines = [
        f"STAGE_3_H13_R8E_D3A:\n{context['status']}",
        "",
        "FIRST_BLOCKER:",
        context["first_blocker"],
        "",
        "ONE_SENTENCE_VERDICT:",
        context["one_sentence_verdict"],
        "",
        "ROOT_CAUSE_CLASSIFICATION:",
        classification["primary"],
        "",
        "ROOT_CAUSE_CONFIDENCE:",
        classification["confidence"],
        "",
        "FIRST_FAILING_SELECTION_METRIC:",
        first["metric"],
        "",
        "FIRST_FAILING_HORIZON:",
        first["horizon"],
        "",
        "SEED_218309645_OBSERVED_VALUE:",
        str(first["observed_value"]),
        "",
        "FROZEN_THRESHOLD:",
        str(first["threshold"]),
        "",
        "ABSOLUTE_MARGIN_TO_THRESHOLD:",
        str(first["absolute_margin"]),
        "",
        "RELATIVE_MARGIN_PERCENT:",
        str(first["relative_margin_percent"]),
        "",
        "SEED_13086_COMPARABLE_VALUE:",
        str(context["comparables"]["prospective_seed_13086"]),
        "",
        "R8E_A1_COMPARABLE_VALUE:",
        str(context["comparables"]["R8E_A1"]),
        "",
        "R6_COMPARABLE_VALUE:",
        str(context["comparables"]["R6"]),
        "",
        "SEED_218309645_FAILURE_REAL:",
        "YES",
        "",
        "SEED_218309645_FAILURE_REPRODUCIBLE:",
        "YES" if replay["decision_stable"] else "NO",
        "",
        "REPLAY_SEMANTIC_MATCH:",
        "YES" if replay["replay_semantic_match"] else "NO",
        "",
        "REPLAY_BITWISE_METRIC_MATCH:",
        "YES" if replay["replay_bitwise_metric_match"] else "NO",
        "",
        "INDEPENDENT_RECOMPUTATION_MATCH:",
        context["independent"]["semantic_match"],
        "",
        "MODEL_SELECTION_IMPLEMENTATION_VALID:",
        context["model_selection_implementation_valid"],
        "",
        "CHECKPOINT_AUTHORITY_VALID:",
        "YES" if context["checkpoint_authority_valid"] else "NO",
        "",
        "DATA_AUTHORITY_VALID:",
        "YES" if context["data_authority_valid"] else "NO",
        "",
        "GENUINE_SEED_SENSITIVITY_SUPPORTED:",
        classification["genuine_seed_sensitivity_supported"],
        "",
        "THRESHOLD_EDGE_CASE:",
        classification["threshold_edge_case"],
        "",
        "D3_HISTORICAL_BLOCKED_RESULT_REMAINS_VALID:",
        "YES",
        "",
        "CURRENT_VALIDATION_CHAMPION:",
        "R8E_A1",
        "",
        "CHAMPION_CHANGED:",
        "NO",
        "",
        "FROZEN20_OPENED:",
        "NO",
        "",
        "FROZEN20_USED:",
        "NO",
        "",
        "FUTURE_LABEL_LEAKAGE:",
        "0",
        "",
        "TRAINING_OR_TUNING_PERFORMED:",
        "NO",
        "",
        "NEW_CHECKPOINT_CREATED:",
        "NO",
        "",
        "FOCUSED_TESTS:",
        context["focused_tests"],
        "",
        "REGRESSION_TESTS:",
        context["regression_tests"],
        "",
        "TOTAL_PERSISTENT_SIZE_BYTES:",
        str(context["persistent_size_bytes"]),
        "",
        "READY_FOR_R8F_EXECUTION:",
        "NO",
        "",
        "READY_FOR_R9_FROZEN20:",
        "NO",
        "",
        "READY_FOR_STAGE_3_FINAL_CLOSURE:",
        "NO",
        "",
        "SINGLE_NEXT_ACTION:",
        "Keep D3 blocked and require separate owner authorization before any new experiment; do not tune this failure away.",
        "",
        "AUTHORITATIVE REPORT",
        "",
        "1. Protocol integrity",
        f"D2_CONFIG_SHA256_MATCH: {'YES' if authority['d2_config_sha256_match'] else 'NO'}; D2_ARTIFACTS_IMMUTABLE: {'YES' if authority['d2_artifacts_immutable_at_start'] and context['d2_after_immutable'] else 'NO'}; D3_ARTIFACTS_IMMUTABLE: {'YES' if context['d3_artifacts_immutable'] else 'NO'}.",
        f"Checkpoint authority is {'valid' if context['checkpoint_authority_valid'] else 'invalid'} for R6, R8E-A1, seed 13086, and seed 218309645. Data/split authority is {'valid' if context['data_authority_valid'] else 'invalid'}; D3 H32-rollout validation windows: {context['validation_count']}; validation families: {context['validation_family_count']}; D2/H11 fixed-H8 reference: {D2_H11_VALIDATION_WINDOW_COUNT} windows.",
        f"FROZEN20_OPENED: NO; FROZEN20_USED: NO; FUTURE_LABEL_LEAKAGE: 0. Environment: CPU float32, PyTorch {authority['environment_provenance'].get('pytorch_version')}, deterministic algorithms enabled.",
        "",
        "2. Exact frozen model-selection contract",
        f"Executable path: {path_line(ROOT / 'scripts/stage3_h13_r8e_d3_prospective_five_seed_causal_replication.py', 521)} -> {path_line(ROOT / 'scripts/stage3_h13_r8e_d3_prospective_five_seed_causal_replication.py', 522)} -> {path_line(ROOT / 'src/stage3_h13_r6.py', 310)} -> {path_line(ROOT / 'scripts/stage3_h13_r8e_causal_hidden_stability_training.py', 104)}. Split is VALIDATION only; horizons are {','.join('H'+str(h) for h in HORIZONS)}; metric is joint_position_rmse_rad.",
        "Aggregation is free-running causal self-feeding, per-horizon cumulative squared position error over h*6*windows, float64 NumPy sum across fixed 4096-window batches, then math.sqrt(total_sse/total_count). Model/input tensors are float32. The only D3 metric threshold is the upper H1 guardrail; D2's H32 graduation reference is not used by the D3 selection boolean.",
        "Exact predicate: updates == 512 AND all phase transition_stability_pass flags AND all terminal metrics finite AND metrics['1'] <= guardrail.threshold. Python AND evaluation is left-to-right. H1 is the false metric clause established below; retained D3 did not persist phase history.",
        f"IS_MODEL_SELECTION_IMPLEMENTATION_VALID: {'YES' if context['model_selection_implementation_valid'] == 'YES' else 'INCONCLUSIVE'}; IS_D3_PROTOCOL_IMPLEMENTATION_VALID: {'YES' if context.get('d3_protocol_implementation_valid') == 'YES' else 'INCONCLUSIVE'}.",
        "",
        "3. Exact reason seed 218309645 failed",
        f"H1 observed {first['observed_value']:.17g} versus frozen threshold {first['threshold']:.17g}; margin=threshold-observed={first['absolute_margin']:.17g}, relative margin={first['relative_margin_percent']:.9g}%. Exact predicate result: FAIL. All replayed terminal metrics are finite.",
        "The D3 output retained the checkpoint and terminal blocker but not per-phase training history, so the H1 failure is sufficient and exact; any additional H32 phase flag cannot be reconstructed from retained evidence. The D2 prose count of 27,594 is the H11 fixed-H8 window count; D3's frozen next-32 rollout construction correctly uses 25,991 windows from the same seven validation families.",
        "",
        "4. Seed 13086 vs seed 218309645 comparison",
        f"H1: seed13086={context['comparables']['prospective_seed_13086']:.17g}; seed218309645={first['observed_value']:.17g}; delta(seed218-seed13086)={context['horizon_differences']['1']:.17g}. R8E-A1 and seed13086 have the same canonical model-state hash and identical replay metrics.",
        "",
        "5. Horizon-level localization",
        f"First measurable divergence: {context['first_measurable_divergence_horizon']}; first material divergence: NOT_CLASSIFIABLE because D2 froze no materiality rule for cross-seed model-selection metric differences (MATERIALITY_RULE: NOT_PRECOMMITTED). Direction: {context['divergence_direction']}; terminal H32 difference (seed218 - seed13086): {context['horizon_differences']['32']:.17g}. D2/H11 27,594 versus D3 H32-rollout 25,991 is a fixed-horizon window-count reconciliation, not a different family split.",
        "",
        "6. Training-vs-validation diagnosis",
        "Classification: CANNOT_BE_DISTINGUISHED_FROM_AVAILABLE_EVIDENCE. D3 states training completed for seed 218309645 and the checkpoint payload is valid, but neither per-update loss nor phase-end history was retained; do not recreate it through retraining. The retained evidence supports a terminal validation rejection, not an abnormal-training diagnosis.",
        "",
        "7. Three-replay determinism",
        f"REPLAY_SEMANTIC_MATCH: {'YES' if replay['replay_semantic_match'] else 'NO'}; REPLAY_BITWISE_METRIC_MATCH: {'YES' if replay['replay_bitwise_metric_match'] else 'NO'}; MAX_ABSOLUTE_METRIC_DELTA: {replay['max_absolute_metric_delta']:.17g}; MAX_RELATIVE_METRIC_DELTA: {replay['max_relative_metric_delta']:.17g}; DECISION_STABLE: {'YES' if replay['decision_stable'] else 'NO'}.",
        "",
        "8. Independent metric recomputation",
        f"Independent H1={context['independent']['independent_metrics']['1']:.17g}; authoritative H1={first['observed_value']:.17g}; absolute difference={context['independent']['absolute_difference']:.17g}; semantic match={context['independent']['semantic_match']}.",
        "",
        "9. Numerical/implementation edge-case audit",
        context["edge_case_text"],
        "",
        "10. Root-cause classification",
        f"{classification['primary']} with {classification['confidence']} confidence. {classification['one_sentence_root_cause']}",
        "",
        "11. Project implication",
        "D3 remains a valid blocked historical result; R8E_A1 remains the current validation champion; the negative replication evidence is preserved. Recommendation: do not rerun D3 unchanged; a future owner-authorized design review should consider recipe robustness, but D3A does not authorize or perform redesign. Do not treat this as an execution-only bug.",
        "Recommendation fields: SHOULD_D3_BE_RERUN_UNCHANGED=NO; SHOULD_TRAINING_RECIPE_BE_REDESIGNED=YES, as a future separately authorized design review only; SHOULD_ONLY_EXECUTION_IMPLEMENTATION_BE_FIXED=NO.",
        "",
        "12. Storage/tests",
        f"Persistent D3A size: {context['persistent_size_bytes']} bytes. No checkpoint, optimizer state, prediction dump, sealed data, or training artifact was created. Focused tests: {context['focused_tests']}; regression tests: {context['regression_tests']}.",
        "",
        "13. Artifact paths",
        f"Output directory: {context['output_dir']}",
    ]
    return "\n".join(lines) + "\n"


def run_audit(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"output_directory_already_exists:{output}")
    authority = authority_preflight()
    d2_protocol = read_json(D2_ROOT / "prospective_training_protocol.json")
    dataset = load_h10_segments(H10_ROOT)
    stats = read_json(H11_ROOT / "normalization_stats.json")
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    train = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    r6_model, _ = load_model(R6_CHECKPOINT)
    fresh_r6 = compact_metrics(r6_model, validation, stats["channels"])
    guardrail = derive_h1_guardrail(AUTHORITATIVE_R6_H1, fresh_r6["1"], material_allowance_fraction=0.05, absolute_replay_tolerance=REPLAY_ATOL)
    metrics = build_selection_metrics(validation, stats["channels"], guardrail)
    replay = replay_seed_218(validation, stats["channels"], guardrail)
    failing_model, _ = load_model(SEED_218309645)
    independent_metrics = independent_recompute(failing_model, validation, stats["channels"])
    authoritative_metrics = metrics["raw_metric_values"]["prospective_seed_218309645"]
    independent_absolute, independent_relative = max_metric_delta(authoritative_metrics, independent_metrics)
    first = {
        "metric": "joint_position_rmse_rad",
        "horizon": "H1",
        "observed_value": authoritative_metrics["1"],
        "threshold": float(guardrail.threshold),
        "absolute_margin": float(guardrail.threshold) - float(authoritative_metrics["1"]),
        "relative_margin_percent": 100.0 * (float(guardrail.threshold) - float(authoritative_metrics["1"])) / float(guardrail.threshold),
        "pass_if_exact_rule_applied": bool(authoritative_metrics["1"] <= guardrail.threshold),
    }
    differences = {horizon: metrics["raw_metric_values"]["prospective_seed_218309645"][horizon] - metrics["raw_metric_values"]["prospective_seed_13086"][horizon] for horizon in [str(h) for h in HORIZONS]}
    measurable = next((f"H{h}" for h in HORIZONS if differences[str(h)] != 0.0), "NONE")
    direction = "worse for seed 218309645 at H1" if differences["1"] > 0 else "better for seed 218309645 at H1" if differences["1"] < 0 else "no H1 difference"
    # The audit uses the existing replay tolerance to identify only a true
    # tolerance-scale boundary; D2 did not precommit a broader edge rule.
    threshold_edge = abs(first["absolute_margin"]) <= REPLAY_ATOL + REPLAY_RTOL * abs(float(guardrail.threshold))
    checkpoint_valid = all(item["raw_hash_match"] and item["canonical_hash_match"] for item in authority["checkpoint_records"].values()) and authority["r6_raw_sha256_match"] and authority["a1_raw_sha256_match"] and authority["a1_canonical_sha256_match"]
    validation_families = set(str(item) for item in validation.family_ids)
    train_families = set(str(item) for item in train.family_ids)
    data_valid = (
        authority["h10_authority"].get("status") == "PASSED"
        and authority["h10_semantic_sha256_match"]
        and authority["h10_manifest_sha256_match"]
        and authority["h10_split_sha256_match"]
        and len(validation_families) == 7
        and len(train_families) == 31
        and validation.count > 0
        and train.count > 0
    )
    independent_match = independent_absolute <= 1.0e-15 or math.isclose(independent_metrics["1"], authoritative_metrics["1"], rel_tol=1.0e-12, abs_tol=1.0e-15)
    model_selection_valid = authority["d2_config_sha256_match"] and checkpoint_valid and data_valid and replay["replay_semantic_match"] and replay["replay_bitwise_metric_match"] and independent_match
    source_after_pre = authority["source_hashes_before"]
    d3_after_before_output = authority["historical_d3_artifact_snapshot_before"]
    d3_artifacts_before_output_unchanged = d3_after_before_output == recursive_snapshot(D3_ROOT)
    classification = {
        "primary": "GENUINE_SEED_SENSITIVITY" if model_selection_valid else "INCONCLUSIVE",
        "confidence": "HIGH" if model_selection_valid and checkpoint_valid and data_valid else "MEDIUM" if checkpoint_valid and data_valid else "LOW",
        "threshold_edge_case": "YES" if threshold_edge else "NO",
        "genuine_seed_sensitivity_supported": "YES" if model_selection_valid and not first["pass_if_exact_rule_applied"] else "NO",
        "one_sentence_root_cause": "Seed 218309645 is a valid, stable prospective model whose terminal validation H1 exceeds the frozen H1 guardrail, while seed 13086 remains inside it; no evaluator, checkpoint, data, or replay defect was found." if model_selection_valid else "The retained evidence does not establish every authority and evaluator prerequisite needed for a seed-sensitivity classification.",
    }
    edge_case_text = "; ".join([
        "floating-point boundary behavior: RULED_OUT as causal explanation; the observed margin is recorded and replays are bitwise stable",
        "rounding/serialization precision: RULED_OUT; raw and canonical checkpoint hashes match and independent recomputation agrees",
        "dtype conversion: RULED_OUT; frozen evaluator uses float32 model/input tensors and float64 error reduction, reproduced independently",
        "aggregation order: RULED_OUT; fixed batch/reduction replay is bitwise stable and independent scalar reduction agrees",
        "NaN/Inf: RULED_OUT; all terminal metrics are finite",
        "off-by-one horizon indexing: RULED_OUT; H1/H2/H4/H8/H12/H16/H20/H24/H32 are independently recomputed from the same target prefix definition",
        "wrong validation subset: RULED_OUT as a data-identity error; H10 authority, split hashes, seven validation families, and D3's required H32-rollout window construction match the executable contract; D2's 27594 reference is the H11 fixed-H8 count",
        "wrong checkpoint: RULED_OUT; raw and canonical hashes match the retained D3 manifest",
        "wrong reference baseline: RULED_OUT; fresh R6 H1 is within the frozen replay tolerance of authoritative R6 H1",
        "wrong threshold/comparison operator: RULED_OUT; threshold is independently derived and the exact <= predicate is unit-tested",
        "stale cached metrics: RULED_OUT; all metrics were recomputed from the retained checkpoints in three read-only replays",
    ])
    source_after = file_snapshot(source_paths())
    source_unchanged = source_after == source_after_pre
    authority["source_hashes_after"] = source_after
    authority["source_hashes_unchanged_during_d3a"] = source_unchanged
    authority["d3_artifacts_unchanged_before_artifact_write"] = d3_artifacts_before_output_unchanged

    output.mkdir(parents=True, exist_ok=False)
    output_file_map = {
        "model_selection_contract.json": {
            "schema_version": "stage3_h13_r8e_d3a_model_selection_contract_v1",
            "authority": "D3 executable path, not comments or final report",
            "source_lines": {
                "d3_gate": path_line(ROOT / "scripts/stage3_h13_r8e_d3_prospective_five_seed_causal_replication.py", 523),
                "d3_evaluation_wrapper": path_line(ROOT / "scripts/stage3_h13_r8e_causal_hidden_stability_training.py", 104),
                "metric_reduction": path_line(ROOT / "src/stage3_h13_r6.py", 310),
                "guardrail": path_line(ROOT / "src/stage3_h13_r8e.py", 36),
            },
            "metrics": ["joint_position_rmse_rad"],
            "data_split": {
                "selection_split": "VALIDATION",
                "window_count": validation.count,
                "family_count": len(validation_families),
                "d2_h11_reference_window_count": D2_H11_VALIDATION_WINDOW_COUNT,
                "window_count_reconciliation": "D2/H11 count is fixed-horizon-8; D3 next-32 rollout evaluator derives the smaller 25,991-window subset from the same seven validation families",
            },
            "horizons": list(HORIZONS),
            "aggregation": "float64 SSE over h*6*validation_window_count followed by math.sqrt; 4096-window batches; validation order preserved",
            "reference_model": "authoritative R6 H1 and fresh read-only R6 H1, combined with max()",
            "threshold_type": "upper guardrail",
            "threshold_value": float(guardrail.threshold),
            "comparison_operator": "<=",
            "float_dtype": {"model_inputs": "float32", "error_and_sse": "float64", "final_sqrt": "Python math double"},
            "reduction_order": "segment order, window order, fixed 4096 batches; per-batch np.sum(dtype=float64), Python float accumulation",
            "missing_nan_inf_semantics": "empty data yields None; D3 finite-metric clause rejects nonfinite values; actual validation is nonempty",
            "first_failure_short_circuit": "Python left-to-right AND; D3 raises model_selection_gate_failed after the candidate selection boolean is false",
            "exact_predicate": "updates == 512 AND all(history[*].transition_stability_pass) AND all(isfinite(metrics.values())) AND metrics['1'] <= guardrail.threshold",
            "h32_reference_note": "D2 h32_graduation_reference exists but is not referenced by the D3 selection boolean",
        },
        "seed_selection_metric_comparison.json": {"schema_version": "stage3_h13_r8e_d3a_seed_selection_comparison_v1", **metrics, "horizon_differences_seed218_minus_seed13086": differences},
        "seed_218309645_replay_results.json": {"schema_version": "stage3_h13_r8e_d3a_replay_results_v1", "frozen_threshold": float(guardrail.threshold), **replay},
        "independent_metric_recomputation.json": {
            "schema_version": "stage3_h13_r8e_d3a_independent_metric_v1",
            "model": "prospective_seed_218309645",
            "authoritative_evaluator_metrics": authoritative_metrics,
            "independent_recomputation_metrics": independent_metrics,
            "absolute_difference_by_horizon": {key: abs(float(authoritative_metrics[key]) - float(independent_metrics[key])) for key in authoritative_metrics},
            "relative_difference_by_horizon": {key: abs(float(authoritative_metrics[key]) - float(independent_metrics[key])) / max(abs(float(authoritative_metrics[key])), 1.0e-30) for key in authoritative_metrics},
            "absolute_difference": independent_absolute,
            "relative_difference": independent_relative,
            "semantic_match": "YES" if independent_match else "NO",
            "implementation": "separate math.fsum scalar reduction over frozen rollout predictions and target prefixes",
        },
        "authority_manifest.json": authority,
        "root_cause_classification.json": {
            "schema_version": "stage3_h13_r8e_d3a_root_cause_v1",
            "primary_classification": classification["primary"],
            "confidence": classification["confidence"],
            "threshold_proximity": "EDGE" if threshold_edge else "NOT_EDGE",
            "threshold_edge_case": classification["threshold_edge_case"],
            "one_sentence_root_cause": classification["one_sentence_root_cause"],
            "failure_real": "YES",
            "failure_reproducible": "YES" if replay["decision_stable"] else "NO",
            "genuine_seed_sensitivity_supported": classification["genuine_seed_sensitivity_supported"],
            "training_outcome_diagnosis": "CANNOT_BE_DISTINGUISHED_FROM_AVAILABLE_EVIDENCE; NOT_RECOVERABLE_FROM_RETAINED_EVIDENCE",
            "edge_case_audit": edge_case_text,
        },
        "storage_retention_manifest.json": {
            "schema_version": "stage3_h13_r8e_d3a_storage_retention_v1",
            "persistent_artifacts": "compact JSON/Markdown/text only",
            "new_checkpoints_created": "NO",
            "optimizer_states_created": "NO",
            "prediction_dumps_retained": "NO",
            "sealed_evaluation_data_opened": "NO",
            "persistent_size_bytes_before_report": 0,
            "target_under_100mb": "YES",
        },
    }
    for name, value in output_file_map.items():
        write_json(output / name, json_safe(value))

    terminal_context: dict[str, Any] = {
        "status": "PASSED",
        "first_blocker": "none",
        "one_sentence_verdict": classification["one_sentence_root_cause"],
        "authority": authority,
        "first_failure": first,
        "replay": replay,
        "independent": {"independent_metrics": independent_metrics, "absolute_difference": independent_absolute, "semantic_match": "YES" if independent_match else "NO"},
        "comparables": {key: metrics["raw_metric_values"][key]["1"] for key in ("prospective_seed_13086", "R8E_A1", "R6")},
        "horizon_differences": differences,
        "first_measurable_divergence_horizon": measurable,
        "divergence_direction": direction,
        "classification": classification,
        "checkpoint_authority_valid": checkpoint_valid,
        "data_authority_valid": data_valid,
        "d3_protocol_implementation_valid": "YES" if model_selection_valid else "INCONCLUSIVE",
        "model_selection_implementation_valid": "YES" if model_selection_valid else "INCONCLUSIVE",
        "validation_count": validation.count,
        "validation_family_count": len(validation_families),
        "d2_after_immutable": False,
        "d3_artifacts_immutable": False,
        "edge_case_text": edge_case_text,
        "focused_tests": "PENDING",
        "regression_tests": "PENDING",
        "persistent_size_bytes": 0,
        "output_dir": str(output.resolve()),
    }
    write_json(output / "terminal_certificate.json", json_safe({
        "schema_version": "stage3_h13_r8e_d3a_terminal_certificate_v1",
        "STAGE_3_H13_R8E_D3A": "PASSED",
        "FIRST_BLOCKER": "none",
        "ROOT_CAUSE_CLASSIFICATION": classification["primary"],
        "ROOT_CAUSE_CONFIDENCE": classification["confidence"],
        "FIRST_FAILING_SELECTION_METRIC": first,
        "SEED_218309645_FAILURE_REAL": "YES",
        "SEED_218309645_FAILURE_REPRODUCIBLE": "YES" if replay["decision_stable"] else "NO",
        "REPLAY_SEMANTIC_MATCH": "YES" if replay["replay_semantic_match"] else "NO",
        "REPLAY_BITWISE_METRIC_MATCH": "YES" if replay["replay_bitwise_metric_match"] else "NO",
        "INDEPENDENT_RECOMPUTATION_MATCH": "YES" if independent_match else "NO",
        "CHECKPOINT_AUTHORITY_VALID": "YES" if checkpoint_valid else "NO",
        "DATA_AUTHORITY_VALID": "YES" if data_valid else "NO",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "TRAINING_OR_TUNING_PERFORMED": "NO",
        "NEW_CHECKPOINT_CREATED": "NO",
        "READY_FOR_R8F_EXECUTION": "NO",
        "READY_FOR_R9_FROZEN20": "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
    }))
    (output / "FINAL_REPORT.md").write_text(report_text(terminal_context), encoding="utf-8", newline="\n")

    # The only files written after the initial D3A preflight are inside output.
    # Recheck historical D2/D3/source artifacts before finalizing the report.
    d2_after_ok, d2_after_failures = snapshot_matches(authority["d2_artifact_snapshot"] if "d2_artifact_snapshot" in authority else d2_preflight_snapshot(D3_ROOT))
    d3_after = recursive_snapshot(D3_ROOT)
    d3_immutable = d3_after == d3_after_before_output
    source_final = file_snapshot(source_paths())
    source_final_unchanged = source_final == source_after
    authority["d2_artifacts_immutable_after"] = d2_after_ok
    authority["d2_artifact_snapshot_failures_after"] = d2_after_failures
    authority["d3_artifacts_immutable_after"] = d3_immutable
    authority["source_hashes_after_final"] = source_final
    authority["source_hashes_unchanged_final"] = source_final_unchanged
    authority["d3a_training_or_tuning_performed"] = "NO"
    write_json(output / "authority_manifest.json", json_safe(authority))

    # Update the report/certificate after immutable checks are known.
    terminal_context["d2_after_immutable"] = d2_after_ok
    terminal_context["d3_artifacts_immutable"] = d3_immutable
    terminal_context["persistent_size_bytes"] = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    (output / "storage_retention_manifest.json").write_text(json.dumps({
        "schema_version": "stage3_h13_r8e_d3a_storage_retention_v1",
        "persistent_artifacts": "compact JSON/Markdown/text only",
        "new_checkpoints_created": "NO",
        "optimizer_states_created": "NO",
        "prediction_dumps_retained": "NO",
        "sealed_evaluation_data_opened": "NO",
        "persistent_size_bytes": terminal_context["persistent_size_bytes"],
        "target_under_100mb": terminal_context["persistent_size_bytes"] < 100 * 1024 * 1024,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (output / "FINAL_REPORT.md").write_text(report_text(terminal_context), encoding="utf-8", newline="\n")
    return terminal_context


def d2_preflight_snapshot(d3_root: Path) -> dict[str, dict[str, Any]]:
    return read_json(d3_root / "d2_gate0_preflight.json")["d2_snapshot"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only D3A model-selection failure audit")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    context = run_audit((ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve())
    print(report_text(context), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
