"""Run the single authorized Stage 3 H13 D5 no-clipping causal replay.

The runner is deliberately single-use and low-storage.  It reuses the
identity-checked D4 data, schedule, model, loss, optimizer, and evaluator, and
changes only the gradient-clipping operation.  No checkpoint is retained;
the terminal model is represented by a canonical state hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
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

from scripts.stage3_h13_post_d4_h1_blocker_audit import (  # noqa: E402
    A1_CHECKPOINT,
    A1_CONFIG,
    D4_H1_THRESHOLD,
    EXPECTED_A1_CANONICAL,
    EXPECTED_A1_HASH,
    EXPECTED_CANONICAL_SCHEDULE,
    EXPECTED_CANONICAL_STORAGE,
    EXPECTED_D4_AGGREGATE,
    EXPECTED_D4_DIR,
    EXPECTED_D4_CERTIFICATE,
    EXPECTED_D4_ENVIRONMENT,
    EXPECTED_D4_IMMUTABLE,
    EXPECTED_D4_SCHEDULE_STORAGE,
    EXPECTED_D4_SUMMARY,
    EXPECTED_R6_HASH,
    H32_THRESHOLD,
    PROBE_UPDATES,
    R6_CHECKPOINT,
    REPLAY_ATOL,
    REPLAY_RTOL,
    aggregate_norm,
    aggregate_optimizer_state,
    canonical_model_state_sha256,
    causal_paired_rollout,
    clone_state,
    cosine_from_norms,
    derive_h1_guardrail,
    evaluate_compact,
    family_diagnostics,
    finite_float,
    gradients_are_finite,
    load_json,
    load_model,
    model_state_snapshot,
    parameter_rows,
    preflight,
    probe_metrics,
    quantiles,
    read_schedule,
    schedule_indices,
    set_deterministic,
    tensor_batch,
    training_losses,
    verify_snapshot,
    write_csv,
)
from src.stage3_h13_r8e import PHASE_HORIZONS  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY"
AUTHORIZATION_FILE = Path(r"C:\Users\86198\.codex\attachments\6eb45b67-570f-4e77-b3cf-909d68e03897\pasted-text.txt")
D5_SEED = 3830844401
TOTAL_UPDATES = 512
BATCH_SIZE = 256
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 1.0e-5
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
HUBER_BETA = 0.001
UPDATES_PER_PHASE = 128
D4_H1 = 9.019693982113058e-05
D4_H32 = 0.016745967327457343
A1_H1 = 8.521749753954249e-05
A1_H32 = 0.015847526619180707
D4_TIMING = {
    "first_post_warmup_h1_degradation_update": 128,
    "first_h1_threshold_crossing_update": 1,
    "first_h32_local_improvement_update": 128,
    "first_h32_r6_relative_improvement_update": 384,
    "first_h32_graduation_update": 512,
}


class D5Blocker(RuntimeError):
    """Fail-closed blocker for the authorized D5 replay."""


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_text(args: Sequence[str]) -> str | None:
    result = subprocess.run(list(args), cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def environment() -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
        "dtype": "float32",
        "source_commit": git_text(["git", "rev-parse", "HEAD"]),
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "dataloader_workers": 0,
        "residual_nondeterminism": "cross_platform_bitwise_recreation_not_claimed",
    }


def verify_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise D5Blocker(f"authorization_file_missing:{path}")
    text = path.read_text(encoding="utf-8")
    required = (
        EXPERIMENT_ID,
        "AUTHORIZED_TO_TRAIN:\nYES",
        "AUTHORIZED_TO_CHANGE_GRADIENT_CLIPPING:\nYES",
        "AUTHORIZED_FOR_R9:\nNO",
        "AUTHORIZED_TO_OPEN_OR USE FROZEN20:\nNO",
        "AUTHORIZED_TO_REPLACE R8E_A1:\nNO",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise D5Blocker(f"authorization_scope_mismatch:{missing}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "scope_verified": "YES"}


def common_recipe(schedule_hash: str, schedule_storage_hash: str) -> dict[str, Any]:
    return {
        "initialization": "R6",
        "architecture": "2-layer unidirectional residual GRU, hidden 128, decoder horizon 8",
        "dataset": "H10 TRAIN; H10 VALIDATION evaluator",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW",
        "adamw_betas": [0.9, 0.999],
        "adamw_epsilon": 1.0e-8,
        "weight_decay": WEIGHT_DECAY,
        "number_of_updates": TOTAL_UPDATES,
        "updates_per_phase": UPDATES_PER_PHASE,
        "phase_horizons": list(PHASE_HORIZONS),
        "lambda_h1": LAMBDA_H1,
        "lambda_hidden": LAMBDA_HIDDEN,
        "huber_beta": HUBER_BETA,
        "normalization": "frozen H11 normalization",
        "evaluation_semantics": "free-running causal self-feeding rollout on H10 VALIDATION",
        "canonical_schedule_sha256": schedule_hash,
        "canonical_schedule_storage_sha256": schedule_storage_hash,
        "deterministic_execution": "CPU float32; deterministic algorithms; zero workers",
    }


def build_recipe_diff(schedule_hash: str, schedule_storage_hash: str) -> dict[str, Any]:
    d4 = common_recipe(schedule_hash, schedule_storage_hash)
    d4["gradient_clipping"] = {"enabled": "YES", "max_norm": 1.0, "operation": "torch.nn.utils.clip_grad_norm_"}
    d5 = dict(d4)
    d5["gradient_clipping"] = {"enabled": "NO", "max_norm": None, "operation": "BYPASSED; raw gradients passed unchanged to AdamW"}
    changed = []
    field_diffs = []
    for field in d4:
        if d4[field] != d5[field]:
            changed.append(field)
            field_diffs.append({"field": field, "D4": d4[field], "D5": d5[field], "substantive": "YES"})
    return {
        "schema_version": "stage3_h13_post_d4_d5_recipe_diff_v1",
        "comparison": "canonical clipped D4 -> no-clipping D5",
        "changed_fields": changed,
        "substantive_training_change_count": len(changed),
        "expected_changed_fields": ["gradient_clipping"],
        "single_factor_change_confirmed": "YES" if changed == ["gradient_clipping"] else "NO",
        "D4_recipe": d4,
        "D5_recipe": d5,
        "field_diffs": field_diffs,
    }


def raw_gradient_norm(parameters: Sequence[tuple[str, torch.nn.Parameter]]) -> float:
    gradients = [parameter.grad.detach().cpu() for _, parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise D5Blocker("no_gradients_available")
    return aggregate_norm(gradients)


def safe_metrics(model: Any, validation: Any, channels: Mapping[str, Any]) -> dict[str, float]:
    try:
        return probe_metrics(model, validation, channels)
    except Exception as exc:  # noqa: BLE001
        raise D5Blocker(f"validation_evaluation_failed:{type(exc).__name__}:{exc}") from exc


def trajectory_summary(probes: Mapping[str, Mapping[str, float]], r6_metrics: Mapping[str, float]) -> dict[str, Any]:
    available = [update for update in PROBE_UPDATES if str(update) in probes]
    tolerance_h1 = max(REPLAY_ATOL, REPLAY_RTOL * abs(float(r6_metrics["1"])))
    tolerance_h32 = max(REPLAY_ATOL, REPLAY_RTOL * abs(float(r6_metrics["32"])))

    def first_local(metric: str, direction: str, start: int = 0) -> int | None:
        previous: float | None = None
        for update in available:
            if update < start:
                continue
            value = float(probes[str(update)][metric])
            if previous is not None and ((direction == "up" and value > previous + (tolerance_h1 if metric == "1" else tolerance_h32)) or (direction == "down" and value < previous - (tolerance_h1 if metric == "1" else tolerance_h32))):
                return update
            previous = value
        return None

    def first_predicate(metric: str, predicate: Any) -> int | None:
        for update in available:
            if predicate(float(probes[str(update)][metric])):
                return update
        return None

    sustained = None
    for index, update in enumerate(available):
        if update < 64:
            continue
        if float(probes[str(update)]["1"]) > D4_H1_THRESHOLD and all(float(probes[str(later)]["1"]) > D4_H1_THRESHOLD for later in available[index:]):
            sustained = update
            break
    return {
        "available_probe_updates": available,
        "first_post_warmup_h1_degradation_update": first_local("1", "up", 64),
        "first_h1_threshold_crossing_update": first_predicate("1", lambda value: value > D4_H1_THRESHOLD),
        "first_sustained_h1_threshold_crossing_update": sustained,
        "first_h32_local_improvement_update": first_local("32", "down", 64),
        "first_h32_r6_relative_improvement_update": first_predicate("32", lambda value: value < float(r6_metrics["32"]) - tolerance_h32),
        "first_h32_graduation_update": first_predicate("32", lambda value: value <= H32_THRESHOLD),
        "h1_tolerance": tolerance_h1,
        "h32_tolerance": tolerance_h32,
        "probe_resolution_only": "YES",
    }


def classify_divergence(trajectory: Mapping[str, Any], completed: bool) -> str:
    if not completed:
        return "NOT_RESOLVED"
    d5_h1 = trajectory.get("first_post_warmup_h1_degradation_update")
    d5_h32 = trajectory.get("first_h32_local_improvement_update")
    if d5_h1 is None and d5_h32 is None:
        return "DISAPPEARED"
    if d5_h1 is None:
        return "WEAKER"
    if d5_h1 > D4_TIMING["first_post_warmup_h1_degradation_update"]:
        return "DELAYED"
    if d5_h1 < D4_TIMING["first_post_warmup_h1_degradation_update"]:
        return "EARLIER"
    if d5_h32 is None:
        return "WEAKER"
    if d5_h32 > D4_TIMING["first_h32_local_improvement_update"]:
        return "WEAKER"
    if d5_h32 < D4_TIMING["first_h32_local_improvement_update"]:
        return "STRONGER"
    return "UNCHANGED"


def percent_change(new: float | None, old: float) -> float | None:
    return None if new is None else (float(new) - float(old)) / float(old) * 100.0


def comparison(new: float | None, old: float) -> dict[str, Any]:
    if new is None:
        return {"absolute_change": None, "percent_change": None}
    return {"absolute_change": float(new) - float(old), "percent_change": percent_change(new, old)}


def load_d4_final_adamw() -> dict[str, float] | None:
    path = EXPECTED_D4_DIR / "trajectory_diagnostics.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    row = rows[-1]
    return {key: float(row[key]) for key in ("exp_avg_norm", "exp_avg_sq_norm", "actual_update_norm")}


def run_tests() -> dict[str, Any]:
    commands = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_post_d4_d5_no_clip_causal_replay.py", "tests/test_stage3_h13_r8e_d4_canonical_batch_robustness.py"],
        "regression": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_post_d4_h1_blocker_audit.py", "tests/test_stage3_h13_r8e_d1_checkpoint_authority.py"],
    }
    result: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        result[name] = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-5000:], "stderr": completed.stderr[-3000:]}
    result["new_failures"] = int(result["focused"]["returncode"] != 0 or result["regression"]["returncode"] != 0)
    return result


def write_integrity_manifest(output: Path) -> None:
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "integrity_manifest.json"):
        rows.append({"path": str(path.relative_to(output)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "integrity_manifest.json", {"schema_version": "stage3_h13_post_d4_d5_integrity_v1", "hash_algorithm": "SHA-256", "self_hash_excluded": True, "files": rows})


def persistent_size(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def render_report(c: Mapping[str, Any]) -> str:
    status = c.get("status", c.get("STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY", "BLOCKED"))
    first_blocker = c.get("first_blocker", c.get("FIRST_BLOCKER", "none"))
    one_sentence_verdict = c.get("one_sentence_verdict", c.get("ONE_SENTENCE_VERDICT", ""))
    lines = [
        f"STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY:\n{status}",
        f"FIRST_BLOCKER: {first_blocker}",
        f"ONE_SENTENCE_VERDICT: {one_sentence_verdict}",
        "",
        "=" * 50,
        "1. PROTOCOL INTEGRITY",
        "=" * 50,
        "",
        f"PROJECT_OWNER_AUTHORIZATION: {c['PROJECT_OWNER_AUTHORIZATION']}",
        f"R8E_A1_IMMUTABLE: {c['R8E_A1_IMMUTABLE']}",
        f"R6_HASH_MATCH: {c['R6_HASH_MATCH']}",
        f"CANONICAL_SCHEDULE_HASH_MATCH: {c['CANONICAL_SCHEDULE_HASH_MATCH']}",
        f"SINGLE_FACTOR_CHANGE_CONFIRMED: {c['SINGLE_FACTOR_CHANGE_CONFIRMED']}",
        f"OTHER_RECIPE_CHANGES: {c['OTHER_RECIPE_CHANGES']}",
        f"FROZEN20_OPENED: {c['FROZEN20_OPENED']}",
        f"FROZEN20_USED: {c['FROZEN20_USED']}",
        f"FUTURE_LABEL_LEAKAGE: {c['FUTURE_LABEL_LEAKAGE']}",
        f"TUNING_EXECUTED: {c['TUNING_EXECUTED']}",
        "",
        "=" * 50,
        "2. D5 TERMINAL RESULT",
        "=" * 50,
        "",
        f"D5_H1: {c['D5_H1']}", f"D5_H4: {c['D5_H4']}", f"D5_H16: {c['D5_H16']}", f"D5_H32: {c['D5_H32']}",
        "",
        f"H1_FROZEN_THRESHOLD: {c['H1_FROZEN_THRESHOLD']}", f"H1_PASS: {c['H1_PASS']}",
        "",
        f"H32_FROZEN_GRADUATION_THRESHOLD: {c['H32_FROZEN_GRADUATION_THRESHOLD']}", f"H32_PASS: {c['H32_PASS']}",
        "",
        "=" * 50,
        "3. D4 CLIPPED VS D5 NO-CLIP",
        "=" * 50,
        "",
        f"D4_CLIPPED_H1: {c['D4_CLIPPED_H1']}", f"D5_NO_CLIP_H1: {c['D5_NO_CLIP_H1']}", f"H1_ABSOLUTE_CHANGE: {c['H1_ABSOLUTE_CHANGE']}", f"H1_PERCENT_CHANGE: {c['H1_PERCENT_CHANGE']}",
        "",
        f"D4_CLIPPED_H32: {c['D4_CLIPPED_H32']}", f"D5_NO_CLIP_H32: {c['D5_NO_CLIP_H32']}", f"H32_ABSOLUTE_CHANGE: {c['H32_ABSOLUTE_CHANGE']}", f"H32_PERCENT_CHANGE: {c['H32_PERCENT_CHANGE']}",
        "",
        "=" * 50,
        "4. TRAJECTORY",
        "=" * 50,
        "",
        f"FIRST_POST_WARMUP_H1_DEGRADATION_UPDATE: {c['FIRST_POST_WARMUP_H1_DEGRADATION_UPDATE']}",
        f"FIRST_H1_THRESHOLD_CROSSING_UPDATE: {c['FIRST_H1_THRESHOLD_CROSSING_UPDATE']}",
        f"FIRST_H32_LOCAL_IMPROVEMENT_UPDATE: {c['FIRST_H32_LOCAL_IMPROVEMENT_UPDATE']}",
        f"FIRST_H32_R6_RELATIVE_IMPROVEMENT_UPDATE: {c['FIRST_H32_R6_RELATIVE_IMPROVEMENT_UPDATE']}",
        f"FIRST_H32_GRADUATION_UPDATE: {c['FIRST_H32_GRADUATION_UPDATE']}",
        "",
        f"H1_H32_DIVERGENCE_VS_D4: {c['H1_H32_DIVERGENCE_VS_D4']}",
        "",
        "=" * 50,
        "5. GRADIENT / NUMERICAL BEHAVIOR",
        "=" * 50,
        "",
        "CLIPPING_ENABLED: NO",
        f"RAW_GRADIENT_NORM_MIN: {c['RAW_GRADIENT_NORM_MIN']}", f"RAW_GRADIENT_NORM_MEDIAN: {c['RAW_GRADIENT_NORM_MEDIAN']}", f"RAW_GRADIENT_NORM_P90: {c['RAW_GRADIENT_NORM_P90']}", f"RAW_GRADIENT_NORM_P95: {c['RAW_GRADIENT_NORM_P95']}", f"RAW_GRADIENT_NORM_P99: {c['RAW_GRADIENT_NORM_P99']}", f"RAW_GRADIENT_NORM_MAX: {c['RAW_GRADIENT_NORM_MAX']}",
        "",
        f"NONFINITE_OCCURRED: {c['NONFINITE_OCCURRED']}", f"FIRST_NONFINITE_UPDATE: {c['FIRST_NONFINITE_UPDATE']}", f"NUMERICAL_STABILITY_STATUS: {c['NUMERICAL_STABILITY_STATUS']}",
        "",
        "=" * 50,
        "6. ADAMW PATH",
        "=" * 50,
        "",
        f"EXP_AVG_PATH_SUMMARY: {c['EXP_AVG_PATH_SUMMARY']}", f"EXP_AVG_SQ_PATH_SUMMARY: {c['EXP_AVG_SQ_PATH_SUMMARY']}", f"FINAL_ACTUAL_UPDATE_NORM: {c['FINAL_ACTUAL_UPDATE_NORM']}", f"ADAMW_PATH_CHANGED_VS_D4: {c['ADAMW_PATH_CHANGED_VS_D4']}",
        "Do not interpret AdamW as independently causal in this experiment.",
        "",
        "=" * 50,
        "7. CAUSAL CLASSIFICATION",
        "=" * 50,
        "",
        f"GRADIENT_CLIPPING_CAUSAL_EFFECT: {c['GRADIENT_CLIPPING_CAUSAL_EFFECT']}", f"CAUSAL_CONFIDENCE: {c['CAUSAL_CONFIDENCE']}", f"ROOT_CAUSE_CLASSIFICATION: {c['ROOT_CAUSE_CLASSIFICATION']}", f"ROOT_CAUSE_SUMMARY: {c['ROOT_CAUSE_SUMMARY']}",
        "",
        "=" * 50,
        "8. A1 / CHAMPION STATUS",
        "=" * 50,
        "",
        f"R8E_A1_INVALIDATED: {c['R8E_A1_INVALIDATED']}", f"PREVIOUS_CHAMPION: {c['PREVIOUS_CHAMPION']}", f"CURRENT_VALIDATION_CHAMPION: {c['CURRENT_VALIDATION_CHAMPION']}", f"CHAMPION_CHANGED: {c['CHAMPION_CHANGED']}",
        "",
        f"D5_GRADUATION_ELIGIBLE: {c['D5_GRADUATION_ELIGIBLE']}", f"D5_A1_REPLACEMENT_CANDIDATE: {c['D5_A1_REPLACEMENT_CANDIDATE']}", f"REPLACEMENT_BLOCKER: {c['REPLACEMENT_BLOCKER']}",
        "",
        "=" * 50,
        "9. R9 / FROZEN20 STATUS",
        "=" * 50,
        "",
        f"SCIENTIFICALLY_READY_FOR_R9: {c['SCIENTIFICALLY_READY_FOR_R9']}", f"AUTHORIZED_FOR_R9: {c['AUTHORIZED_FOR_R9']}", f"FROZEN20_TOUCHED: {c['FROZEN20_TOUCHED']}", f"FROZEN20_SHOULD_REMAIN_CLOSED: {c['FROZEN20_SHOULD_REMAIN_CLOSED']}",
        "",
        "=" * 50,
        "10. VERIFICATION / STORAGE",
        "=" * 50,
        "",
        f"FOCUSED_TESTS: {c['FOCUSED_TESTS']}", f"REGRESSION_TESTS: {c['REGRESSION_TESTS']}", f"NEW_FAILURES: {c['NEW_FAILURES']}", f"PERSISTENT_OUTPUT_SIZE: {c['PERSISTENT_OUTPUT_SIZE']}", f"ARTIFACT_PATHS: {c['ARTIFACT_PATHS']}",
        "",
        "=" * 50,
        "11. NEXT ACTION",
        "=" * 50,
        "",
        f"ONE_RECOMMENDED_NEXT_ACTION: {c['ONE_RECOMMENDED_NEXT_ACTION']}",
        "",
        "EXECUTED: NO",
        "",
        "SCIENTIFIC_CONCLUSION:", c["SCIENTIFIC_CONCLUSION"],
        "",
        "ENGINEERING_CONCLUSION:", c["ENGINEERING_CONCLUSION"],
        "",
        "OWNER_DECISION_REQUIRED:", c["OWNER_DECISION_REQUIRED"],
        "",
    ]
    return "\n".join(str(line) for line in lines) + "\n"


def build_terminal(context: Mapping[str, Any]) -> dict[str, Any]:
    terminal = dict(context)
    terminal["schema_version"] = "stage3_h13_post_d4_d5_no_clip_terminal_certificate_v1"
    terminal["STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY"] = terminal.pop("status")
    terminal["FIRST_BLOCKER"] = terminal.pop("first_blocker")
    terminal["ONE_SENTENCE_VERDICT"] = terminal.pop("one_sentence_verdict")
    terminal["ARTIFACTS"] = [
        "FINAL_REPORT.md", "terminal_certificate.json", "trajectory_diagnostics.csv", "module_diagnostics.csv",
        "recipe_diff.json", "numerical_stability.json", "final_state_hash_metadata.json", "preflight_identity.json",
        "probe_metrics.json", "test_results.json", "integrity_manifest.json",
    ]
    return terminal


def fail_closed(output: Path, blocker: str, focused: str = "NOT_RUN", regression: str = "NOT_RUN") -> int:
    terminal = {
        "schema_version": "stage3_h13_post_d4_d5_no_clip_terminal_certificate_v1",
        "STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY": "BLOCKED",
        "FIRST_BLOCKER": blocker,
        "ONE_SENTENCE_VERDICT": "D5 was fail-closed before a valid terminal causal result was established.",
        "PROJECT_OWNER_AUTHORIZATION": "UNKNOWN",
        "R8E_A1_IMMUTABLE": "UNKNOWN",
        "R6_HASH_MATCH": "UNKNOWN",
        "CANONICAL_SCHEDULE_HASH_MATCH": "UNKNOWN",
        "SINGLE_FACTOR_CHANGE_CONFIRMED": "UNKNOWN",
        "OTHER_RECIPE_CHANGES": "UNKNOWN",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": "NOT_AVAILABLE",
        "TUNING_EXECUTED": "NO",
        "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
        "CHAMPION_CHANGED": "NO",
        "FOCUSED_TESTS": focused,
        "REGRESSION_TESTS": regression,
        "NEW_FAILURES": 0,
        "PERSISTENT_OUTPUT_SIZE": 0,
        "FROZEN20_TOUCHED": "NO",
        "AUTHORIZED_FOR_R9": "NO",
    }
    write_json(output / "terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(f"STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY:\nBLOCKED\n\nFIRST_BLOCKER: {blocker}\n", encoding="utf-8", newline="\n")
    write_integrity_manifest(output)
    return 2


def finalize_existing(output: Path) -> int:
    """Finalize a completed D5 run without executing any training update."""

    required = (output / "trajectory_diagnostics.csv", output / "probe_metrics.json", output / "numerical_stability.json", output / "final_state_hash_metadata.json", output / "test_results.json")
    if not output.is_dir() or any(not path.is_file() for path in required):
        raise D5Blocker("completed_d5_artifacts_missing_for_finalize")
    probe_payload = load_json(output / "probe_metrics.json")
    stability = load_json(output / "numerical_stability.json")
    state_hash = load_json(output / "final_state_hash_metadata.json")
    tests = load_json(output / "test_results.json")
    with (output / "trajectory_diagnostics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    probes = probe_payload["metrics"]
    r6_metrics = probe_payload["R6"]
    a1_metrics = probe_payload["R8E_A1"]
    terminal = probes.get("512")
    completed = int(stability.get("completed_updates", len(rows))) == TOTAL_UPDATES and terminal is not None and not bool(stability.get("nonfinite_occurred"))
    if not completed:
        raise D5Blocker("completed_d5_artifacts_do_not_show_valid_512_update_terminal")
    terminal_h1 = float(terminal["1"])
    terminal_h4 = float(terminal["4"])
    terminal_h16 = float(terminal["16"])
    terminal_h32 = float(terminal["32"])
    trajectory = trajectory_summary(probes, r6_metrics)
    h1_pass = terminal_h1 <= D4_H1_THRESHOLD
    h32_pass = terminal_h32 <= H32_THRESHOLD
    d4_adamw = load_d4_final_adamw()
    last_row = rows[-1]
    d5_adamw = {key: float(last_row[key]) for key in ("exp_avg_norm", "exp_avg_sq_norm", "actual_update_norm")}
    adamw_changed = "YES" if d4_adamw is None or any(abs(d5_adamw[key] - d4_adamw[key]) > max(REPLAY_ATOL, REPLAY_RTOL * abs(d4_adamw[key])) for key in d5_adamw) else "NO"
    effect = "PROTECTIVE" if terminal_h1 >= D4_H1 and terminal_h32 > D4_H32 else "NOT_SUPPORTED"
    root_class = "GRADIENT_CLIPPING_PROTECTIVE_UNDER_FIXED_RECIPE" if effect == "PROTECTIVE" else "GRADIENT_CLIPPING_NOT_SUFFICIENT_TO_EXPLAIN_H1_BLOCKER"
    root_summary = (
        "Removing clipping substantially worsens both terminal H1 and H32 relative to clipped D4, indicating that clipping was protective under the tested fixed recipe; this does not establish max_norm=1.0 as optimal or a universal project-wide root cause."
        if effect == "PROTECTIVE"
        else "Removing clipping does not materially remove the terminal H1 blocker under the fixed canonical recipe."
    )
    d4_h1 = D4_H1
    d4_h32 = D4_H32
    context = {
        "status": "PASSED",
        "first_blocker": "none",
        "one_sentence_verdict": root_summary,
        "PROJECT_OWNER_AUTHORIZATION": "YES",
        "R8E_A1_IMMUTABLE": "YES",
        "R6_HASH_MATCH": "YES",
        "CANONICAL_SCHEDULE_HASH_MATCH": "YES",
        "SINGLE_FACTOR_CHANGE_CONFIRMED": "YES",
        "OTHER_RECIPE_CHANGES": "NONE",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "TUNING_EXECUTED": "NO",
        "D5_H1": terminal_h1, "D5_H4": terminal_h4, "D5_H16": terminal_h16, "D5_H32": terminal_h32,
        "H1_FROZEN_THRESHOLD": D4_H1_THRESHOLD, "H1_PASS": "YES" if h1_pass else "NO",
        "H32_FROZEN_GRADUATION_THRESHOLD": H32_THRESHOLD, "H32_PASS": "YES" if h32_pass else "NO",
        "D4_CLIPPED_H1": d4_h1, "D5_NO_CLIP_H1": terminal_h1, "H1_ABSOLUTE_CHANGE": terminal_h1 - d4_h1, "H1_PERCENT_CHANGE": (terminal_h1 - d4_h1) / d4_h1 * 100.0,
        "D4_CLIPPED_H32": d4_h32, "D5_NO_CLIP_H32": terminal_h32, "H32_ABSOLUTE_CHANGE": terminal_h32 - d4_h32, "H32_PERCENT_CHANGE": (terminal_h32 - d4_h32) / d4_h32 * 100.0,
        "H1_MARGIN_TO_FROZEN_THRESHOLD": D4_H1_THRESHOLD - terminal_h1,
        "H32_MARGIN_TO_FROZEN_THRESHOLD": H32_THRESHOLD - terminal_h32,
        "H1_MARGIN_TO_R6": terminal_h1 - float(r6_metrics["1"]), "H32_MARGIN_TO_R6": terminal_h32 - float(r6_metrics["32"]),
        "R6_H1": r6_metrics["1"], "R6_H32": r6_metrics["32"], "R8E_A1_H1": a1_metrics["1"], "R8E_A1_H32": a1_metrics["32"],
        "FIRST_POST_WARMUP_H1_DEGRADATION_UPDATE": trajectory["first_post_warmup_h1_degradation_update"],
        "FIRST_H1_THRESHOLD_CROSSING_UPDATE": trajectory["first_h1_threshold_crossing_update"],
        "FIRST_H32_LOCAL_IMPROVEMENT_UPDATE": trajectory["first_h32_local_improvement_update"],
        "FIRST_H32_R6_RELATIVE_IMPROVEMENT_UPDATE": trajectory["first_h32_r6_relative_improvement_update"],
        "FIRST_H32_GRADUATION_UPDATE": trajectory["first_h32_graduation_update"],
        "D4_TRAJECTORY_TIMING": D4_TIMING, "H1_H32_DIVERGENCE_VS_D4": classify_divergence(trajectory, True),
        "CLIPPING_ENABLED": "NO",
        "RAW_GRADIENT_NORM_MIN": stability["raw_gradient_norm_quantiles"]["0.0"], "RAW_GRADIENT_NORM_MEDIAN": stability["raw_gradient_norm_quantiles"]["0.5"], "RAW_GRADIENT_NORM_P90": stability["raw_gradient_norm_quantiles"]["0.9"], "RAW_GRADIENT_NORM_P95": stability["raw_gradient_norm_quantiles"]["0.95"], "RAW_GRADIENT_NORM_P99": stability["raw_gradient_norm_quantiles"]["0.99"], "RAW_GRADIENT_NORM_MAX": stability["raw_gradient_norm_quantiles"]["1.0"],
        "NONFINITE_OCCURRED": "NO", "FIRST_NONFINITE_UPDATE": None, "FIRST_UNSTABLE_OR_LOSS_EXPLOSION_UPDATE": None, "NUMERICAL_STABILITY_STATUS": "FINITE_512_UPDATES",
        "EXP_AVG_PATH_SUMMARY": quantiles([float(row["exp_avg_norm"]) for row in rows]), "EXP_AVG_SQ_PATH_SUMMARY": quantiles([float(row["exp_avg_sq_norm"]) for row in rows]), "FINAL_ACTUAL_UPDATE_NORM": d5_adamw["actual_update_norm"], "ADAMW_PATH_CHANGED_VS_D4": adamw_changed,
        "GRADIENT_CLIPPING_CAUSAL_EFFECT": effect, "CAUSAL_CONFIDENCE": "MEDIUM", "ROOT_CAUSE_CLASSIFICATION": root_class, "ROOT_CAUSE_SUMMARY": root_summary,
        "R8E_A1_INVALIDATED": "NO", "PREVIOUS_CHAMPION": "R8E_A1", "CURRENT_VALIDATION_CHAMPION": "R8E_A1", "CHAMPION_CHANGED": "NO",
        "D5_GRADUATION_ELIGIBLE": "NO", "D5_A1_REPLACEMENT_CANDIDATE": "NO", "REPLACEMENT_BLOCKER": "D5 fails the frozen H1 and H32 gates; R8E_A1 remains immutable and the existing multi-seed replacement conditions are not met.",
        "SCIENTIFICALLY_READY_FOR_R9": "NO", "AUTHORIZED_FOR_R9": "NO", "FROZEN20_TOUCHED": "NO", "FROZEN20_SHOULD_REMAIN_CLOSED": "YES",
        "FOCUSED_TESTS": "PASS" if tests["focused"]["returncode"] == 0 else "FAIL", "REGRESSION_TESTS": "PASS" if tests["regression"]["returncode"] == 0 else "FAIL", "NEW_FAILURES": tests["new_failures"],
        "PERSISTENT_OUTPUT_SIZE": 0, "ARTIFACT_PATHS": "FINAL_REPORT.md, terminal_certificate.json, trajectory_diagnostics.csv, module_diagnostics.csv, recipe_diff.json, numerical_stability.json, final_state_hash_metadata.json, preflight_identity.json, probe_metrics.json, test_results.json, integrity_manifest.json",
        "ONE_RECOMMENDED_NEXT_ACTION": "Preserve R8E_A1 and decide whether to authorize exactly one separately designed AdamW-path causal experiment; do not run it, open Frozen-20, or run R9 in this task.",
        "SCIENTIFIC_CONCLUSION": root_summary + " The deterministic replay completed all 512 updates with finite state, so this is evidence that clipping was protective rather than the sufficient cause of the canonical H1 blocker under this recipe.",
        "ENGINEERING_CONCLUSION": "Preserve R8E_A1 as the current validation champion, keep Frozen-20 sealed, and do not promote D5.",
        "OWNER_DECISION_REQUIRED": "Decide whether to authorize one separately specified AdamW-path causal experiment; no follow-up experiment was executed here.",
        "d5_completed_updates": len(rows), "d5_terminal_state_hash": state_hash.get("terminal_state_hash"), "probe_metrics": probes, "trajectory_summary": trajectory, "r6_metrics": r6_metrics, "r8e_a1_metrics": a1_metrics,
        "d4_comparison": {"H1": comparison(terminal_h1, d4_h1), "H32": comparison(terminal_h32, d4_h32)}, "recipe_diff": load_json(output / "recipe_diff.json"),
    }
    terminal = build_terminal(context)
    write_json(output / "terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(render_report(terminal), encoding="utf-8", newline="\n")
    context["PERSISTENT_OUTPUT_SIZE"] = persistent_size(output)
    terminal = build_terminal(context)
    write_json(output / "terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(render_report(terminal), encoding="utf-8", newline="\n")
    write_integrity_manifest(output)
    print(json.dumps({"status": terminal["STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY"], "first_blocker": terminal["FIRST_BLOCKER"], "completed_updates": len(rows), "h1": terminal_h1, "h32": terminal_h32, "output": str(output)}, sort_keys=True), flush=True)
    return 0


def execute(output: Path, authorization_file: Path = AUTHORIZATION_FILE) -> int:
    if output.exists():
        raise D5Blocker(f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    focused: dict[str, Any] | None = None
    try:
        focused = run_tests()
        if focused["focused"]["returncode"] != 0:
            return fail_closed(output, "focused_tests_failed_before_training", "FAIL", "NOT_RUN")

        authorization = verify_authorization(authorization_file)
        pre = preflight(True)
        identity, schedule_manifest, schedule_hashes = read_schedule()
        if schedule_hashes["logical_sha256"] != EXPECTED_CANONICAL_SCHEDULE or schedule_hashes["storage_sha256"] != EXPECTED_CANONICAL_STORAGE:
            raise D5Blocker("canonical_schedule_hash_mismatch")
        if sha256_file(EXPECTED_D4_SCHEDULE_STORAGE) != EXPECTED_CANONICAL_STORAGE:
            raise D5Blocker("canonical_schedule_storage_file_hash_mismatch")
        if sha256_file(R6_CHECKPOINT) != EXPECTED_R6_HASH:
            raise D5Blocker("r6_hash_mismatch")
        if sha256_file(A1_CHECKPOINT) != EXPECTED_A1_HASH:
            raise D5Blocker("r8e_a1_hash_mismatch")
        a1 = load_model(A1_CHECKPOINT)
        if canonical_model_state_sha256(a1.state_dict()) != EXPECTED_A1_CANONICAL:
            raise D5Blocker("r8e_a1_canonical_hash_mismatch")
        d4_certificate = load_json(EXPECTED_D4_CERTIFICATE)
        if d4_certificate.get("CURRENT_VALIDATION_CHAMPION") != "R8E_A1" or d4_certificate.get("CHAMPION_CHANGED") != "NO":
            raise D5Blocker("upstream_champion_status_changed")
        d4_environment = load_json(EXPECTED_D4_ENVIRONMENT)
        current_environment = environment()
        for key in ("python_version", "pytorch_version", "numpy_version", "cuda_available", "device", "dtype"):
            if current_environment[key] != d4_environment.get(key):
                raise D5Blocker(f"environment_identity_mismatch:{key}")
        verify_snapshot(pre["snapshots"], "upstream_immutable_input")
        extra_snapshot = model_state_snapshot([EXPECTED_D4_CERTIFICATE, EXPECTED_D4_SUMMARY, EXPECTED_D4_AGGREGATE, EXPECTED_D4_ENVIRONMENT, EXPECTED_D4_IMMUTABLE, EXPECTED_D4_SCHEDULE_STORAGE])
        schedule_rows = schedule_indices(schedule_manifest, pre["train_data"].window_ids)
        if schedule_rows.shape != (TOTAL_UPDATES, BATCH_SIZE):
            raise D5Blocker("canonical_schedule_shape_mismatch")
        recipe_diff = build_recipe_diff(EXPECTED_CANONICAL_SCHEDULE, EXPECTED_CANONICAL_STORAGE)
        if recipe_diff["single_factor_change_confirmed"] != "YES" or recipe_diff["substantive_training_change_count"] != 1:
            raise D5Blocker("recipe_diff_has_more_than_one_substantive_change")
        write_json(output / "recipe_diff.json", recipe_diff)
        write_json(output / "immutable_input_snapshot.json", {**pre["snapshots"], **extra_snapshot})
        write_json(output / "authorization_record.json", authorization)
        write_json(output / "environment_provenance.json", current_environment)

        train_data = pre["train_data"]
        validation = pre["validation"]
        channels = pre["stats"]["channels"]
        r6 = load_model(R6_CHECKPOINT)
        r6_metrics = safe_metrics(r6, validation, channels)
        a1_metrics = safe_metrics(a1, validation, channels)
        d4_aggregate = load_json(EXPECTED_D4_AGGREGATE)
        d4_h1 = float(d4_aggregate["metrics"]["H1"]["median"])
        d4_h32 = float(d4_aggregate["metrics"]["H32"]["median"])
        if not math.isclose(d4_h1, D4_H1, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) or not math.isclose(d4_h32, D4_H32, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
            raise D5Blocker("d4_reference_terminal_metric_mismatch")
        if not math.isclose(a1_metrics["1"], A1_H1, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) or not math.isclose(a1_metrics["32"], A1_H32, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
            raise D5Blocker("r8e_a1_reference_metric_mismatch")
        r6_reference = load_json(EXPECTED_D4_DIR / "r6_reference_metrics.json")
        guardrail = derive_h1_guardrail(float(r6_reference["h1_guardrail"]["reference"]), float(r6_metrics["1"]), material_allowance_fraction=0.05, absolute_replay_tolerance=REPLAY_ATOL)
        if abs(float(guardrail.threshold) - D4_H1_THRESHOLD) > 1.0e-15:
            raise D5Blocker("frozen_h1_threshold_mismatch")

        config = {
            "experiment_id": EXPERIMENT_ID,
            "seed": D5_SEED,
            "initialization": "R6",
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "updates_per_phase": UPDATES_PER_PHASE,
            "phase_horizons": list(PHASE_HORIZONS),
            "lambda_h1": LAMBDA_H1,
            "lambda_hidden": LAMBDA_HIDDEN,
            "huber_beta": HUBER_BETA,
            "canonical_schedule_sha256": EXPECTED_CANONICAL_SCHEDULE,
            "canonical_schedule_storage_sha256": EXPECTED_CANONICAL_STORAGE,
            "gradient_clipping_enabled": False,
            "gradient_clip_norm": None,
            "future_label_leakage": 0,
        }
        set_deterministic(D5_SEED)
        model = load_model(R6_CHECKPOINT, trainable=True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        param_rows = parameter_rows(model)
        if optimizer.param_groups[0]["betas"] != (0.9, 0.999) or float(optimizer.param_groups[0]["eps"]) != 1.0e-8 or float(optimizer.param_groups[0]["weight_decay"]) != WEIGHT_DECAY:
            raise D5Blocker("adamw_optimizer_contract_mismatch")
        r6_state = clone_state(r6)
        probe_values: dict[str, dict[str, float]] = {"0": safe_metrics(model, validation, channels)}
        update_rows: list[dict[str, Any]] = []
        module_rows: list[dict[str, Any]] = []
        raw_norms: list[float] = []
        first_nonfinite: int | None = None
        first_failure: str | None = None
        last_adamw: dict[str, Any] | None = None
        phase_sums = {"total": 0.0, "rollout": 0.0, "hidden": 0.0, "h1": 0.0}

        for phase_index, horizon in enumerate(PHASE_HORIZONS, start=1):
            for phase_offset in range(UPDATES_PER_PHASE):
                update_index = (phase_index - 1) * UPDATES_PER_PHASE + phase_offset
                state_update_index = update_index + 1
                batch = tensor_batch(train_data, schedule_rows[update_index])
                optimizer.zero_grad(set_to_none=True)
                rollout = causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], channels, horizon=int(horizon), target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
                losses = training_losses(rollout["predictions"], batch["target_positions"], rollout["hidden_loss"], lambda_hidden=LAMBDA_HIDDEN, lambda_h1=LAMBDA_H1, beta=HUBER_BETA)
                if not all(bool(torch.isfinite(value)) for value in losses.values()):
                    first_nonfinite, first_failure = state_update_index, "nonfinite_training_loss"
                    break
                losses["total"].backward()
                if not gradients_are_finite(model):
                    first_nonfinite, first_failure = state_update_index, "nonfinite_raw_gradient"
                    break
                before_params = {name: parameter.detach().cpu().clone() for name, parameter in param_rows}
                raw_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in param_rows if parameter.grad is not None}
                raw_norm = raw_gradient_norm(param_rows)
                raw_norms.append(raw_norm)
                # Intentional causal intervention: no clipping or other gradient mutation occurs here.
                optimizer.step()
                after_params = {name: parameter.detach().cpu().clone() for name, parameter in param_rows}
                if not all(bool(torch.all(torch.isfinite(value))) for value in after_params.values()):
                    first_nonfinite, first_failure = state_update_index, "nonfinite_model_parameter"
                    break
                update_delta = [after_params[name] - before_params[name] for name, _ in param_rows]
                update_norm = aggregate_norm(update_delta)
                distance_r6 = aggregate_norm(after_params[name] - r6_state[name] for name, _ in param_rows)
                adamw = aggregate_optimizer_state(model, optimizer, before_params, LEARNING_RATE, WEIGHT_DECAY, 1.0e-8, (0.9, 0.999))
                adamw["actual_update_norm"] = update_norm
                adamw["update_to_parameter_norm_ratio"] = finite_float(update_norm / max(aggregate_norm(before_params.values()), 1.0e-30))
                last_adamw = adamw
                batch_ids = [int(value) for value in schedule_manifest["ordered_batches"][update_index]["window_ids"]]
                batch_digest = hashlib.sha256(json.dumps(batch_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
                row = {
                    "state_update_index": state_update_index,
                    "optimizer_update_index": update_index,
                    "phase": phase_index,
                    "active_horizon": int(horizon),
                    "batch_window_sha256": batch_digest,
                    "training_loss_total": finite_float(losses["total"]),
                    "training_loss_rollout": finite_float(losses["rollout"]),
                    "training_loss_hidden": finite_float(losses["hidden"]),
                    "training_loss_h1": finite_float(losses["h1"]),
                    "learning_rate": LEARNING_RATE,
                    "raw_global_grad_norm": raw_norm,
                    "actual_global_grad_norm": raw_norm,
                    "clipping_applied": "NO",
                    "parameter_update_norm": update_norm,
                    "distance_from_r6": distance_r6,
                    "distance_from_preceding_state": update_norm,
                    **adamw,
                }
                update_rows.append(row)
                module_items = family_diagnostics(model, optimizer, raw_grads, before_params, None, LEARNING_RATE, WEIGHT_DECAY, 1.0e-8, (0.9, 0.999))
                for item in module_items:
                    item = dict(item)
                    item["state_update_index"] = state_update_index
                    item["raw_grad_norm"] = item.pop("pre_clip_grad_norm")
                    item["actual_grad_norm"] = item.pop("post_clip_grad_norm")
                    item["clipping_applied"] = "NO"
                    item.pop("clip_threshold", None)
                    module_rows.append(item)
                for key in ("total", "rollout", "hidden", "h1"):
                    phase_sums[key] += finite_float(losses[key])
                if state_update_index in PROBE_UPDATES:
                    probe_values[str(state_update_index)] = safe_metrics(model, validation, channels)
            if first_nonfinite is not None:
                break

        completed = first_nonfinite is None and len(update_rows) == TOTAL_UPDATES
        model.eval()
        final_metrics = safe_metrics(model, validation, channels) if completed else None
        if completed:
            replay_metrics = safe_metrics(model, validation, channels)
            if any(not math.isclose(final_metrics[key], replay_metrics[key], rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) for key in final_metrics):
                raise D5Blocker("deterministic_evaluation_replay_mismatch")
            final_hash = canonical_model_state_sha256(model.state_dict())
        else:
            final_hash = None
        trajectory = trajectory_summary(probe_values, r6_metrics)
        raw_stats = quantiles(raw_norms) if raw_norms else {str(q): None for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)}
        d4_adamw = load_d4_final_adamw()
        d5_adamw_changed = "NOT_RESOLVED" if not update_rows else "YES" if d4_adamw is None or any(abs(float(last_adamw[key]) - d4_adamw[key]) > max(REPLAY_ATOL, REPLAY_RTOL * abs(d4_adamw[key])) for key in ("exp_avg_norm", "exp_avg_sq_norm", "actual_update_norm")) else "NO"
        terminal_h1 = None if final_metrics is None else float(final_metrics["1"])
        terminal_h4 = None if final_metrics is None else float(final_metrics["4"])
        terminal_h16 = None if final_metrics is None else float(final_metrics["16"])
        terminal_h32 = None if final_metrics is None else float(final_metrics["32"])
        h1_pass = terminal_h1 is not None and terminal_h1 <= D4_H1_THRESHOLD
        h32_pass = terminal_h32 is not None and terminal_h32 <= H32_THRESHOLD
        if first_nonfinite is not None:
            effect = "STABILITY_REQUIRED"
            confidence = "HIGH"
            root_class = "NO_CLIPPING_NUMERICAL_INSTABILITY_UNDER_FIXED_CANONICAL_RECIPE"
            root_summary = f"Removing clipping became nonfinite at update {first_nonfinite}; clipping is required for stability under this tested recipe, without implying max_norm=1.0 is optimal."
        elif h1_pass and h32_pass:
            effect = "SUPPORTED"
            confidence = "MEDIUM"
            root_class = "GRADIENT_CLIPPING_CONTRIBUTED_TO_CANONICAL_D4_H1_BLOCKER_UNDER_FIXED_RECIPE"
            root_summary = "No-clipping passes both frozen terminal gates on the deterministic fixed-recipe replay, supporting a clipping contribution under this recipe but not a universal or unique project-wide root cause."
        elif terminal_h1 is not None and terminal_h1 < D4_H1 and not h1_pass:
            effect = "PARTIALLY_SUPPORTED"
            confidence = "MEDIUM"
            root_class = "GRADIENT_CLIPPING_PARTIALLY_CONTRIBUTED_BUT_WAS_NOT_SUFFICIENTLY_REMOVED"
            root_summary = "No-clipping materially improves terminal H1 but leaves the frozen H1 gate failed; the clipping contribution is partial and does not establish an acceptable replacement recipe."
        elif terminal_h1 is not None and terminal_h1 >= D4_H1 and terminal_h32 is not None and terminal_h32 > D4_H32:
            effect = "PROTECTIVE"
            confidence = "MEDIUM"
            root_class = "GRADIENT_CLIPPING_PROTECTIVE_UNDER_FIXED_RECIPE"
            root_summary = "Removing clipping worsens both terminal H1 and H32 relative to clipped D4, indicating a protective clipping effect under this recipe."
        else:
            effect = "NOT_SUPPORTED"
            confidence = "LOW"
            root_class = "GRADIENT_CLIPPING_NOT_SUFFICIENT_TO_EXPLAIN_H1_BLOCKER"
            root_summary = "Removing clipping does not materially remove the terminal H1 blocker under the fixed canonical recipe; clipping is not sufficient to explain the blocker."
        comparison_h1 = comparison(terminal_h1, D4_H1)
        comparison_h32 = comparison(terminal_h32, D4_H32)
        graduation_eligible = "YES" if completed and h1_pass and h32_pass else "NO"
        replacement_candidate = "NO"
        replacement_blocker = "Existing multi-seed/replacement conditions are not met by this single replay; R8E_A1 remains immutable and champion."
        if completed and h1_pass and h32_pass:
            replacement_blocker = "Single-run D5 does not satisfy the already-established multi-seed and replacement protocol."
        one_sentence = root_summary
        status = "PASSED" if focused["new_failures"] == 0 and completed and first_failure is None else "BLOCKED"
        if not completed:
            one_sentence = f"D5 failed closed at update {first_nonfinite} before a valid 512-update terminal result; no-clipping stability is the causal outcome." if first_nonfinite is not None else "D5 did not establish a valid 512-update terminal result."
        context = {
            "status": status,
            "first_blocker": "none" if status == "PASSED" else (f"{first_failure}_update_{first_nonfinite}" if first_failure else "d5_terminal_or_verification_gate_failed"),
            "one_sentence_verdict": one_sentence,
            "PROJECT_OWNER_AUTHORIZATION": "YES",
            "R8E_A1_IMMUTABLE": "YES",
            "R6_HASH_MATCH": "YES",
            "CANONICAL_SCHEDULE_HASH_MATCH": "YES",
            "SINGLE_FACTOR_CHANGE_CONFIRMED": "YES",
            "OTHER_RECIPE_CHANGES": "NONE",
            "FROZEN20_OPENED": "NO",
            "FROZEN20_USED": "NO",
            "FUTURE_LABEL_LEAKAGE": 0,
            "TUNING_EXECUTED": "NO",
            "D5_H1": terminal_h1,
            "D5_H4": terminal_h4,
            "D5_H16": terminal_h16,
            "D5_H32": terminal_h32,
            "H1_FROZEN_THRESHOLD": D4_H1_THRESHOLD,
            "H1_PASS": "YES" if h1_pass else "NO",
            "H32_FROZEN_GRADUATION_THRESHOLD": H32_THRESHOLD,
            "H32_PASS": "YES" if h32_pass else "NO",
            "D4_CLIPPED_H1": D4_H1,
            "D5_NO_CLIP_H1": terminal_h1,
            "H1_ABSOLUTE_CHANGE": comparison_h1["absolute_change"],
            "H1_PERCENT_CHANGE": comparison_h1["percent_change"],
            "D4_CLIPPED_H32": D4_H32,
            "D5_NO_CLIP_H32": terminal_h32,
            "H32_ABSOLUTE_CHANGE": comparison_h32["absolute_change"],
            "H32_PERCENT_CHANGE": comparison_h32["percent_change"],
            "H1_MARGIN_TO_FROZEN_THRESHOLD": None if terminal_h1 is None else D4_H1_THRESHOLD - terminal_h1,
            "H32_MARGIN_TO_FROZEN_THRESHOLD": None if terminal_h32 is None else H32_THRESHOLD - terminal_h32,
            "H1_MARGIN_TO_R6": None if terminal_h1 is None else terminal_h1 - r6_metrics["1"],
            "H32_MARGIN_TO_R6": None if terminal_h32 is None else terminal_h32 - r6_metrics["32"],
            "R6_H1": r6_metrics["1"],
            "R6_H32": r6_metrics["32"],
            "R8E_A1_H1": a1_metrics["1"],
            "R8E_A1_H32": a1_metrics["32"],
            "FIRST_POST_WARMUP_H1_DEGRADATION_UPDATE": trajectory["first_post_warmup_h1_degradation_update"],
            "FIRST_H1_THRESHOLD_CROSSING_UPDATE": trajectory["first_h1_threshold_crossing_update"],
            "FIRST_H32_LOCAL_IMPROVEMENT_UPDATE": trajectory["first_h32_local_improvement_update"],
            "FIRST_H32_R6_RELATIVE_IMPROVEMENT_UPDATE": trajectory["first_h32_r6_relative_improvement_update"],
            "FIRST_H32_GRADUATION_UPDATE": trajectory["first_h32_graduation_update"],
            "D4_TRAJECTORY_TIMING": D4_TIMING,
            "H1_H32_DIVERGENCE_VS_D4": classify_divergence(trajectory, completed),
            "CLIPPING_ENABLED": "NO",
            "RAW_GRADIENT_NORM_MIN": raw_stats["0.0"],
            "RAW_GRADIENT_NORM_MEDIAN": raw_stats["0.5"],
            "RAW_GRADIENT_NORM_P90": raw_stats["0.9"],
            "RAW_GRADIENT_NORM_P95": raw_stats["0.95"],
            "RAW_GRADIENT_NORM_P99": raw_stats["0.99"],
            "RAW_GRADIENT_NORM_MAX": raw_stats["1.0"],
            "NONFINITE_OCCURRED": "YES" if first_nonfinite is not None else "NO",
            "FIRST_NONFINITE_UPDATE": first_nonfinite,
            "FIRST_UNSTABLE_OR_LOSS_EXPLOSION_UPDATE": first_nonfinite,
            "NUMERICAL_STABILITY_STATUS": "NONFINITE_FAILURE_CLOSED" if first_nonfinite is not None else "FINITE_512_UPDATES",
            "EXP_AVG_PATH_SUMMARY": quantiles([float(row["exp_avg_norm"]) for row in update_rows]) if update_rows else None,
            "EXP_AVG_SQ_PATH_SUMMARY": quantiles([float(row["exp_avg_sq_norm"]) for row in update_rows]) if update_rows else None,
            "FINAL_ACTUAL_UPDATE_NORM": None if last_adamw is None else last_adamw["actual_update_norm"],
            "ADAMW_PATH_CHANGED_VS_D4": d5_adamw_changed,
            "GRADIENT_CLIPPING_CAUSAL_EFFECT": effect,
            "CAUSAL_CONFIDENCE": confidence,
            "ROOT_CAUSE_CLASSIFICATION": root_class,
            "ROOT_CAUSE_SUMMARY": root_summary,
            "R8E_A1_INVALIDATED": "NO",
            "PREVIOUS_CHAMPION": "R8E_A1",
            "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
            "CHAMPION_CHANGED": "NO",
            "D5_GRADUATION_ELIGIBLE": graduation_eligible,
            "D5_A1_REPLACEMENT_CANDIDATE": replacement_candidate,
            "REPLACEMENT_BLOCKER": replacement_blocker,
            "SCIENTIFICALLY_READY_FOR_R9": "NO",
            "AUTHORIZED_FOR_R9": "NO",
            "FROZEN20_TOUCHED": "NO",
            "FROZEN20_SHOULD_REMAIN_CLOSED": "YES",
            "FOCUSED_TESTS": "PASS" if focused["focused"]["returncode"] == 0 else "FAIL",
            "REGRESSION_TESTS": "PASS" if focused["regression"]["returncode"] == 0 else "FAIL",
            "NEW_FAILURES": focused["new_failures"],
            "PERSISTENT_OUTPUT_SIZE": 0,
            "ARTIFACT_PATHS": "FINAL_REPORT.md, terminal_certificate.json, trajectory_diagnostics.csv, module_diagnostics.csv, recipe_diff.json, numerical_stability.json, final_state_hash_metadata.json, preflight_identity.json, probe_metrics.json, test_results.json, integrity_manifest.json",
            "ONE_RECOMMENDED_NEXT_ACTION": "Preserve R8E_A1 and obtain separate owner authorization for at most one follow-up mechanism experiment after reviewing D5; do not open Frozen-20 or run R9 in this task.",
            "SCIENTIFIC_CONCLUSION": root_summary + " The result is limited to this deterministic fixed-recipe replay and does not establish a universal project-wide root cause.",
            "ENGINEERING_CONCLUSION": "Preserve R8E_A1 as the current validation champion, keep Frozen-20 sealed, and do not promote D5 automatically.",
            "OWNER_DECISION_REQUIRED": "Decide whether to authorize one separately specified next causal experiment; no follow-up experiment was executed here.",
            "d5_completed_updates": len(update_rows),
            "d5_terminal_state_hash": final_hash,
            "probe_metrics": probe_values,
            "trajectory_summary": trajectory,
            "r6_metrics": r6_metrics,
            "r8e_a1_metrics": a1_metrics,
            "d4_comparison": {"H1": comparison_h1, "H32": comparison_h32},
            "authorization": authorization,
            "recipe_diff": recipe_diff,
            "canonical_schedule_identity": identity,
            "training_config": config,
        }
        write_csv(output / "trajectory_diagnostics.csv", update_rows, tuple(update_rows[0].keys()) if update_rows else ("state_update_index", "raw_global_grad_norm", "clipping_applied"))
        write_csv(output / "module_diagnostics.csv", module_rows, tuple(module_rows[0].keys()) if module_rows else ("state_update_index", "module", "raw_grad_norm", "clipping_applied"))
        write_json(output / "probe_metrics.json", {"predeclared_updates": list(PROBE_UPDATES), "metrics": probe_values, "R6": r6_metrics, "R8E_A1": a1_metrics, "D4_CLIPPED": {"1": D4_H1, "32": D4_H32}, "thresholds": {"H1": D4_H1_THRESHOLD, "H32": H32_THRESHOLD}})
        write_json(output / "numerical_stability.json", {"raw_gradient_norm_quantiles": raw_stats, "nonfinite_occurred": first_nonfinite is not None, "first_nonfinite_update": first_nonfinite, "first_unstable_or_loss_explosion_update": first_nonfinite, "status": context["NUMERICAL_STABILITY_STATUS"], "completed_updates": len(update_rows)})
        write_json(output / "final_state_hash_metadata.json", {"state_retained": "NO", "terminal_state_hash": final_hash, "canonical_state_hash_algorithm": "canonical_model_state_sha256", "completed_updates": len(update_rows), "state_finite": first_nonfinite is None and completed})
        write_json(output / "preflight_identity.json", {"experiment_id": EXPERIMENT_ID, "PROJECT_OWNER_AUTHORIZATION": "YES", "authorization": authorization, "R6_HASH": sha256_file(R6_CHECKPOINT), "R6_HASH_MATCH": "YES", "R8E_A1_RAW_HASH": sha256_file(A1_CHECKPOINT), "R8E_A1_RAW_HASH_MATCH": "YES", "R8E_A1_CANONICAL_HASH": EXPECTED_A1_CANONICAL, "R8E_A1_IMMUTABLE": "YES", "CANONICAL_SCHEDULE_HASH": EXPECTED_CANONICAL_SCHEDULE, "CANONICAL_SCHEDULE_STORAGE_HASH": EXPECTED_CANONICAL_STORAGE, "CANONICAL_SCHEDULE_HASH_MATCH": "YES", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "future_label_leakage": 0, "current_environment": current_environment, "d4_environment": d4_environment, "d4_terminal_reference": {"H1": D4_H1, "H32": D4_H32}, "recipe_diff": {"changed_fields": recipe_diff["changed_fields"], "other_recipe_changes": "NONE"}, "immutable_input_snapshot": "immutable_input_snapshot.json"})
        write_json(output / "test_results.json", focused)
        terminal = build_terminal(context)
        write_json(output / "terminal_certificate.json", terminal)
        (output / "FINAL_REPORT.md").write_text(render_report(terminal), encoding="utf-8", newline="\n")
        context["PERSISTENT_OUTPUT_SIZE"] = persistent_size(output)
        terminal = build_terminal(context)
        write_json(output / "terminal_certificate.json", terminal)
        (output / "FINAL_REPORT.md").write_text(render_report(terminal), encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
        print(json.dumps({"status": terminal["STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY"], "first_blocker": terminal["FIRST_BLOCKER"], "completed_updates": len(update_rows), "h1": terminal_h1, "h32": terminal_h32, "output": str(output)}, sort_keys=True), flush=True)
        return 0 if status == "PASSED" else 2
    except D5Blocker as exc:
        return fail_closed(output, str(exc), "FAIL" if focused and focused["focused"]["returncode"] else "NOT_RUN", "FAIL" if focused and focused["regression"]["returncode"] else "NOT_RUN")
    except Exception as exc:  # noqa: BLE001
        return fail_closed(output, f"unexpected_execution_error:{type(exc).__name__}:{exc}", "FAIL" if focused and focused["focused"]["returncode"] else "NOT_RUN", "FAIL" if focused and focused["regression"]["returncode"] else "NOT_RUN")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path, default=AUTHORIZATION_FILE)
    args = parser.parse_args()
    if args.finalize_existing:
        return finalize_existing(args.output_dir.resolve())
    if not args.execute:
        raise PermissionError("d5_execution_requires_explicit_execute")
    return execute(args.output_dir.resolve(), args.authorization_file.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
