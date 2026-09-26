"""Execute the sealed Stage 3 H13 D9 magnitude-only causal replay.

The CONTROL branch is the canonical D7 trajectory.  The INTERVENTION branch
uses the same forward graph and component gradients, replacing only the
whole-model rollout component by the fixed scalar norm cap relative to H1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402
from src.stage3_h13_r8e import EVALUATION_HORIZONS  # noqa: E402
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256  # noqa: E402
from tools import stage3_h13_post_d7_d8_gru_conflict_causal_intervention_replay as d8  # noqa: E402
from tools import stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay as d7  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D8_D9_H1_ROLLOUT_GRADIENT_MAGNITUDE_DOMINANCE_CAUSAL_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d8_d9_h1_rollout_gradient_magnitude_dominance_causal_design_review_20260817T112646Z"
D7_DIR = ROOT / "outputs/stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay_20260817T063800Z"
D8_DIR = ROOT / "outputs/stage3_h13_post_d7_d8_gru_conflict_causal_intervention_replay_20260817T100717Z"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\8c1e80e7-ed5f-49c3-9cf9-bd982a8c94dc\pasted-text.txt")

EXPECTED_DESIGN_REPORT_SHA256 = "3b49f4b391ccbf03dcaa004da498049116665cc6d0ea423a8d4664fe2f20b948"
EXPECTED_DESIGN_JSON_SHA256 = "5431f7d916d92473687a9a737eb202918cf49ae3c6a7b6b9761dd257f548bd3a"
EXPECTED_DESIGN_MANIFEST_SHA256 = "b1e4a9dbf57aa1f0c93a4f8a6781df52996e7d71444f46469f7a77c26138a800"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_D7_REPORT_SHA256 = "3d5fc7daeabd4718339fd6bca688f42c0a260007547c54547636214dbabc243a"
EXPECTED_D7_MANIFEST_SHA256 = "38bfc88bc6caca455a77bc1e5d7b4b3b07a19ee9792aa0e6ec17f1e5bcadbfbb"
EXPECTED_D8_REPORT_SHA256 = "a4555013d63c33cc63c8a5638207efe6362aa1aaa155d012f1d2805dd27a9b45"
EXPECTED_D8_MANIFEST_SHA256 = "79b84de138f6b10fa412014cc665437d1cdbf7003b28356e9ed79d9346ffe25f"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_EXP_AVG_HASH = "7226e7ae693c1083d6928df6968d67d32d33477c16a8770adcf3050fc9492cdb"
EXPECTED_EXP_AVG_SQ_HASH = "eaf5ebdd6aaddefec25950e80d1a14baeb92a06dddcd11b246546e44c8489d89"
EXPECTED_RNG_HASH = "857d5ddd6c4271019b55a53774ad68c70e435abd38cf430b7f0d57ed047848d8"
EXPECTED_H1_THRESHOLD = 8.59791431235184e-05
EXPECTED_H32_THRESHOLD = 0.01856902565856056
MATERIAL_H1_THRESHOLD = 4.099006815405639e-06
LAMBDA_HIDDEN = 0.005
LAMBDA_H1 = 1.0
HUBER_BETA = 0.001
CLIP_MAX_NORM = 1.0
ADAMW_LR = 5.0e-05
EPSILON = 1.0e-12
REPLAY_RTOL = 5.0e-06
REPLAY_ATOL = 5.0e-09
UPDATES = tuple(range(385, 393))
START_UPDATE = 384
GRU_NAMES = (
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1",
)
ALL_NAMES = GRU_NAMES + ("head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias")
VALIDATION_HORIZONS = tuple(int(x) for x in EVALUATION_HORIZONS)


class StageBlocker(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def finite_float(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise StageBlocker(f"nonfinite_observed_value:{result}")
    return result


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    payload = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}) + b"\0" + tensor.numpy().tobytes()
    return sha256_bytes(payload)


def tensor_map_hash(values: Mapping[str, torch.Tensor]) -> str:
    rows = [{"name": name, "dtype": str(values[name].dtype), "shape": list(values[name].shape), "sha256": tensor_hash(values[name])} for name in values]
    return sha256_bytes(canonical_json(rows))


def state_digest(model: Any) -> str:
    return canonical_model_state_sha256(model.state_dict())


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    return base.optimizer_semantic_hash(optimizer)


def capture_rng() -> dict[str, Any]:
    return base.capture_rng()


def rng_digest(state: Mapping[str, Any]) -> str:
    return base.rng_digest(state)


def norm(values: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for value in values:
        total += float(torch.sum(value.detach().double().square()).cpu())
    return finite_float(math.sqrt(max(total, 0.0)))


def dot(left: Iterable[torch.Tensor], right: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        total += float(torch.sum(a.detach().double() * b.detach().double()).cpu())
    return finite_float(total)


def cosine(dot_value: float, left_norm: float, right_norm: float) -> float | None:
    denominator = left_norm * right_norm
    return None if denominator <= 0.0 else finite_float(dot_value / denominator)


def assert_finite(values: Iterable[torch.Tensor], label: str) -> None:
    for value in values:
        if not bool(torch.all(torch.isfinite(value))):
            raise StageBlocker(f"nonfinite_{label}")


def clone_named_parameters(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters()}


def clone_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> Any:
    return d8.clone_optimizer_state_dict(optimizer)


def clone_named_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    return d8.clone_named_optimizer_state(optimizer)


def materialize_grads(grads: Sequence[torch.Tensor | None], named: Sequence[tuple[str, torch.nn.Parameter]]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if len(grads) != len(named):
        raise StageBlocker("gradient_parameter_count_mismatch")
    values: dict[str, torch.Tensor] = {}
    masks: dict[str, str] = {}
    for (name, parameter), gradient in zip(named, grads):
        if gradient is None:
            values[name] = torch.zeros_like(parameter.detach()).cpu().contiguous()
            masks[name] = "none_materialized_zero"
        else:
            values[name] = gradient.detach().cpu().contiguous().clone()
            masks[name] = "connected"
    assert_finite(values.values(), "component_gradient")
    return values, masks


def capture_parameter_gradients(named: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    values = {
        name: torch.zeros_like(parameter.detach()).cpu().contiguous() if parameter.grad is None else parameter.grad.detach().cpu().contiguous().clone()
        for name, parameter in named
    }
    assert_finite(values.values(), "live_gradient")
    return values


def make_table(named: Sequence[tuple[str, torch.nn.Parameter]]) -> list[dict[str, Any]]:
    if [name for name, _ in named] != list(ALL_NAMES):
        raise StageBlocker("runtime_parameter_order_mismatch")
    table = []
    offset = 0
    for name, parameter in named:
        end = offset + int(parameter.numel())
        table.append({"name": name, "module": "gru" if name.startswith("gru.") else "head", "shape": list(parameter.shape), "numel": int(parameter.numel()), "flat_offset_start": offset, "flat_offset_end": end})
        offset = end
    return table


def flat(values: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    return torch.cat([values[row["name"]].detach().cpu().contiguous().reshape(-1).double() for row in table], dim=0)


def residual(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = flat({row["name"]: left[row["name"]] - right[row["name"]] for row in table}, table)
    l2 = finite_float(torch.linalg.vector_norm(value))
    max_abs = finite_float(torch.max(torch.abs(value))) if value.numel() else 0.0
    scale = max(norm(left[row["name"]] for row in table), norm(right[row["name"]] for row in table), 1.0)
    tolerance = max(1.0e-7, 2.0e-5 * scale)
    passed = l2 <= tolerance and max_abs <= 1.0e-7 + 2.0e-5 * scale
    return {"l2": l2, "max_abs": max_abs, "scale": scale, "tolerance": tolerance, "pass_fail": "PASS" if passed else "FAIL"}


def register_hidden_observer(model: Any) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def hook(_module: Any, inputs: Any, output: Any) -> None:
        sequence, hidden = output
        calls.append({"input_shape": list(inputs[0].shape), "sequence": sequence, "hidden": hidden})

    return calls, model.gru.register_forward_hook(hook)


def build_combined(component: Mapping[str, Mapping[str, torch.Tensor]], rollout_after: Mapping[str, torch.Tensor], named: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: component["H1"][name] + component["hidden_weighted"][name] + rollout_after[name] for name, _ in named}


def intervention_rollout(component: Mapping[str, Mapping[str, torch.Tensor]], table: Sequence[Mapping[str, Any]], branch: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    before = component["rollout"]
    before_flat = flat(before, table)
    h1_flat = flat(component["H1"], table)
    h1_norm = finite_float(torch.linalg.vector_norm(h1_flat))
    before_norm = finite_float(torch.linalg.vector_norm(before_flat))
    ratio = before_norm / max(h1_norm, EPSILON)
    if branch == "CONTROL":
        alpha = 1.0
        after = {name: value.clone() for name, value in before.items()}
        applied = False
    else:
        alpha = min(1.0, h1_norm / max(before_norm, EPSILON))
        after = {name: (value * alpha).contiguous() for name, value in before.items()}
        applied = True
    after_flat = flat(after, table)
    after_norm = finite_float(torch.linalg.vector_norm(after_flat))
    post_ratio = after_norm / max(h1_norm, EPSILON)
    scale_residual = after_flat - alpha * before_flat
    scale_l2 = finite_float(torch.linalg.vector_norm(scale_residual))
    scale_max = finite_float(torch.max(torch.abs(scale_residual))) if scale_residual.numel() else 0.0
    if before_norm > EPSILON and after_norm > EPSILON:
        before_unit = before_flat / before_norm
        after_unit = after_flat / after_norm
        direction_residual = finite_float(torch.linalg.vector_norm(after_unit - before_unit))
        direction_cosine = finite_float(torch.dot(before_unit, after_unit))
    else:
        direction_residual = 0.0 if scale_l2 == 0.0 else float("inf")
        direction_cosine = None
    scale_tolerance = max(1.0e-7, 2.0e-5 * max(before_norm, 1.0))
    direction_pass = direction_cosine is None or (math.isfinite(direction_cosine) and abs(direction_cosine - 1.0) <= 2.0e-5 and direction_residual <= 2.0e-5)
    scale_pass = math.isfinite(scale_l2) and scale_l2 <= scale_tolerance and scale_max <= 1.0e-7 + 2.0e-5 * max(before_norm, 1.0)
    return after, {
        "treatment_applied": "YES" if applied else "NO",
        "epsilon": EPSILON,
        "h1_norm_before": h1_norm,
        "rollout_norm_before": before_norm,
        "rollout_to_h1_ratio_before": ratio,
        "alpha": alpha,
        "rollout_norm_after": after_norm,
        "rollout_to_h1_ratio_after": post_ratio,
        "rollout_cosine_before_vs_after": direction_cosine,
        "rollout_normalized_direction_residual": direction_residual,
        "rollout_scaling_residual_l2": scale_l2,
        "rollout_scaling_residual_max_abs": scale_max,
        "rollout_scaling_residual_tolerance": scale_tolerance,
        "rollout_direction_preservation_pass": "PASS" if direction_pass else "FAIL",
        "rollout_scaling_equation_pass": "PASS" if scale_pass else "FAIL",
        "fixed_max_rollout_to_h1_ratio": 1.0,
    }


def direct_component_checks(component: Mapping[str, Mapping[str, torch.Tensor]], rollout_after: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]], named: Sequence[tuple[str, torch.nn.Parameter]], alpha: float) -> dict[str, Any]:
    h1_before = component["H1"]
    hidden_before = component["hidden_weighted"]
    h1_copy = {name: value.clone() for name, value in h1_before.items()}
    hidden_copy = {name: value.clone() for name, value in hidden_before.items()}
    h1_equal = residual(h1_before, h1_copy, table)
    hidden_equal = residual(hidden_before, hidden_copy, table)
    expected_total = {name: h1_before[name] + hidden_before[name] + rollout_after[name] for name, _ in named}
    non_rollout_before = {name: h1_before[name] + hidden_before[name] for name, _ in named}
    non_rollout_after = {name: expected_total[name] - rollout_after[name] for name, _ in named}
    non_rollout_equal = residual(non_rollout_before, non_rollout_after, table)
    return {
        "H1_direct_component_copy_equality": h1_equal,
        "hidden_direct_component_copy_equality": hidden_equal,
        "direct_non_rollout_gradient_equality": non_rollout_equal,
        "H1_directly_unchanged": h1_equal["pass_fail"] == "PASS",
        "hidden_directly_unchanged": hidden_equal["pass_fail"] == "PASS",
        "direct_non_rollout_unchanged": non_rollout_equal["pass_fail"] == "PASS",
        "alpha_used": alpha,
    }


def optimizer_metrics(model: Any, optimizer: torch.optim.Optimizer, before: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    return d8.adamw_metrics(model, optimizer, before, clipped, after, table)


def validation_snapshot(model: Any, validation: Any, channels: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    return d8.validation_snapshot(model, validation, channels)


def update_artifacts(output: Path, branch: str, update: int, before_model: Mapping[str, torch.Tensor], after_model: Mapping[str, torch.Tensor], before_optimizer: Any, after_optimizer: Any, before_named_optimizer: Mapping[str, Any], after_named_optimizer: Mapping[str, Any], component: Mapping[str, Mapping[str, torch.Tensor]], canonical_sum: Mapping[str, torch.Tensor], total_autograd: Mapping[str, torch.Tensor], raw: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], rollout_after: Mapping[str, torch.Tensor], adamw_tensors: Mapping[str, Any], delta: Mapping[str, torch.Tensor], payload: Mapping[str, Any]) -> None:
    d8.update_artifacts(output, branch, update, before_model, after_model, before_optimizer, after_optimizer, before_named_optimizer, after_named_optimizer, component, canonical_sum, total_autograd, raw, clipped, rollout_after, adamw_tensors, delta, payload)
    root = output / "updates" / branch / f"update_{update:05d}"
    d8.serialize_state(root / "gradients" / "g_rollout_after.pt", rollout_after)
    write_json(root / "intervention_checks.json", payload["manipulation_checks"])


def authoritative_d7_rows() -> dict[int, dict[str, str]]:
    path = D7_DIR / "update_metrics.csv"
    if not path.is_file():
        raise StageBlocker("authoritative_d7_update_metrics_missing")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {int(row["update"]): row for row in csv.DictReader(stream)}


def compare_control_to_d7(rows: Sequence[Mapping[str, Any]], authority_rows: Mapping[int, Mapping[str, str]]) -> dict[str, Any]:
    return d8.compare_control_to_d7(rows, authority_rows)


def verify_authorization() -> dict[str, Any]:
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("current_execution_authorization_missing")
    text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required = [
        "Execute the already-reviewed and now explicitly authorized stage",
        EXPERIMENT_ID,
        "This message constitutes explicit project-owner authorization",
        "Execute exactly two branches",
        "CONTROL",
        "INTERVENTION",
        "385,386,387,388,389,390,391,392",
        "Do not execute update 393",
        "FROZEN20_OPENED: NO",
        "R9_EXECUTED: NO",
        "R8E_A1_MODIFIED: NO",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise StageBlocker(f"current_authorization_scope_not_proven:{missing}")
    return {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"}


def verified_manifest_files(directory: Path, manifest: Mapping[str, Any], label: str) -> dict[str, str]:
    entries = manifest.get("files") or manifest.get("artifact_sha256") or {}
    if not isinstance(entries, Mapping):
        raise StageBlocker(f"{label}_manifest_entries_missing")
    verified = {}
    for name, expected in entries.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise StageBlocker(f"{label}_artifact_hash_mismatch:{name}")
        verified[str(name)] = str(expected)
    return verified


def verify_sealed_inputs() -> dict[str, Any]:
    if not DESIGN_DIR.is_dir():
        raise StageBlocker("sealed_design_directory_missing")
    manifest_path = DESIGN_DIR / "SHA256_MANIFEST.json"
    if sha256_file(manifest_path) != EXPECTED_DESIGN_MANIFEST_SHA256:
        raise StageBlocker("sealed_design_manifest_hash_mismatch")
    design_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_design_files = {"FINAL_REPORT.md": EXPECTED_DESIGN_REPORT_SHA256, "experiment_design.json": EXPECTED_DESIGN_JSON_SHA256}
    if design_manifest.get("files") != expected_design_files:
        raise StageBlocker("sealed_design_manifest_contents_mismatch")
    design_hashes = {}
    for name, expected in expected_design_files.items():
        path = DESIGN_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            raise StageBlocker(f"sealed_design_hash_mismatch:{name}")
        design_hashes[name] = expected
    design = json.loads((DESIGN_DIR / "experiment_design.json").read_text(encoding="utf-8"))
    contract = design["classification"]
    future = design["future_replay_contract"]
    if design.get("review_status") != "PASSED" or design.get("training_executed") is not False or design.get("backward_pass_executed") is not False or design.get("optimizer_step_executed") is not False or design.get("d9_executed") is not False:
        raise StageBlocker("sealed_design_status_invalid")
    if future.get("start_update") != START_UPDATE or future.get("execution_updates") != list(UPDATES):
        raise StageBlocker("sealed_design_window_invalid")
    if future.get("branches_exactly") != ["CONTROL", "INTERVENTION"]:
        raise StageBlocker("sealed_design_branch_contract_invalid")
    if future.get("intervention", {}).get("fixed_max_ratio") != 1.0 or future.get("intervention", {}).get("application_scope") != "whole_model component vector in canonical parameter order":
        raise StageBlocker("sealed_design_intervention_contract_invalid")
    if future.get("parameter_order") != list(ALL_NAMES):
        raise StageBlocker("sealed_design_parameter_order_invalid")
    if future.get("clipping", {}).get("implementation") != "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)":
        raise StageBlocker("sealed_design_clipping_contract_invalid")
    if future.get("adamw", {}).get("scheduler") != "ABSENT_BY_DESIGN" or future.get("adamw", {}).get("learning_rate") != ADAMW_LR:
        raise StageBlocker("sealed_design_adamw_contract_invalid")
    if design.get("endpoints", {}).get("primary_H1") != "update-392 validation H1 RMSE; lower is better" or design.get("endpoints", {}).get("H1_guardrail") != EXPECTED_H1_THRESHOLD or design.get("endpoints", {}).get("H32_guardrail") != EXPECTED_H32_THRESHOLD or design.get("endpoints", {}).get("material_improvement_threshold") != MATERIAL_H1_THRESHOLD:
        raise StageBlocker("sealed_design_endpoint_contract_invalid")
    if design.get("classification", {}).get("priority", [None])[0] != "INVALID":
        raise StageBlocker("sealed_design_classification_priority_invalid")
    if not BUNDLE_PATH.is_file() or sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("certified_start_bundle_hash_mismatch")

    d7_manifest_path = D7_DIR / "SHA256_MANIFEST.json"
    d7_report_path = D7_DIR / "FINAL_REPORT.md"
    d8_manifest_path = D8_DIR / "SHA256_MANIFEST.json"
    d8_report_path = D8_DIR / "FINAL_REPORT.md"
    if sha256_file(d7_manifest_path) != EXPECTED_D7_MANIFEST_SHA256 or sha256_file(d7_report_path) != EXPECTED_D7_REPORT_SHA256:
        raise StageBlocker("authoritative_d7_bundle_hash_mismatch")
    if sha256_file(d8_manifest_path) != EXPECTED_D8_MANIFEST_SHA256 or sha256_file(d8_report_path) != EXPECTED_D8_REPORT_SHA256:
        raise StageBlocker("authoritative_d8_bundle_hash_mismatch")
    d7_manifest = json.loads(d7_manifest_path.read_text(encoding="utf-8"))
    d8_manifest = json.loads(d8_manifest_path.read_text(encoding="utf-8"))
    d7_verified = verified_manifest_files(D7_DIR, d7_manifest, "authoritative_d7")
    d8_verified = verified_manifest_files(D8_DIR, d8_manifest, "authoritative_d8")
    d8_classification = json.loads((D8_DIR / "causal_classification.json").read_text(encoding="utf-8"))
    if d8_classification.get("classification") != "GRU_CONFLICT_NOT_CAUSAL":
        raise StageBlocker("authoritative_d8_classification_mismatch")
    if base.EXPECTED_SOURCE_COMMIT != EXPECTED_SOURCE_REVISION:
        raise StageBlocker("base_source_revision_constant_mismatch")
    return {
        "design_directory": str(DESIGN_DIR.resolve()), "design_hashes": design_hashes, "design": design,
        "bundle_path": str(BUNDLE_PATH.resolve()), "bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "d7_directory": str(D7_DIR.resolve()), "d7_final_report_sha256": EXPECTED_D7_REPORT_SHA256, "d7_manifest_sha256": EXPECTED_D7_MANIFEST_SHA256, "d7_verified_files": d7_verified,
        "d8_directory": str(D8_DIR.resolve()), "d8_final_report_sha256": EXPECTED_D8_REPORT_SHA256, "d8_manifest_sha256": EXPECTED_D8_MANIFEST_SHA256, "d8_verified_files": d8_verified, "d8_classification": d8_classification,
        "source_revision": EXPECTED_SOURCE_REVISION,
    }


def verify_certified_bundle(bundle: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    bundle_identity = d7.validate_bundle(bundle, authority)
    state_hashes = bundle.get("state_hashes", {})
    expected = {
        "model_semantic_hash": EXPECTED_MODEL_HASH,
        "optimizer_semantic_hash": EXPECTED_OPTIMIZER_HASH,
        "exp_avg_semantic_hash": EXPECTED_EXP_AVG_HASH,
        "exp_avg_sq_semantic_hash": EXPECTED_EXP_AVG_SQ_HASH,
        "rng_state_sha256": EXPECTED_RNG_HASH,
    }
    for key, value in expected.items():
        if state_hashes.get(key) != value:
            raise StageBlocker(f"certified_state_hash_mismatch:{key}")
    if bundle.get("training_position", {}).get("completed_optimizer_steps") != START_UPDATE:
        raise StageBlocker("certified_completed_steps_mismatch")
    return {"bundle_identity": bundle_identity, "state_hashes": state_hashes, "expected_state_hashes": expected}


def classify(control_h1: float, intervention_h1: float, control_h32: float, intervention_h32: float, integrity: bool, intervention_finite: bool = True) -> tuple[str, dict[str, bool]]:
    effect = control_h1 - intervention_h1
    conditions = {
        "integrity_pass": integrity,
        "control_finite": math.isfinite(control_h1) and math.isfinite(control_h32),
        "intervention_finite": intervention_finite and math.isfinite(intervention_h1) and math.isfinite(intervention_h32),
        "intervention_h1_guardrail_pass": intervention_h1 <= EXPECTED_H1_THRESHOLD if math.isfinite(intervention_h1) else False,
        "intervention_h32_guardrail_pass": intervention_h32 <= EXPECTED_H32_THRESHOLD if math.isfinite(intervention_h32) else False,
        "h1_improvement_material": effect >= MATERIAL_H1_THRESHOLD if math.isfinite(effect) else False,
        "intervention_h1_harmful": intervention_h1 - control_h1 >= MATERIAL_H1_THRESHOLD if math.isfinite(intervention_h1) else False,
        "intervention_h32_harmful": intervention_h32 > EXPECTED_H32_THRESHOLD if math.isfinite(intervention_h32) else False,
    }
    if not integrity:
        return "INVALID", conditions
    if conditions["control_finite"] and not conditions["intervention_finite"]:
        return "MAGNITUDE_INTERVENTION_DESTABILIZING", conditions
    if conditions["intervention_h1_harmful"] or conditions["intervention_h32_harmful"]:
        return "MAGNITUDE_INTERVENTION_HARMFUL", conditions
    if conditions["intervention_h32_guardrail_pass"] and conditions["h1_improvement_material"] and conditions["intervention_h1_guardrail_pass"]:
        return "MAGNITUDE_DOMINANCE_CAUSAL", conditions
    if conditions["intervention_h32_guardrail_pass"] and conditions["h1_improvement_material"] and not conditions["intervention_h1_guardrail_pass"]:
        return "MAGNITUDE_DOMINANCE_PARTIALLY_CAUSAL", conditions
    if conditions["intervention_h32_guardrail_pass"] and math.isfinite(effect) and effect < MATERIAL_H1_THRESHOLD:
        return "MAGNITUDE_DOMINANCE_NOT_CAUSAL", conditions
    return "INCONCLUSIVE", conditions


def run_update(branch: str, model: Any, optimizer: torch.optim.Optimizer, pre: Mapping[str, Any], authority: Mapping[str, Any], output: Path, previous_row: Mapping[str, Any] | None, zero_based_index: int) -> dict[str, Any]:
    update = zero_based_index + 1
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    table = make_table(named)
    rng_before = capture_rng()
    before_model = clone_named_parameters(model)
    before_optimizer = clone_optimizer_state_dict(optimizer)
    before_named_optimizer = clone_named_optimizer_state(optimizer)
    before_model_hash = state_digest(model)
    before_optimizer_hash = optimizer_digest(optimizer)
    batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_based_index])
    batch_hash = base.batch_digest(authority["schedule_manifest"], zero_based_index)
    expected_batch = authority["schedule_manifest"]["ordered_batches"][zero_based_index].get("batch_window_sha256")
    if expected_batch is not None and batch_hash != expected_batch:
        raise StageBlocker(f"batch_hash_generation_mismatch_update_{update}")
    calls, handle = register_hidden_observer(model)
    try:
        optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for _, parameter in named):
            raise StageBlocker(f"zero_grad_failed_update_{update}")
        rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=32, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
        if rollout.get("free_branch_reads_target_positions") is not False:
            raise StageBlocker(f"future_label_leakage_update_{update}")
        losses = base.training_losses(rollout["predictions"], batch["target_positions"], rollout["hidden_loss"], lambda_hidden=LAMBDA_HIDDEN, lambda_h1=LAMBDA_H1, beta=HUBER_BETA)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise StageBlocker(f"nonfinite_loss_update_{update}")
        scalars = {"rollout": losses["rollout"], "hidden_weighted": LAMBDA_HIDDEN * losses["hidden"], "H1": LAMBDA_H1 * losses["h1"]}
        component: dict[str, dict[str, torch.Tensor]] = {}
        masks: dict[str, dict[str, str]] = {}
        for component_name, scalar in scalars.items():
            grads = torch.autograd.grad(scalar, [parameter for _, parameter in named], retain_graph=True, create_graph=False, allow_unused=True)
            if any(parameter.grad is not None for _, parameter in named):
                raise StageBlocker(f"diagnostic_grad_mutation_{component_name}_update_{update}")
            component[component_name], masks[component_name] = materialize_grads(grads, named)
        total_autograd_raw = torch.autograd.grad(losses["total"], [parameter for _, parameter in named], retain_graph=True, create_graph=False, allow_unused=True)
        if any(parameter.grad is not None for _, parameter in named):
            raise StageBlocker(f"diagnostic_total_grad_mutation_update_{update}")
        total_autograd, total_masks = materialize_grads(total_autograd_raw, named)
        canonical_sum = {name: component["H1"][name] + component["rollout"][name] + component["hidden_weighted"][name] for name, _ in named}
        canonical_additivity = residual(canonical_sum, total_autograd, table)
        if canonical_additivity["pass_fail"] != "PASS":
            raise StageBlocker(f"canonical_component_additivity_failed_update_{update}")
        rollout_after, manipulation = intervention_rollout(component, table, branch)
        if manipulation["rollout_direction_preservation_pass"] != "PASS" or manipulation["rollout_scaling_equation_pass"] != "PASS":
            raise StageBlocker(f"rollout_magnitude_manipulation_failed_update_{update}")
        direct_checks = direct_component_checks(component, rollout_after, table, named, manipulation["alpha"])
        if not direct_checks["H1_directly_unchanged"] or not direct_checks["hidden_directly_unchanged"] or not direct_checks["direct_non_rollout_unchanged"]:
            raise StageBlocker(f"direct_non_rollout_equality_failed_update_{update}")
        if branch == "CONTROL":
            losses["total"].backward()
            if not base.gradients_are_finite(model):
                raise StageBlocker(f"nonfinite_control_gradient_update_{update}")
        else:
            combined = build_combined(component, rollout_after, named)
            for name, parameter in named:
                if all(masks[key][name] == "none_materialized_zero" for key in ("H1", "rollout", "hidden_weighted")):
                    parameter.grad = None
                else:
                    parameter.grad = combined[name].to(device=parameter.device, dtype=parameter.dtype).clone()
        raw = capture_parameter_gradients(named)
        raw_vs_autograd = residual(raw, total_autograd, table)
        direct_combined = canonical_sum if branch == "CONTROL" else build_combined(component, rollout_after, named)
        raw_vs_direct = residual(raw, direct_combined, table)
        if branch == "CONTROL" and raw_vs_autograd["pass_fail"] != "PASS":
            raise StageBlocker(f"control_raw_gradient_mismatch_update_{update}")
        if raw_vs_direct["pass_fail"] != "PASS":
            raise StageBlocker(f"recombined_gradient_mismatch_update_{update}")
        raw_norm = norm(raw.values())
        returned_preclip = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM))
        if not math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise StageBlocker(f"clip_returned_norm_mismatch_update_{update}")
        clipped = capture_parameter_gradients(named)
        clipped_norm = norm(clipped.values())
        coefficient = min(1.0, CLIP_MAX_NORM / returned_preclip) if returned_preclip > 0.0 else 1.0
        optimizer.step()
        after_model = clone_named_parameters(model)
        after_optimizer = clone_optimizer_state_dict(optimizer)
        after_named_optimizer = clone_named_optimizer_state(optimizer)
        after_model_hash = state_digest(model)
        after_optimizer_hash = optimizer_digest(optimizer)
        if not d7.state_is_finite(model, optimizer):
            raise StageBlocker(f"nonfinite_state_after_update_{update}")
    finally:
        handle.remove()
    validation, validation_meta = validation_snapshot(model, pre["validation"], pre["stats"]["channels"])
    delta = {name: after_model[name] - before_model[name] for name, _ in named}
    raw_to_clipped = {name: raw[name] - clipped[name] for name, _ in named}
    adamw, adamw_tensors = optimizer_metrics(model, optimizer, before_model, clipped, after_model, table)
    whole_component_metrics = d8.add_total_ratios(d8.pair_metrics(component, table, "whole_model"), raw_norm)
    post_component = {"H1": component["H1"], "rollout": rollout_after, "hidden_weighted": component["hidden_weighted"]}
    post_component_metrics = d8.add_total_ratios(d8.pair_metrics(post_component, table, "whole_model"), raw_norm)
    rng_after_training = capture_rng()
    row: dict[str, Any] = {
        "branch": branch, "update": update, "zero_based_schedule_index": zero_based_index, "schedule_index": zero_based_index, "batch_window_sha256": batch_hash,
        "active_horizon": 32, "active_training_phase": "phase_4_H32", "canonical_state_hash_before": before_model_hash, "canonical_state_hash_after": after_model_hash,
        "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash, "rng_before_digest": rng_digest(rng_before), "rng_after_training_digest": rng_digest(rng_after_training),
        "rng_before_validation_digest": validation_meta["rng_before_digest"], "rng_after_validation_digest": validation_meta["rng_after_digest"],
        "schedule_identity": {"logical_sha256": authority["schedule_hashes"]["logical_sha256"], "storage_sha256": authority["schedule_hashes"]["storage_sha256"]},
        "rollout_loss": finite_float(losses["rollout"]), "hidden_loss_unweighted": finite_float(losses["hidden"]), "hidden_loss_weighted": finite_float(scalars["hidden_weighted"]), "h1_loss_unweighted": finite_float(losses["h1"]), "h1_loss_weighted": finite_float(scalars["H1"]), "total_loss": finite_float(losses["total"]),
        "objective_weights": {"lambda_rollout": 1.0, "lambda_hidden": LAMBDA_HIDDEN, "lambda_h1": LAMBDA_H1, "huber_beta": HUBER_BETA},
        "validation": validation, "eval_h1": validation["1"], "eval_h32": validation["32"], **{f"eval_h{key}": value for key, value in validation.items() if key not in ("1", "32")}, "h1_pass": "PASS" if validation["1"] <= EXPECTED_H1_THRESHOLD else "FAIL", "h32_pass": "PASS" if validation["32"] <= EXPECTED_H32_THRESHOLD else "FAIL", "validation_meta": validation_meta,
        "component_gradient_hashes": {key: tensor_map_hash(value) for key, value in component.items()}, "canonical_sum_hash": tensor_map_hash(canonical_sum), "total_autograd_hash": tensor_map_hash(total_autograd), "rollout_after_hash": tensor_map_hash(rollout_after),
        "manipulation_checks": {**manipulation, **direct_checks}, "H1_gradient_unchanged": "YES" if direct_checks["H1_directly_unchanged"] else "NO", "hidden_gradient_unchanged": "YES" if direct_checks["hidden_directly_unchanged"] else "NO", "direct_non_rollout_gradient_equality_pre_global_clip": "YES" if direct_checks["direct_non_rollout_unchanged"] else "NO",
        "raw_total_gradient_norm": raw_norm, "returned_pre_clipping_total_norm": returned_preclip, "clipping_threshold": CLIP_MAX_NORM, "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO", "effective_global_scaling_factor": coefficient, "clipped_total_gradient_norm": clipped_norm,
        "canonical_component_additivity": canonical_additivity, "raw_vs_total_autograd": raw_vs_autograd, "raw_vs_direct_recombined": raw_vs_direct, "raw_gradient_hash": tensor_map_hash(raw), "clipped_gradient_hash": tensor_map_hash(clipped), "parameter_update_digest": d8.d6.named_tensor_digest(delta), "parameter_update_norm": adamw["aggregate"]["whole_model"]["actual_delta_norm"], "raw_gradient_digest": d8.d6.named_tensor_digest(raw), "clipped_gradient_digest": d8.d6.named_tensor_digest(clipped), "raw_to_clipped_direction_cosine": cosine(dot(raw.values(), clipped.values()), raw_norm, clipped_norm), "raw_to_clipped_residual_norm": norm(raw_to_clipped.values()),
        "full_model_gradient_metrics": whole_component_metrics, "post_treatment_gradient_metrics": post_component_metrics, "adamw": adamw, "finite_status": "FINITE", "future_label_leakage": 0, "model_mode_before": True, "model_mode_after_training": True, "optimizer_step_counter_after": adamw["step_after_by_parameter"], "optimizer_semantics_unchanged": "YES", "clipping_implementation_unchanged": "YES",
    }
    if previous_row is None:
        row["transition_change_from_prior_update"] = {"available": False}
    else:
        row["transition_change_from_prior_update"] = {"previous_update": previous_row["update"], "h1": validation["1"] - previous_row["eval_h1"], "h32": validation["32"] - previous_row["eval_h32"]}
    update_artifacts(output, branch, update, before_model, after_model, before_optimizer, after_optimizer, before_named_optimizer, after_named_optimizer, component, canonical_sum, total_autograd, raw, clipped, rollout_after, adamw_tensors, delta, row)
    return row


def write_manifest(output: Path) -> dict[str, str]:
    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"SHA256_MANIFEST.json", "FINAL_REPORT.md"}):
        files[str(path.relative_to(output)).replace("\\", "/")] = sha256_file(path)
    write_json(output / "SHA256_MANIFEST.json", {"schema_version": "stage3_h13_post_d8_d9_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "final_report_excluded": True, "files": files})
    return files


def render_report(summary: Mapping[str, Any], artifact_hashes: Mapping[str, str]) -> str:
    e = summary["endpoints"]
    integrity = summary["integrity"]
    lines = [
        f"STAGE_3_H13_POST_D8_D9_H1_ROLLOUT_GRADIENT_MAGNITUDE_DOMINANCE_CAUSAL_REPLAY: {summary['status']}",
        f"FIRST_BLOCKER: {summary['first_blocker']}",
        f"CAUSAL_CLASSIFICATION: {summary['causal_classification']}",
        f"ONE_SENTENCE_VERDICT: {summary['one_sentence_verdict']}",
        "",
        "## 1. PROTOCOL_INTEGRITY", "",
        f"STATUS: {integrity['STATUS']}",
        f"SEALED_INPUTS_MATCH: {integrity['SEALED_INPUTS_MATCH']}",
        f"SOURCE_REVISION_MATCH: {integrity['SOURCE_REVISION_MATCH']}",
        f"FUTURE_LABEL_LEAKAGE: {integrity['FUTURE_LABEL_LEAKAGE']}",
        f"UPDATES_EXECUTED: {integrity['UPDATES_EXECUTED']}",
        "",
        "## 2. CONTROL_REPRODUCTION", "",
        f"CONTROL_REPRODUCED_D7: {summary['control_reproduction']['CONTROL_REPRODUCES_CANONICAL_D7']}",
        f"FIRST_MISMATCH: {summary['control_reproduction']['first_mismatch']}",
        "",
        "## 3. BRANCH_START_IDENTITY", "",
        f"INDEPENDENT_LOADS: {summary['branch_identity']['independent_loads']}",
        f"MODEL_SEMANTIC_HASH_EQUAL: {summary['branch_identity']['model_semantic_hash_equal']}",
        f"OPTIMIZER_SEMANTIC_HASH_EQUAL: {summary['branch_identity']['optimizer_semantic_hash_equal']}",
        f"RNG_IDENTITY_EQUAL: {summary['branch_identity']['control_rng_digest'] == summary['branch_identity']['intervention_rng_digest']}",
        "",
        "## 4. PER_UPDATE_MANIPULATION_CHECKS", "",
        "| update | branch | rollout/H1 before | alpha | rollout/H1 after | cosine | direction residual | scaling residual | H1 unchanged | hidden unchanged |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in summary["control_rows"] + summary["intervention_rows"]:
        m = row["manipulation_checks"]
        lines.append(f"| {row['update']} | {row['branch']} | {m['rollout_to_h1_ratio_before']:.17g} | {m['alpha']:.17g} | {m['rollout_to_h1_ratio_after']:.17g} | {m['rollout_cosine_before_vs_after']} | {m['rollout_normalized_direction_residual']:.6g} | {m['rollout_scaling_residual_l2']:.6g} | {row['H1_gradient_unchanged']} | {row['hidden_gradient_unchanged']} |")
    lines += [
        "",
        "## 5. ROLLOUT_MAGNITUDE_EFFECT", "",
        "The intervention is the fixed whole-model scalar cap alpha=min(1, ||g_H1||/max(||g_rollout||, 1e-12)); no direction surgery or per-parameter rescaling was used.",
        "",
        "## 6. DIRECTION_PRESERVATION", "",
        f"ROLLOUT_DIRECTION_PRESERVED_ALL_UPDATES: {integrity['ROLLOUT_DIRECTION_PRESERVED_ALL_UPDATES']}",
        f"ROLLOUT_SCALING_EQUATION_PASS_ALL_UPDATES: {integrity['ROLLOUT_SCALING_EQUATION_PASS_ALL_UPDATES']}",
        "",
        "## 7. GLOBAL_CLIPPING_MEDIATION", "",
        f"CLIPPING_SEMANTICS_UNCHANGED: {integrity['CLIPPING_SEMANTICS_UNCHANGED']}",
        "Per-update raw norms, clipping coefficients, activation flags, clipped gradients, and raw/post direction residuals are retained in update_metrics.json and update artifacts.",
        "",
        "## 8. ADAMW_MEDIATION", "",
        f"ADAMW_SEMANTICS_UNCHANGED: {integrity['ADAMW_SEMANTICS_UNCHANGED']}",
        "Per-update moments, step counters, adaptive updates, weight-decay updates, actual deltas, decomposition residuals, and state hashes are retained.",
        "",
        "## 9. UPDATE_392_PRIMARY_ENDPOINTS", "",
        f"CONTROL_H1_UPDATE_392: {e['CONTROL_H1_UPDATE_392']:.17g}",
        f"INTERVENTION_H1_UPDATE_392: {e['INTERVENTION_H1_UPDATE_392']:.17g}",
        f"H1_CAUSAL_EFFECT_CONTROL_MINUS_INTERVENTION: {e['H1_CAUSAL_EFFECT_CONTROL_MINUS_INTERVENTION']:.17g}",
        f"H1_MATERIAL_THRESHOLD: {MATERIAL_H1_THRESHOLD:.17g}",
        f"INTERVENTION_H1_GUARDRAIL_PASS: {e['INTERVENTION_H1_GUARDRAIL_PASS']}",
        f"CONTROL_H32_UPDATE_392: {e['CONTROL_H32_UPDATE_392']:.17g}",
        f"INTERVENTION_H32_UPDATE_392: {e['INTERVENTION_H32_UPDATE_392']:.17g}",
        f"INTERVENTION_H32_GUARDRAIL_PASS: {e['INTERVENTION_H32_GUARDRAIL_PASS']}",
        "",
        "## 10. SECONDARY_HORIZONS", "",
        "| horizon | CONTROL | INTERVENTION | intervention-control |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in VALIDATION_HORIZONS:
        cv = summary["control_final_validation"][str(horizon)]
        iv = summary["intervention_final_validation"][str(horizon)]
        lines.append(f"| H{horizon} | {cv:.17g} | {iv:.17g} | {iv-cv:.17g} |")
    lines += [
        "",
        "## 11. CAUSAL_CLASSIFICATION_APPLICATION", "",
        f"CAUSAL_CLASSIFICATION: {summary['causal_classification']}",
        json.dumps(summary["classification_conditions"], sort_keys=True),
        "",
        "## 12. SCIENTIFIC_INTERPRETATION", "",
        summary["one_sentence_verdict"],
        "The conclusion is based only on the matched CONTROL-versus-INTERVENTION replay and the sealed endpoint rule, not on the observational ratios alone.",
        "",
        "## 13. ARTIFACT_PATHS_AND_SHA256", "",
        f"OUTPUT_DIRECTORY: {summary['output_directory']}",
    ]
    for name, digest in artifact_hashes.items():
        lines.append(f"{name}: {digest}")
    lines += [
        "",
        "## 14. FINAL_PROHIBITED_ACCESS_CONFIRMATION", "",
        "FROZEN20_OPENED: NO",
        "FROZEN20_USED: NO",
        "R9_EXECUTED: NO",
        "R8E_A1_MODIFIED: NO",
        "HISTORICAL_ARTIFACTS_MODIFIED: NO",
        "NEXT_EXPERIMENT_EXECUTED: NO",
        "",
        "CONTROL_REPRODUCED_D7: " + summary["control_reproduction"]["CONTROL_REPRODUCES_CANONICAL_D7"],
        "H1_DIRECTLY_UNCHANGED_PRECLIP_ALL_UPDATES: " + integrity["H1_DIRECTLY_UNCHANGED_PRECLIP_ALL_UPDATES"],
        "HIDDEN_DIRECTLY_UNCHANGED_PRECLIP_ALL_UPDATES: " + integrity["HIDDEN_DIRECTLY_UNCHANGED_PRECLIP_ALL_UPDATES"],
    ]
    return "\n".join(lines) + "\n"


def fail_closed(output: Path, blocker: str, authorization: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "BLOCKED", "first_blocker": blocker, "causal_classification": "INVALID",
        "one_sentence_verdict": "Execution failed closed before a valid D9 causal interpretation was established.",
        "integrity": {"STATUS": "FAIL", "FAIL_CLOSED": "YES", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HISTORICAL_ARTIFACTS_MODIFIED": "NO"},
        "authorization": authorization,
    }
    write_json(output / "terminal_certificate.json", summary)
    (output / "FINAL_REPORT.md").write_text("\n".join([
        "STAGE_3_H13_POST_D8_D9_H1_ROLLOUT_GRADIENT_MAGNITUDE_DOMINANCE_CAUSAL_REPLAY: BLOCKED",
        f"FIRST_BLOCKER: {blocker}",
        "CAUSAL_CLASSIFICATION: INVALID",
        "ONE_SENTENCE_VERDICT: Execution failed closed before a valid D9 causal interpretation was established.",
        "",
        "FROZEN20_OPENED: NO", "FROZEN20_USED: NO", "R9_EXECUTED: NO", "R8E_A1_MODIFIED: NO", "HISTORICAL_ARTIFACTS_MODIFIED: NO", "NEXT_EXPERIMENT_EXECUTED: NO", "",
    ]) + "\n", encoding="utf-8", newline="\n")
    write_manifest(output)
    print(json.dumps({"status": "BLOCKED", "output": str(output.resolve()), "blocker": blocker}, sort_keys=True), flush=True)
    return 2


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authorization = None
    a1_before = None
    try:
        authorization = verify_authorization()
        sealed = verify_sealed_inputs()
        pre = base.preflight(True)
        authority = base.verify_authority(pre)
        environment = authority["current_environment"]
        if environment.get("source_revision") != EXPECTED_SOURCE_REVISION or environment.get("device") != "cpu" or environment.get("dtype") != "float32" or environment.get("deterministic_algorithms") is not True:
            raise StageBlocker("execution_environment_or_source_identity_mismatch")
        a1_before = sha256_file(base.A1_CHECKPOINT)
        if a1_before != base.EXPECTED_A1_HASH:
            raise StageBlocker("r8e_a1_immutability_gate_failed_before_execution")
        bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
        certified_bundle = verify_certified_bundle(bundle, authority)
        base.set_deterministic(base.SEED)
        control_model, control_optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        intervention_model, intervention_optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        control_loaded = d7.verify_loaded_identity(bundle, control_model, control_optimizer)
        intervention_loaded = d7.verify_loaded_identity(bundle, intervention_model, intervention_optimizer)
        control_table = make_table([(name, parameter) for name, parameter in control_model.named_parameters() if parameter.requires_grad])
        intervention_table = make_table([(name, parameter) for name, parameter in intervention_model.named_parameters() if parameter.requires_grad])
        if control_table != intervention_table:
            raise StageBlocker("branch_parameter_order_identity_mismatch")
        if state_digest(control_model) != EXPECTED_MODEL_HASH or state_digest(intervention_model) != EXPECTED_MODEL_HASH or optimizer_digest(control_optimizer) != EXPECTED_OPTIMIZER_HASH or optimizer_digest(intervention_optimizer) != EXPECTED_OPTIMIZER_HASH:
            raise StageBlocker("independent_branch_start_identity_mismatch")
        base.restore_rng(bundle["rng_state"])
        control_rng_start = rng_digest(capture_rng())
        base.restore_rng(bundle["rng_state"])
        intervention_rng_start = rng_digest(capture_rng())
        if control_rng_start != EXPECTED_RNG_HASH or intervention_rng_start != control_rng_start:
            raise StageBlocker("independent_branch_rng_identity_mismatch")
        branch_identity = {
            "model_semantic_hash_equal": state_digest(control_model) == state_digest(intervention_model), "optimizer_semantic_hash_equal": optimizer_digest(control_optimizer) == optimizer_digest(intervention_optimizer),
            "control_start_model_hash": state_digest(control_model), "intervention_start_model_hash": state_digest(intervention_model), "control_start_optimizer_hash": optimizer_digest(control_optimizer), "intervention_start_optimizer_hash": optimizer_digest(intervention_optimizer),
            "control_rng_digest": control_rng_start, "intervention_rng_digest": intervention_rng_start, "starting_update": START_UPDATE, "independent_loads": "YES", "optimizer_steps_all_384": "YES",
        }
        if not branch_identity["model_semantic_hash_equal"] or not branch_identity["optimizer_semantic_hash_equal"]:
            raise StageBlocker("branch_semantic_state_equality_failed")
        write_json(output / "sealed_input_verification.json", {**sealed, "certified_bundle": certified_bundle})
        write_json(output / "authorization_verification.json", authorization)
        write_json(output / "branch_identity_equality.json", branch_identity)
        write_json(output / "parameter_order_verification.json", {"canonical_order": list(ALL_NAMES), "control_table": control_table, "intervention_table": intervention_table, "same_order": True})
        write_json(output / "execution_configuration_snapshot.json", {
            "schema_version": "stage3_h13_post_d8_d9_execution_configuration_v1", "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(), "source_revision": environment, "authorization": authorization, "sealed_design_hashes": sealed["design_hashes"],
            "starting_state": {"path": str(BUNDLE_PATH.resolve()), "sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_steps": START_UPDATE, "model_hash": EXPECTED_MODEL_HASH, "optimizer_hash": EXPECTED_OPTIMIZER_HASH, "exp_avg_hash": EXPECTED_EXP_AVG_HASH, "exp_avg_sq_hash": EXPECTED_EXP_AVG_SQ_HASH, "rng_digest": control_rng_start},
            "updates": list(UPDATES), "objective_weights": {"lambda_hidden": LAMBDA_HIDDEN, "lambda_h1": LAMBDA_H1, "rollout": 1.0, "huber_beta": HUBER_BETA}, "optimizer": base.optimizer_contract(control_optimizer),
            "gradient_clipping": {"operation": "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)", "max_norm": CLIP_MAX_NORM}, "intervention": {"type": "ROLLOUT_GRADIENT_NORM_CAP_RELATIVE_TO_H1", "fixed_max_ratio": 1.0, "epsilon": EPSILON, "application": "whole-model rollout component after extraction and before aggregate/clip/AdamW"}, "validation_horizons": list(VALIDATION_HORIZONS),
            "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "r9_executed": "NO", "r8e_a1_hash_before": a1_before,
        })
        write_json(output / "protocol_integrity_identity.json", {"schema_version": "stage3_h13_post_d8_d9_protocol_integrity_identity_v1", "certified_start_state": EXPECTED_BUNDLE_SHA256, "control_loaded": control_loaded, "intervention_loaded": intervention_loaded, "branch_identity": branch_identity, "source_revision": EXPECTED_SOURCE_REVISION, "schedule": authority["schedule_hashes"], "schedule_slice": authority["schedule_slice_hashes"], "optimizer": base.optimizer_contract(control_optimizer), "future_label_leakage": 0, "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "historical_artifacts_snapshot": "captured_and_checked"})

        base.restore_rng(bundle["rng_state"])
        control_rows: list[dict[str, Any]] = []
        for zero_index in range(384, 392):
            control_rows.append(run_update("CONTROL", control_model, control_optimizer, pre, authority, output, control_rows[-1] if control_rows else None, zero_index))
        base.restore_rng(bundle["rng_state"])
        intervention_rows: list[dict[str, Any]] = []
        for zero_index in range(384, 392):
            intervention_rows.append(run_update("INTERVENTION", intervention_model, intervention_optimizer, pre, authority, output, intervention_rows[-1] if intervention_rows else None, zero_index))
        if [row["update"] for row in control_rows] != list(UPDATES) or [row["update"] for row in intervention_rows] != list(UPDATES):
            raise StageBlocker("exact_update_window_not_completed")
        control_reproduction = compare_control_to_d7(control_rows, authoritative_d7_rows())
        if control_reproduction["CONTROL_REPRODUCES_CANONICAL_D7"] != "YES":
            raise StageBlocker(f"control_reproduction_failed:{control_reproduction['first_mismatch']}")
        all_rows = control_rows + intervention_rows
        if any(row["finite_status"] != "FINITE" or row["future_label_leakage"] != 0 for row in all_rows):
            raise StageBlocker("finite_or_leakage_gate_failed")
        if any(row["manipulation_checks"]["rollout_direction_preservation_pass"] != "PASS" or row["manipulation_checks"]["rollout_scaling_equation_pass"] != "PASS" for row in all_rows):
            raise StageBlocker("rollout_direction_or_scaling_gate_failed")
        if any(row["H1_gradient_unchanged"] != "YES" or row["hidden_gradient_unchanged"] != "YES" or row["direct_non_rollout_gradient_equality_pre_global_clip"] != "YES" for row in all_rows):
            raise StageBlocker("direct_non_rollout_equality_gate_failed")
        c392 = next(row for row in control_rows if row["update"] == 392)
        i392 = next(row for row in intervention_rows if row["update"] == 392)
        endpoint_values = {"CONTROL_H1_UPDATE_392": c392["eval_h1"], "INTERVENTION_H1_UPDATE_392": i392["eval_h1"], "H1_CAUSAL_EFFECT_CONTROL_MINUS_INTERVENTION": c392["eval_h1"] - i392["eval_h1"], "INTERVENTION_H1_GUARDRAIL_PASS": "YES" if i392["eval_h1"] <= EXPECTED_H1_THRESHOLD else "NO", "CONTROL_H32_UPDATE_392": c392["eval_h32"], "INTERVENTION_H32_UPDATE_392": i392["eval_h32"], "INTERVENTION_H32_GUARDRAIL_PASS": "YES" if i392["eval_h32"] <= EXPECTED_H32_THRESHOLD else "NO", "CONTROL_H1_PASS": "YES" if c392["eval_h1"] <= EXPECTED_H1_THRESHOLD else "NO", "CONTROL_H32_PASS": "YES" if c392["eval_h32"] <= EXPECTED_H32_THRESHOLD else "NO"}
        integrity = {
            "STATUS": "PASS", "SEALED_INPUTS_MATCH": "YES", "SOURCE_REVISION_MATCH": "YES", "CERTIFIED_START_STATE_MATCH": "YES", "BRANCH_STATE_IDENTITY_AT_START": "YES", "BATCH_SCHEDULE_MATCH": "YES", "UPDATES_EXECUTED": "385-392_ONLY", "FUTURE_LABEL_LEAKAGE": 0,
            "ROLLOUT_DIRECTION_PRESERVED_ALL_UPDATES": "YES", "ROLLOUT_SCALING_EQUATION_PASS_ALL_UPDATES": "YES", "H1_DIRECTLY_UNCHANGED_PRECLIP_ALL_UPDATES": "YES", "HIDDEN_DIRECTLY_UNCHANGED_PRECLIP_ALL_UPDATES": "YES", "DIRECT_NON_ROLLOUT_GRADIENT_EQUALITY_PRECLIP_ALL_UPDATES": "YES", "CLIPPING_SEMANTICS_UNCHANGED": "YES", "ADAMW_SEMANTICS_UNCHANGED": "YES", "CONTROL_REPRODUCED_D7": "YES", "ALL_REQUIRED_METRICS_FINITE": "YES", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HISTORICAL_ARTIFACTS_MODIFIED": "NO", "NEXT_EXPERIMENT_EXECUTED": "NO",
        }
        classification, conditions = classify(endpoint_values["CONTROL_H1_UPDATE_392"], endpoint_values["INTERVENTION_H1_UPDATE_392"], endpoint_values["CONTROL_H32_UPDATE_392"], endpoint_values["INTERVENTION_H32_UPDATE_392"], True)
        summary = {
            "status": "PASSED", "first_blocker": "none", "causal_classification": classification, "one_sentence_verdict": f"The matched D9 rollout-magnitude cap replay preserved rollout direction and changed update-392 H1 by {endpoint_values['H1_CAUSAL_EFFECT_CONTROL_MINUS_INTERVENTION']:.17g} (CONTROL minus INTERVENTION), with intervention H32 {endpoint_values['INTERVENTION_H32_UPDATE_392']:.17g}.", "output_directory": str(output.resolve()), "integrity": integrity, "control_reproduction": control_reproduction, "branch_identity": branch_identity, "classification_conditions": conditions, "endpoints": endpoint_values, "control_rows": control_rows, "intervention_rows": intervention_rows, "control_final_validation": c392["validation"], "intervention_final_validation": i392["validation"], "sealed_input_hashes": sealed["design_hashes"], "authorization": authorization,
        }
        write_json(output / "control_reproduction_evidence.json", control_reproduction)
        write_json(output / "manipulation_evidence.json", {"fixed_max_ratio": 1.0, "epsilon": EPSILON, "updates": [{"update": row["update"], "branch": row["branch"], "checks": row["manipulation_checks"]} for row in all_rows], "rollout_direction_preserved_all_updates": "YES", "H1_directly_unchanged_preclip_all_updates": "YES", "hidden_directly_unchanged_preclip_all_updates": "YES"})
        write_json(output / "validation_endpoint_evidence.json", {"thresholds": {"H1_guardrail": EXPECTED_H1_THRESHOLD, "H32_guardrail": EXPECTED_H32_THRESHOLD, "H1_material_effect": MATERIAL_H1_THRESHOLD}, "horizons": list(VALIDATION_HORIZONS), "control": c392["validation"], "intervention": i392["validation"], "endpoints": endpoint_values})
        write_json(output / "global_clipping_mediation.json", {"control": [{"update": row["update"], "raw_norm": row["raw_total_gradient_norm"], "clip_coefficient": row["effective_global_scaling_factor"], "activated": row["clipping_activated"], "post_clip_norm": row["clipped_total_gradient_norm"]} for row in control_rows], "intervention": [{"update": row["update"], "raw_norm": row["raw_total_gradient_norm"], "clip_coefficient": row["effective_global_scaling_factor"], "activated": row["clipping_activated"], "post_clip_norm": row["clipped_total_gradient_norm"]} for row in intervention_rows], "semantics_unchanged": "YES"})
        write_json(output / "adamw_mediation.json", {"control": [{"update": row["update"], "adamw": row["adamw"]} for row in control_rows], "intervention": [{"update": row["update"], "adamw": row["adamw"]} for row in intervention_rows], "semantics_unchanged": "YES"})
        write_json(output / "integrity_leakage_evidence.json", integrity)
        write_json(output / "update_metrics.json", {"CONTROL": control_rows, "INTERVENTION": intervention_rows})
        write_json(output / "causal_classification.json", {"classification": classification, "conditions": conditions, "endpoints": endpoint_values, "contract": sealed["design"]["classification"]})
        stable_names = ("sealed_input_verification.json", "authorization_verification.json", "branch_identity_equality.json", "parameter_order_verification.json", "execution_configuration_snapshot.json", "protocol_integrity_identity.json", "control_reproduction_evidence.json", "manipulation_evidence.json", "validation_endpoint_evidence.json", "global_clipping_mediation.json", "adamw_mediation.json", "integrity_leakage_evidence.json", "update_metrics.json", "causal_classification.json")
        stable_hashes = {name: sha256_file(output / name) for name in stable_names}
        summary["artifact_hashes"] = stable_hashes
        write_json(output / "terminal_certificate.json", summary)
        (output / "FINAL_REPORT.md").write_text(render_report(summary, stable_hashes), encoding="utf-8", newline="\n")
        write_manifest(output)
        base.verify_snapshot(authority["authority_snapshot"])
        if sha256_file(base.A1_CHECKPOINT) != a1_before:
            raise StageBlocker("r8e_a1_mutated_during_execution")
        if rng_digest(capture_rng()) != control_rng_start:
            raise StageBlocker("final_rng_did_not_match_certified_start")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification, "control_h1": endpoint_values["CONTROL_H1_UPDATE_392"], "intervention_h1": endpoint_values["INTERVENTION_H1_UPDATE_392"], "control_h32": endpoint_values["CONTROL_H32_UPDATE_392"], "intervention_h32": endpoint_values["INTERVENTION_H32_UPDATE_392"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), authorization)
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return fail_closed(output, f"unexpected_execution_failure:{type(exc).__name__}:{exc}", authorization)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args().output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
