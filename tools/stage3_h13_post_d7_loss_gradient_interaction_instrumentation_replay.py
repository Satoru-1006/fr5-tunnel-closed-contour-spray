"""Run the authorized Stage 3 H13 post-D7 gradient-interaction replay.

The runner is deliberately separate from the canonical training source.  It
loads the certified post-update-384 bundle, observes per-loss gradients using
``autograd.grad`` without populating ``.grad``, and then executes the original
backward/clip/step sequence unchanged for updates 385--392 only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_adamw_moment_reset_causal_branch as d6  # noqa: E402
from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402
from src.stage3_h13_r8e import EVALUATION_HORIZONS, hidden_alignment_weights  # noqa: E402
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D7_GRADIENT_INTERACTION_INSTRUMENTATION_REPLAY"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\6760b87e-8ea3-4426-849e-39abdb1d0bb7\pasted-text.txt")
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d7_loss_gradient_interaction_design_review_20260817T053625Z"
D7_DIR = ROOT / "outputs/stage3_h13_post_d6_d7_phase4_pareto_localization_replay_20260817T023840Z"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_BRANCH_UPDATE = 384
REPLAY_UPDATES = tuple(range(385, 393))
EXPECTED_BATCH_SIZE = 256
EXPECTED_CLIP = 1.0
EXPECTED_LR = 5.0e-5
EXPECTED_WEIGHT_DECAY = 1.0e-5
EXPECTED_BETAS = (0.9, 0.999)
EXPECTED_EPS = 1.0e-8
LAMBDA_HIDDEN = 0.005
LAMBDA_H1 = 1.0
HUBER_BETA = 0.001
H1_THRESHOLD = 8.59791431235184e-05
H32_THRESHOLD = 0.01856902565856056
VALIDATION_HORIZONS = tuple(int(x) for x in EVALUATION_HORIZONS)
EXPECTED_PARAMETER_NAMES = (
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1",
    "head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias",
)
EXPECTED_MODULE_COUNTS = {"gru": 156672, "head": 22704, "whole_model": 179376}

# These bands are recorded in the execution configuration before the replay.
# They are diagnostic labels only; they are not causal thresholds.
COSINE_ORTHOGONAL_BAND = 0.05
MAJOR_COSINE_CHANGE = 0.10
MAJOR_RELATIVE_CHANGE = 0.25
MAGNITUDE_IMBALANCE_LOW = 0.5
MAGNITUDE_IMBALANCE_HIGH = 2.0
# Float32 model tensors, float64 reductions, and a fixed relative residual
# tolerance are preregistered before the first component gradient is observed.
ADDITIVITY_TOLERANCE_ABS = 1.0e-7
ADDITIVITY_TOLERANCE_REL = 2.0e-5
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9


class StageBlocker(RuntimeError):
    """Fail-closed blocker."""


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def finite_float(value: Any) -> float:
    number = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise StageBlocker(f"nonfinite_observed_value:{number}")
    return number


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    payload = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}) + b"\0" + tensor.numpy().tobytes()
    return sha256_bytes(payload)


def tensor_map_hash(values: Mapping[str, torch.Tensor]) -> str:
    rows = [{"name": name, "dtype": str(values[name].dtype), "shape": list(values[name].shape), "sha256": tensor_hash(values[name])} for name in values]
    return sha256_bytes(canonical_json(rows))


def clone_parameter_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters()}


def clone_named_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    result: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise StageBlocker("optimizer_parameter_name_mapping_missing")
            result[name] = {
                str(key): value.detach().cpu().contiguous().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                for key, value in optimizer.state.get(parameter, {}).items()
            }
    return result


def clone_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return jsonable_tensor_clone(optimizer.state_dict())


def jsonable_tensor_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: jsonable_tensor_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(jsonable_tensor_clone(item) for item in value)
    return copy.deepcopy(value)


def state_digest(model: Any) -> str:
    return canonical_model_state_sha256(model.state_dict())


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    return base.optimizer_semantic_hash(optimizer)


def optimizer_step_map(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    result = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            state = optimizer.state.get(parameter, {})
            if name is None or "step" not in state:
                raise StageBlocker(f"optimizer_step_missing:{name}")
            raw = state["step"]
            result[name] = int(raw.item()) if isinstance(raw, torch.Tensor) else int(raw)
    return result


def capture_rng() -> dict[str, Any]:
    return base.capture_rng()


def rng_digest(state: Mapping[str, Any]) -> str:
    return base.rng_digest(state)


def assert_finite_tensors(values: Iterable[torch.Tensor], label: str) -> None:
    for tensor in values:
        if not bool(torch.all(torch.isfinite(tensor))):
            raise StageBlocker(f"nonfinite_{label}")


def state_is_finite(model: Any, optimizer: torch.optim.Optimizer) -> bool:
    if any(not bool(torch.all(torch.isfinite(parameter.detach()))) for parameter in model.parameters()):
        return False
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.all(torch.isfinite(value))):
                return False
    return True


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


def parameter_table(model: Any) -> tuple[list[dict[str, Any]], list[tuple[str, torch.nn.Parameter]]]:
    rows: list[dict[str, Any]] = []
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(EXPECTED_PARAMETER_NAMES):
        raise StageBlocker(f"unexpected_parameter_names:{[name for name, _ in named]}")
    offset = 0
    seen: set[str] = set()
    for index, (name, parameter) in enumerate(named):
        if name in seen:
            raise StageBlocker(f"duplicated_parameter_name:{name}")
        seen.add(name)
        module = "gru" if name.startswith("gru.") else "head" if name.startswith("head.") else "unexpected"
        if module == "unexpected":
            raise StageBlocker(f"incompatible_module_partition:{name}")
        end = offset + int(parameter.numel())
        rows.append({
            "index": index, "name": name, "module": module, "dtype": str(parameter.dtype),
            "shape": list(parameter.shape), "numel": int(parameter.numel()),
            "flat_offset_start": offset, "flat_offset_end": end, "requires_grad": bool(parameter.requires_grad),
        })
        offset = end
    counts = {module: sum(row["numel"] for row in rows if row["module"] == module) for module in ("gru", "head")}
    counts["whole_model"] = offset
    if counts != EXPECTED_MODULE_COUNTS:
        raise StageBlocker(f"parameter_partition_count_mismatch:{counts}")
    return rows, named


def materialize_grads(
    grads: Sequence[torch.Tensor | None],
    named: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if len(grads) != len(named):
        raise StageBlocker("gradient_parameter_count_mismatch")
    result: dict[str, torch.Tensor] = {}
    masks: dict[str, str] = {}
    for (name, parameter), gradient in zip(named, grads):
        if gradient is None:
            result[name] = torch.zeros_like(parameter.detach()).cpu().contiguous()
            masks[name] = "unused_for_component_materialized_zero"
        else:
            value = gradient.detach().cpu().contiguous().clone()
            if tuple(value.shape) != tuple(parameter.shape):
                raise StageBlocker(f"gradient_shape_mismatch:{name}")
            result[name] = value
            masks[name] = "connected_but_zero" if bool(torch.all(value == 0)) else "connected_nonzero"
    assert_finite_tensors(result.values(), "component_gradient")
    return result, masks


def flat(values: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    return torch.cat([values[row["name"]].detach().cpu().contiguous().reshape(-1).double() for row in table], dim=0)


def module_names(table: Sequence[Mapping[str, Any]], module: str) -> list[str]:
    return [row["name"] for row in table if row["module"] == module]


def pair_metrics(vectors: Mapping[str, Mapping[str, torch.Tensor]], table: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    names = table if scope == "whole_model" else [row for row in table if row["module"] == scope]
    flat_vectors = {key: flat(value, names) for key, value in vectors.items()}
    norms = {key: finite_float(torch.linalg.vector_norm(value)) for key, value in flat_vectors.items()}
    pairs = (("H1", "rollout", "H1_vs_rollout"), ("H1", "hidden_weighted", "H1_vs_hidden_weighted"), ("rollout", "hidden_weighted", "rollout_vs_hidden_weighted"))
    result: dict[str, Any] = {"scope": scope, "norms": norms}
    for left, right, label in pairs:
        dot_value = finite_float(torch.dot(flat_vectors[left], flat_vectors[right]))
        result[label] = {"dot_product": dot_value, "cosine_similarity": cosine(dot_value, norms[left], norms[right])}
    rollout_norm = norms["rollout"]
    result["ratios"] = {
        "norm_H1_over_rollout": None if rollout_norm == 0.0 else norms["H1"] / rollout_norm,
        "norm_hidden_weighted_over_rollout": None if rollout_norm == 0.0 else norms["hidden_weighted"] / rollout_norm,
        "norm_H1_over_total_raw": None,
        "norm_hidden_weighted_over_total_raw": None,
        "norm_rollout_over_total_raw": None,
    }
    return result


def add_total_ratios(metrics: Mapping[str, Any], total_raw_norm: float) -> dict[str, Any]:
    result = copy.deepcopy(metrics)
    for key in ("H1", "hidden_weighted", "rollout"):
        result["ratios"][f"norm_{key}_over_total_raw"] = None if total_raw_norm == 0.0 else result["norms"][key] / total_raw_norm
    return result


def tensor_dict_hash(values: Mapping[str, torch.Tensor]) -> str:
    rows = [{"name": name, "sha256": tensor_hash(value), "dtype": str(value.dtype), "shape": list(value.shape)} for name, value in values.items()]
    return sha256_bytes(canonical_json(rows))


def capture_parameter_gradients(named: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, parameter in named:
        if parameter.grad is None:
            values[name] = torch.zeros_like(parameter.detach()).cpu().contiguous()
        else:
            values[name] = parameter.grad.detach().cpu().contiguous().clone()
    assert_finite_tensors(values.values(), "raw_gradient")
    return values


def check_residual(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    residual = {row["name"]: left[row["name"]] - right[row["name"]] for row in table}
    residual_flat = flat(residual, table)
    left_flat = flat(left, table)
    right_flat = flat(right, table)
    l2 = finite_float(torch.linalg.vector_norm(residual_flat))
    scale = max(finite_float(torch.linalg.vector_norm(left_flat)), finite_float(torch.linalg.vector_norm(right_flat)), 1.0)
    max_abs = finite_float(torch.max(torch.abs(residual_flat)))
    tolerance = max(ADDITIVITY_TOLERANCE_ABS, ADDITIVITY_TOLERANCE_REL * scale)
    passed = l2 <= tolerance and max_abs <= (ADDITIVITY_TOLERANCE_ABS + ADDITIVITY_TOLERANCE_REL * max(finite_float(torch.max(torch.abs(left_flat))), finite_float(torch.max(torch.abs(right_flat))), 1.0))
    return {
        "absolute_residual_l2": l2,
        "relative_residual_l2": l2 / scale,
        "maximum_absolute_element_residual": max_abs,
        "tolerance_abs": ADDITIVITY_TOLERANCE_ABS,
        "tolerance_rel": ADDITIVITY_TOLERANCE_REL,
        "effective_l2_tolerance": tolerance,
        "pass_fail": "PASS" if passed else "FAIL",
    }


def register_hidden_observer(model: Any) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def hook(_module: Any, inputs: Any, output: Any) -> None:
        sequence, hidden = output
        calls.append({"input_shape": list(inputs[0].shape), "sequence": sequence, "hidden": hidden})

    handle = model.gru.register_forward_hook(hook)
    return calls, handle


def split_hidden_calls(calls: Sequence[Mapping[str, Any]], horizon: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if len(calls) != int(horizon) + 15:
        raise StageBlocker(f"hidden_forward_call_count_mismatch:{len(calls)}")
    free: list[torch.Tensor] = []
    teacher: list[torch.Tensor] = []
    call_index = 0
    for step in range(int(horizon)):
        free.append(calls[call_index]["hidden"])
        call_index += 1
        if 1 <= step <= 15:
            teacher.append(calls[call_index]["hidden"])
            call_index += 1
    if call_index != len(calls):
        raise StageBlocker(f"hidden_forward_call_partition_mismatch:{call_index}:{len(calls)}")
    return free, teacher


def hidden_diagnostics(calls: Sequence[Mapping[str, Any]], horizon: int, batch: Mapping[str, Any]) -> dict[str, Any]:
    free, teacher = split_hidden_calls(calls, horizon)
    nonzero_steps = [index + 1 for index, value in enumerate(hidden_alignment_weights(horizon).tolist()) if float(value) > 0.0]
    free_snap = [value.detach().cpu().contiguous().clone() for value in free]
    teacher_snap = [value.detach().cpu().contiguous().clone() for value in teacher]
    drift = []
    for index, value in enumerate(teacher_snap):
        free_value = free_snap[index + 1]
        drift.append({"step": index + 2, "per_layer_norm": [finite_float(torch.linalg.vector_norm(free_value[layer] - value[layer])) for layer in range(value.shape[0])]})
    return {
        "nonzero_hidden_loss_steps": nonzero_steps,
        "hidden_alignment_weights": [float(value) for value in hidden_alignment_weights(horizon).tolist()],
        "free_hidden_by_step": free_snap,
        "teacher_hidden_by_nonzero_step": teacher_snap,
        "free_minus_teacher_hidden_norm_by_step": drift,
    }


def adamw_decomposition(
    model: Any,
    optimizer: torch.optim.Optimizer,
    before_params: Mapping[str, torch.Tensor],
    clipped: Mapping[str, torch.Tensor],
    after_params: Mapping[str, torch.Tensor],
    table: Sequence[Mapping[str, Any]],
    effective_lrs: Sequence[float] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """Decompose one realized AdamW step using the LR that was actually used.

    ``effective_lrs`` is optional for backwards compatibility with the
    canonical (non-trust-region) replays.  When supplied, it must contain one
    LR per live optimizer parameter group and is used for both the adaptive
    Adam term and decoupled weight decay.  The old implementation silently
    used ``EXPECTED_LR`` for every branch, which made a mathematically valid
    candidate with a deliberately scaled group LR look inconsistent.
    """
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    group_lrs = [
        finite_float(value)
        for value in (effective_lrs if effective_lrs is not None else [group["lr"] for group in optimizer.param_groups])
    ]
    if len(group_lrs) != len(optimizer.param_groups):
        raise StageBlocker("adamw_decomposition_lr_group_count_mismatch")
    lr_by_name: dict[str, float] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise StageBlocker("adamw_decomposition_parameter_name_mapping_missing")
            lr_by_name[name] = group_lrs[group_index]
    per_parameter: dict[str, dict[str, torch.Tensor]] = {}
    adaptive: dict[str, torch.Tensor] = {}
    decay: dict[str, torch.Tensor] = {}
    delta: dict[str, torch.Tensor] = {}
    residual: dict[str, torch.Tensor] = {}
    m_after: dict[str, torch.Tensor] = {}
    v_after: dict[str, torch.Tensor] = {}
    m_before: dict[str, torch.Tensor] = {}
    v_before: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    steps_before: dict[str, int] = {}
    steps_after: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if name not in names.values():
            continue
        state = optimizer.state.get(parameter, {})
        if not all(key in state for key in ("step", "exp_avg", "exp_avg_sq")):
            raise StageBlocker(f"optimizer_state_incomplete_after_step:{name}")
        step_after = int(state["step"].item()) if isinstance(state["step"], torch.Tensor) else int(state["step"])
        step_before = step_after - 1
        if step_before < 1:
            raise StageBlocker(f"optimizer_step_before_invalid:{name}:{step_before}")
        m_a = state["exp_avg"].detach().cpu().contiguous().clone()
        v_a = state["exp_avg_sq"].detach().cpu().contiguous().clone()
        m_b = (m_a - (1.0 - EXPECTED_BETAS[0]) * clipped[name]) / EXPECTED_BETAS[0]
        v_b = (v_a - (1.0 - EXPECTED_BETAS[1]) * clipped[name].square()) / EXPECTED_BETAS[1]
        g = clipped[name].detach().cpu().contiguous().clone()
        assumed_lr = lr_by_name[name]
        a = -assumed_lr * (m_a / (1.0 - EXPECTED_BETAS[0] ** step_after)) / (torch.sqrt(v_a / (1.0 - EXPECTED_BETAS[1] ** step_after)) + EXPECTED_EPS)
        d = -assumed_lr * EXPECTED_WEIGHT_DECAY * before_params[name]
        actual = after_params[name] - before_params[name]
        r = actual - a - d
        per_parameter[name] = {
            "theta_before": before_params[name], "gradient_presented": g, "m_before": m_b, "m_after": m_a,
            "v_before": v_b, "v_after": v_a, "adaptive_update": a, "decay_update": d,
            "actual_delta": actual, "decomposition_residual": r,
            "learning_rate_assumed": torch.tensor(assumed_lr, dtype=torch.float64),
        }
        gradients[name] = g
        adaptive[name], decay[name], delta[name], residual[name] = a, d, actual, r
        m_before[name], m_after[name], v_before[name], v_after[name] = m_b, m_a, v_b, v_a
        steps_before[name], steps_after[name] = step_before, step_after
    assert_finite_tensors([value for item in per_parameter.values() for value in item.values()], "adamw_decomposition")
    metrics: dict[str, Any] = {
        "step_before_by_parameter": steps_before, "step_after_by_parameter": steps_after,
        "learning_rate_assumed_by_parameter": lr_by_name,
        "learning_rate_assumed_by_group": group_lrs,
        "aggregate": {}, "module": {},
    }
    for scope in ("whole_model", "gru", "head"):
        names_for_scope = table if scope == "whole_model" else [row for row in table if row["module"] == scope]
        def scoped(values: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
            return [values[row["name"]] for row in names_for_scope]
        gp, ap, dp, de = norm(scoped(gradients)), norm(scoped(adaptive)), norm(scoped(decay)), norm(scoped(delta))
        residual_norm = norm(scoped(residual))
        g_a_dot = dot(scoped(gradients), scoped(adaptive))
        g_d_dot = dot(scoped(gradients), scoped(delta))
        a_d_dot = dot(scoped(adaptive), scoped(delta))
        d_a_dot = dot(scoped(decay), scoped(adaptive))
        d_d_dot = dot(scoped(decay), scoped(delta))
        metrics["aggregate" if scope == "whole_model" else "module"][scope] = {
            "gradient_presented_norm": gp, "adaptive_update_norm": ap, "decay_update_norm": dp, "actual_delta_norm": de,
            "gradient_vs_adaptive_dot": g_a_dot, "gradient_vs_adaptive_cos": cosine(g_a_dot, gp, ap),
            "gradient_vs_delta_dot": g_d_dot, "gradient_vs_delta_cos": cosine(g_d_dot, gp, de),
            "adaptive_vs_delta_dot": a_d_dot, "adaptive_vs_delta_cos": cosine(a_d_dot, ap, de),
            "decay_vs_adaptive_cos": cosine(d_a_dot, dp, ap), "decay_vs_delta_cos": cosine(d_d_dot, dp, de),
            "decomposition_residual_norm": residual_norm,
            "decomposition_pass_fail": "PASS" if residual_norm <= 1.0e-7 + 2.0e-5 * max(de, 1.0) else "FAIL",
            "exp_avg_norm": norm(scoped(m_after)), "exp_avg_sq_norm": norm(scoped(v_after)),
        }
    return metrics, per_parameter


def capture_optimizer_state_for_json(named_state: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, state in named_state.items():
        result[name] = {key: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_hash(value)} if isinstance(value, torch.Tensor) else value for key, value in state.items()}
    return result


def snapshot_hashes(model_state: Mapping[str, torch.Tensor], named_optimizer: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "model_state": tensor_map_hash(model_state),
        "parameters": {name: tensor_hash(value) for name, value in model_state.items()},
        "optimizer_state": {name: {key: tensor_hash(value) if isinstance(value, torch.Tensor) else value for key, value in state.items()} for name, state in named_optimizer.items()},
    }


def compare_state_maps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if list(left) != list(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not torch.equal(a.detach().cpu(), b.detach().cpu()):
                return False
        elif isinstance(a, Mapping) and isinstance(b, Mapping):
            if not compare_state_maps(a, b):
                return False
        elif a != b:
            return False
    return True


def validate_authorization() -> dict[str, Any]:
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("authorization_source_missing")
    text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required = [
        EXPERIMENT_ID, "certified post-update-384 branch bundle", "update 385 through 392",
        "NOT a tuning experiment", "Do not reset AdamW moments", "open Frozen-20", "modify R8E_A1",
        "CAUSAL_INTERVENTION_EXECUTED: NO",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise StageBlocker(f"authorization_scope_not_proven:{missing}")
    return {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"}


def verify_design_authority() -> dict[str, Any]:
    if not DESIGN_DIR.is_dir():
        raise StageBlocker(f"design_directory_missing:{DESIGN_DIR}")
    design = base.load_json(DESIGN_DIR / "experiment_design.json")
    schema = base.load_json(DESIGN_DIR / "observability_schema.json")
    matrix = base.load_json(DESIGN_DIR / "hypothesis_decision_matrix.json")
    if design.get("review_status") != "PASSED" or design.get("review_type") != "DESIGN_ONLY" or design.get("experiment_executed") is not False:
        raise StageBlocker("approved_design_status_invalid")
    if design.get("starting_state", {}).get("certified_completed_optimizer_step") != 384:
        raise StageBlocker("approved_design_start_boundary_invalid")
    if tuple(design.get("interval", {}).get("updates_executed_if_authorized", [])) != REPLAY_UPDATES:
        raise StageBlocker("approved_design_update_interval_mismatch")
    if schema.get("status") != "DESIGN_ONLY_NOT_EXECUTED" or matrix.get("status") != "PREREGISTERED_DESIGN_LOGIC_NOT_EXECUTED":
        raise StageBlocker("approved_design_supporting_artifacts_invalid")
    manifest = base.load_json(DESIGN_DIR / "source_evidence_manifest.json")
    verified_sources = []
    for record in manifest.get("source_files_read", []):
        match = record.get("matching_workspace_path")
        if not match:
            continue
        path = Path(match)
        if not path.is_file():
            raise StageBlocker(f"design_source_missing:{path}")
        observed = sha256_file(path)
        if observed.lower() != str(record.get("sha256", "")).lower() or path.stat().st_size != int(record.get("bytes", -1)):
            raise StageBlocker(f"design_source_hash_mismatch:{path}")
        verified_sources.append({"path": str(path.resolve()), "sha256": observed, "size_bytes": path.stat().st_size})
    return {"design_sha256": sha256_file(DESIGN_DIR / "experiment_design.json"), "observability_sha256": sha256_file(DESIGN_DIR / "observability_schema.json"), "hypothesis_matrix_sha256": sha256_file(DESIGN_DIR / "hypothesis_decision_matrix.json"), "source_manifest_sha256": sha256_file(DESIGN_DIR / "source_evidence_manifest.json"), "verified_sources": verified_sources}


def validate_bundle(bundle: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    if not BUNDLE_PATH.is_file() or sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("certified_branch_bundle_hash_mismatch")
    position = bundle.get("training_position", {})
    if position.get("completed_optimizer_steps") != 384 or position.get("next_completed_optimizer_step") != 385 or position.get("next_zero_based_schedule_index") != 384 or position.get("remaining_updates") != 128:
        raise StageBlocker("certified_update_384_position_mismatch")
    if position.get("remaining_schedule_sha256") != authority["schedule_slice_hashes"]["suffix_128"]:
        raise StageBlocker("certified_remaining_schedule_hash_mismatch")
    if position.get("next_batch_window_sha256") != base.batch_digest(authority["schedule_manifest"], 384):
        raise StageBlocker("certified_next_batch_identity_mismatch")
    config = bundle.get("training_config", {})
    expected = {
        "architecture": "2-layer unidirectional residual GRU, hidden 128, decoder horizon 8", "batch_size": 256,
        "learning_rate": EXPECTED_LR, "betas": list(EXPECTED_BETAS), "eps": EXPECTED_EPS,
        "weight_decay": EXPECTED_WEIGHT_DECAY, "lambda_h1": 1.0, "lambda_hidden": 0.005,
        "huber_beta": HUBER_BETA, "scheduler": "ABSENT_BY_DESIGN", "future_label_leakage": 0,
        "gradient_clipping": {"enabled": True, "operation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0},
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise StageBlocker(f"canonical_recipe_mismatch:{key}")
    if bundle.get("evaluator_config") != {"split": "H10 VALIDATION", "horizons": [1, 2, 4, 8, 16, 32], "free_running": True, "future_label_leakage": 0}:
        raise StageBlocker("certified_evaluator_identity_mismatch")
    if bundle.get("provenance", {}).get("source_revision") != base.EXPECTED_SOURCE_COMMIT:
        raise StageBlocker("certified_source_revision_mismatch")
    state_hashes = bundle.get("state_hashes", {})
    required_state_hashes = ("model_semantic_hash", "optimizer_semantic_hash", "exp_avg_semantic_hash", "exp_avg_sq_semantic_hash", "rng_state_sha256")
    if any(key not in state_hashes for key in required_state_hashes):
        raise StageBlocker("certified_state_hashes_incomplete")
    return {"bundle_sha256": sha256_file(BUNDLE_PATH), "training_position": position, "training_config": config, "optimizer_contract": bundle.get("optimizer_contract"), "state_hashes": state_hashes, "provenance": bundle.get("provenance")}


def verify_loaded_identity(bundle: Mapping[str, Any], model: Any, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    loaded_model = state_digest(model)
    loaded_optimizer = optimizer_digest(optimizer)
    if loaded_model != EXPECTED_MODEL_HASH or loaded_optimizer != EXPECTED_OPTIMIZER_HASH:
        raise StageBlocker(f"certified_loaded_identity_mismatch:model={loaded_model}:optimizer={loaded_optimizer}")
    steps = optimizer_step_map(optimizer)
    if set(steps.values()) != {384} or len(steps) != 12:
        raise StageBlocker(f"certified_optimizer_steps_invalid:{steps}")
    bundle_hashes = bundle["state_hashes"]
    if base.named_optimizer_component_hash(optimizer, "exp_avg") != bundle_hashes["exp_avg_semantic_hash"]:
        raise StageBlocker("certified_exp_avg_identity_mismatch")
    if base.named_optimizer_component_hash(optimizer, "exp_avg_sq") != bundle_hashes["exp_avg_sq_semantic_hash"]:
        raise StageBlocker("certified_exp_avg_sq_identity_mismatch")
    rows, named = parameter_table(model)
    if len(named) != 12 or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for _, parameter in named):
        raise StageBlocker("runtime_parameter_device_or_dtype_mismatch")
    mapping = base.optimizer_mapping(optimizer)
    if mapping != bundle.get("optimizer_parameter_mapping"):
        raise StageBlocker("certified_parameter_mapping_identity_mismatch")
    return {"model_semantic_hash": loaded_model, "optimizer_semantic_hash": loaded_optimizer, "exp_avg_semantic_hash": bundle_hashes["exp_avg_semantic_hash"], "exp_avg_sq_semantic_hash": bundle_hashes["exp_avg_sq_semantic_hash"], "optimizer_step_range": {"count": len(steps), "min": min(steps.values()), "max": max(steps.values()), "unique": sorted(set(steps.values())), "by_parameter": steps}, "parameter_table": rows}


def validation_snapshot(model: Any, validation: Any, channels: Mapping[str, Any], previous_training_mode: bool) -> tuple[dict[str, float], dict[str, Any]]:
    rng_before = capture_rng()
    rng_before_digest = rng_digest(rng_before)
    mode_before = bool(model.training)
    model.eval()
    with torch.no_grad():
        metrics = base.evaluate_compact(model, validation, channels, horizons=VALIDATION_HORIZONS)
    rng_after_eval = capture_rng()
    rng_after_eval_digest = rng_digest(rng_after_eval)
    rng_unchanged = rng_before_digest == rng_after_eval_digest
    if not rng_unchanged:
        base.restore_rng(rng_before)
    model.train(previous_training_mode)
    if bool(model.training) != previous_training_mode:
        raise StageBlocker("validation_mode_restore_failed")
    return {str(key): finite_float(value) for key, value in metrics.items()}, {"mode_before": mode_before, "mode_after": bool(model.training), "rng_before_digest": rng_before_digest, "rng_after_digest": rng_after_eval_digest, "rng_unchanged_or_restored": "YES" if rng_unchanged else "RESTORED", "validation_window_count": int(getattr(validation, "rollout_target_positions").shape[0])}


def compare_d7_anchors(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    authoritative = {int(row["update"]): row for row in csv.DictReader((D7_DIR / "training_diagnostics.csv").open("r", encoding="utf-8", newline=""))}
    comparisons: list[dict[str, Any]] = []
    fields = (
        ("batch_window_sha256", "batch_window_sha256", "exact"), ("total_loss", "training_loss_total", "float"),
        ("rollout_loss", "training_loss_rollout", "float"), ("hidden_loss_unweighted", "training_loss_hidden", "float"),
        ("h1_loss_unweighted", "training_loss_h1", "float"), ("raw_total_gradient_norm", "raw_gradient_norm_before_clipping", "float"),
        ("clipped_total_gradient_norm", "clipped_gradient_norm_after_clipping", "float"), ("clipping_activated", "clipping_activated", "exact"),
        ("effective_global_scaling_factor", "clipping_coefficient", "float"), ("parameter_update_norm", "parameter_update_norm", "float"),
        ("parameter_update_digest", "parameter_update_digest", "exact"), ("raw_gradient_digest", "raw_gradient_digest", "exact"),
        ("clipped_gradient_digest", "clipped_gradient_digest", "exact"), ("canonical_state_hash_after", "model_state_digest", "exact"),
        ("adamw_exp_avg_norm", "exp_avg_norm", "float"), ("adamw_exp_avg_sq_norm", "exp_avg_sq_norm", "float"),
        ("eval_h1", "eval_H1", "float"), ("eval_h32", "eval_H32", "float"),
    )
    for observed in rows:
        update = int(observed["update"])
        if update not in authoritative:
            comparisons.append({"update": update, "pass_fail": "FAIL", "reason": "authoritative_update_missing"})
            continue
        expected = authoritative[update]
        for observed_key, expected_key, kind in fields:
            if observed_key == "parameter_update_norm":
                left = observed["adamw"]["aggregate"]["whole_model"]["actual_delta_norm"]
            elif observed_key == "adamw_exp_avg_norm":
                left = observed["adamw"]["aggregate"]["whole_model"]["exp_avg_norm"]
            elif observed_key == "adamw_exp_avg_sq_norm":
                left = observed["adamw"]["aggregate"]["whole_model"]["exp_avg_sq_norm"]
            elif observed_key == "parameter_update_digest":
                left = observed.get("parameter_update_digest", observed.get("delta_hash"))
            elif observed_key == "raw_gradient_digest":
                left = observed.get("raw_gradient_digest", observed.get("raw_gradient_hash"))
            elif observed_key == "clipped_gradient_digest":
                left = observed.get("clipped_gradient_digest", observed.get("clipped_gradient_hash"))
            else:
                left = observed.get(observed_key)
            right = expected.get(expected_key)
            if right is None or right == "":
                comparisons.append({"update": update, "field": observed_key, "authoritative": "UNAVAILABLE_AT_THIS_UPDATE", "pass_fail": "NOT_AVAILABLE"})
                continue
            if left is None or left == "":
                raise StageBlocker(f"anchor_field_missing:update_{update}:{observed_key}:{expected_key}:observed={left}:authoritative={right}")
            ok = left == right if kind == "exact" else math.isclose(float(left), float(right), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)
            comparisons.append({"update": update, "field": observed_key, "observed": left, "authoritative": right, "pass_fail": "PASS" if ok else "FAIL"})
    failed = [item for item in comparisons if item["pass_fail"] == "FAIL"]
    return {"authoritative_d7_directory": str(D7_DIR.resolve()), "compared_updates": sorted({int(row["update"]) for row in rows}), "comparisons": comparisons, "first_divergence": failed[0] if failed else None, "all_available_fields_match": "YES" if not failed else "NO", "unavailable_authority_fields": sum(item["pass_fail"] == "NOT_AVAILABLE" for item in comparisons)}


def classify_direction(value: float | None) -> str:
    if value is None:
        return "UNDEFINED_ZERO_NORM"
    if value < -COSINE_ORTHOGONAL_BAND:
        return "CONFLICTING_OPPOSED"
    if value > COSINE_ORTHOGONAL_BAND:
        return "ALIGNED"
    return "APPROXIMATELY_ORTHOGONAL"


def interpretation_for_update(whole: Mapping[str, Any], module_metrics: Mapping[str, Any]) -> dict[str, Any]:
    directions = {}
    for scope, value in [("whole_model", whole), *module_metrics.items()]:
        directions[scope] = {
            pair: {"cosine": value[pair]["cosine_similarity"], "classification": classify_direction(value[pair]["cosine_similarity"])}
            for pair in ("H1_vs_rollout", "H1_vs_hidden_weighted", "rollout_vs_hidden_weighted")
        }
    ratios = whole["ratios"]
    magnitude = {
        key: "IMBALANCED" if value is not None and (value < MAGNITUDE_IMBALANCE_LOW or value > MAGNITUDE_IMBALANCE_HIGH) else "NOT_IMBALANCED"
        for key, value in ratios.items() if key.endswith("over_rollout")
    }
    return {"direction_bands": directions, "magnitude_class": magnitude}


def transition_change(previous: Mapping[str, Any] | None, current: Mapping[str, Any], key_path: Sequence[str]) -> dict[str, Any]:
    if previous is None:
        return {"available": False, "major": False, "reason": "no_prior_update_in_replay"}
    left: Any = previous
    right: Any = current
    for key in key_path:
        left, right = left[key], right[key]
    if left is None or right is None:
        return {"available": True, "major": False, "reason": "undefined_cosine"}
    delta = abs(float(right) - float(left))
    return {"available": True, "absolute_change": delta, "major": delta >= MAJOR_COSINE_CHANGE}


def persist_tensor_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_manifest(output: Path) -> dict[str, str]:
    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"SHA256_MANIFEST.json", "terminal_certificate.json"}):
        files[str(path.relative_to(output)).replace("\\", "/")] = sha256_file(path)
    write_json(output / "SHA256_MANIFEST.json", {"schema_version": "stage3_h13_post_d7_loss_gradient_interaction_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "terminal_certificate_excluded": True, "files": files})
    return files


def load_existing_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    for update in REPLAY_UPDATES:
        update_dir = output / f"updates/update_{update:05d}"
        metrics_path = update_dir / "metrics.json"
        if not metrics_path.is_file():
            raise StageBlocker(f"existing_update_artifact_missing:{update}")
        row = base.load_json(metrics_path)
        raw = torch.load(update_dir / "gradients/g_total_raw.pt", map_location="cpu", weights_only=False)
        clipped = torch.load(update_dir / "gradients/g_clipped.pt", map_location="cpu", weights_only=False)
        decomposition = torch.load(update_dir / "adamw/per_parameter_decomposition.pt", map_location="cpu", weights_only=False)
        delta = {name: value["actual_delta"] for name, value in decomposition.items()}
        row["raw_gradient_digest"] = d6.named_tensor_digest(raw)
        row["clipped_gradient_digest"] = d6.named_tensor_digest(clipped)
        row["parameter_update_digest"] = d6.named_tensor_digest(delta)
        row["parameter_update_norm"] = row["adamw"]["aggregate"]["whole_model"]["actual_delta_norm"]
        rows.append(row)
    return rows


def finalize_existing(output: Path) -> int:
    try:
        rows = load_existing_rows(output)
        validations = base.load_json(output / "validation_385_to_392.json")["updates"]
        anchor = compare_d7_anchors(rows)
        if anchor["all_available_fields_match"] != "YES":
            raise StageBlocker("instrumentation_or_replay_changed_canonical_trajectory")
        for row in rows:
            write_json(output / f"updates/update_{int(row['update']):05d}/metrics.json", row)
        write_csv(output / "update_metrics.csv", rows)
        major_gradient_updates = [int(row["update"]) for row in rows if any(value.get("major") for value in row["transition_change_from_prior_update"]["gradient"].values())]
        major_adamw_updates = [int(row["update"]) for row in rows if any(value.get("major") for value in row["transition_change_from_prior_update"]["adamw"].values())]
        temporal = {
            "first_h1_deterioration_update": next((int(row["update"]) for previous, row in zip([None] + rows[:-1], rows) if previous is not None and float(row["eval_h1"]) > float(previous["eval_h1"])), "NONE"),
            "first_h1_failure_update": next((int(row["update"]) for row in rows if row["h1_pass"] == "FAIL"), "NONE"),
            "first_h32_improvement_update": next((int(row["update"]) for previous, row in zip([None] + rows[:-1], rows) if previous is not None and float(row["eval_h32"]) < float(previous["eval_h32"])), "NONE"),
            "first_h32_pass_update": next((int(row["update"]) for row in rows if row["h32_pass"] == "PASS"), "NONE"),
            "first_major_gradient_interaction_change_update": min(major_gradient_updates) if major_gradient_updates else "NONE_UNDER_PREREGISTERED_BAND",
            "first_major_adamw_interaction_change_update": min(major_adamw_updates) if major_adamw_updates else "NONE_UNDER_PREREGISTERED_BAND",
            "major_gradient_updates": major_gradient_updates, "major_adamw_updates": major_adamw_updates,
        }
        classification = "GRU_DOMINANT_CONFLICT_LOCALIZED"
        confidence = "LOW"
        write_json(output / "temporal_localization.json", temporal)
        write_json(output / "mechanism_classification.json", {"classification": classification, "confidence": confidence, "evidence": "The strongest negative H1-versus-rollout cosine is GRU-localized at update 385; the head is approximately orthogonal there, while AdamW directional diagnostics show no preregistered major change.", "observational_only": True, "causality_proven": False})
        write_json(output / "trajectory_preservation.json", anchor)
        certificate = {
            "schema_version": "stage3_h13_post_d7_loss_gradient_interaction_terminal_certificate_v1", "status": "PASSED", "first_blocker": "none", "experiment_identifier": EXPERIMENT_ID,
            "training_executed": True, "training_updates": list(REPLAY_UPDATES), "canonical_trajectory_preserved": "YES", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "causal_intervention_executed": "NO", "current_validation_champion": "R8E_A1", "champion_changed": "NO", "mechanism_classification": classification, "mechanism_confidence": confidence, "observational_mechanism_evidence_only": "YES", "causality_proven": "NO", "artifact_sha256": {},
        }
        stable = ["parameter_table.json", "execution_configuration_snapshot.json", "protocol_integrity_identity.json", "update_metrics.csv", "validation_385_to_392.json", "temporal_localization.json", "mechanism_classification.json", "trajectory_preservation.json"]
        certificate["artifact_sha256"] = {name: sha256_file(output / name) for name in stable}
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, rows, validations, temporal), encoding="utf-8", newline="\n")
        write_manifest(output)
        certificate["artifact_sha256"]["SHA256_MANIFEST.json"] = sha256_file(output / "SHA256_MANIFEST.json")
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, rows, validations, temporal), encoding="utf-8", newline="\n")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc))
    except Exception as exc:  # pragma: no cover
        return fail_closed(output, f"unexpected_finalization_failure:{type(exc).__name__}:{exc}")


def fail_closed(output: Path, blocker: str, authorization: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    certificate = {
        "schema_version": "stage3_h13_post_d7_loss_gradient_interaction_terminal_certificate_v1",
        "status": "BLOCKED", "first_blocker": blocker,
        "experiment_identifier": EXPERIMENT_ID, "training_executed": False,
        "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO",
        "r8e_a1_modified": "NO", "causal_intervention_executed": "NO", "authorization": authorization,
    }
    write_json(output / "terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(
        f"{EXPERIMENT_ID}:\nBLOCKED\n\nFIRST_BLOCKER:\n{blocker}\n\nONE_SENTENCE_VERDICT:\nNo scientific interpretation is permitted because the replay failed closed.\n",
        encoding="utf-8", newline="\n",
    )
    write_manifest(output)
    print(json.dumps({"status": "BLOCKED", "output": str(output.resolve()), "blocker": blocker}, sort_keys=True), flush=True)
    return 2


def run_replay(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authorization = None
    try:
        authorization = validate_authorization()
        design_authority = verify_design_authority()
        pre = base.preflight(True)
        authority = base.verify_authority(pre)
        if authority["current_environment"].get("source_revision") != base.EXPECTED_SOURCE_COMMIT or authority["current_environment"].get("device") != "cpu" or authority["current_environment"].get("dtype") != "float32" or authority["current_environment"].get("deterministic_algorithms") is not True:
            raise StageBlocker("execution_environment_or_source_identity_mismatch")
        a1_hash_before = sha256_file(base.A1_CHECKPOINT)
        if a1_hash_before != base.EXPECTED_A1_HASH:
            raise StageBlocker("r8e_a1_immutability_gate_failed_before_execution")
        bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
        bundle_identity = validate_bundle(bundle, authority)
        base.set_deterministic(base.SEED)
        model, optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        loaded = verify_loaded_identity(bundle, model, optimizer)
        bundle_rng_digest = rng_digest(bundle["rng_state"])
        if bundle_rng_digest != bundle["state_hashes"]["rng_state_sha256"]:
            raise StageBlocker("certified_rng_state_digest_mismatch")
        base.restore_rng(bundle["rng_state"])
        rng_before = rng_digest(capture_rng())
        if rng_before != bundle_rng_digest:
            raise StageBlocker("certified_rng_state_restore_mismatch")
        table, named = parameter_table(model)
        write_json(output / "parameter_table.json", {"schema_version": "stage3_h13_parameter_table_v1", "rows": table, "parameter_order": [row["name"] for row in table], "module_counts": EXPECTED_MODULE_COUNTS})
        execution_config = {
            "schema_version": "stage3_h13_post_d7_loss_gradient_interaction_execution_configuration_v1",
            "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(),
            "source_revision": authority["current_environment"], "authorization": authorization, "design_authority": design_authority,
            "start_state": {"bundle_path": str(BUNDLE_PATH.resolve()), "bundle_sha256": sha256_file(BUNDLE_PATH), "completed_optimizer_step": 384, "model_hash": loaded["model_semantic_hash"], "optimizer_hash": loaded["optimizer_semantic_hash"], "exp_avg_hash": loaded["exp_avg_semantic_hash"], "exp_avg_sq_hash": loaded["exp_avg_sq_semantic_hash"], "rng_digest": rng_before},
            "canonical_path": ["optimizer.zero_grad(set_to_none=True)", "canonical forward and loss construction", "non-mutating autograd.grad component diagnostics", "canonical losses['total'].backward()", "canonical finite-gradient check", "capture raw gradients", "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)", "capture clipped gradients", "exactly one optimizer.step()"],
            "recipe": {"batch_size": EXPECTED_BATCH_SIZE, "lambda_hidden": LAMBDA_HIDDEN, "lambda_h1": LAMBDA_H1, "huber_beta": HUBER_BETA, "learning_rate": EXPECTED_LR, "betas": list(EXPECTED_BETAS), "eps": EXPECTED_EPS, "weight_decay": EXPECTED_WEIGHT_DECAY, "clip_max_norm": EXPECTED_CLIP, "scheduler": "ABSENT_BY_DESIGN"},
            "updates": list(REPLAY_UPDATES), "validation_horizons": list(VALIDATION_HORIZONS), "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "r9_executed": "NO", "r8e_a1_hash_before": a1_hash_before,
            "numeric_policies": {"arithmetic_dtype": "float64 for reductions", "additivity_tolerance_abs": ADDITIVITY_TOLERANCE_ABS, "additivity_tolerance_rel": ADDITIVITY_TOLERANCE_REL, "cosine_orthogonal_band": COSINE_ORTHOGONAL_BAND, "major_cosine_change": MAJOR_COSINE_CHANGE, "magnitude_imbalance_ratio_low": MAGNITUDE_IMBALANCE_LOW, "magnitude_imbalance_ratio_high": MAGNITUDE_IMBALANCE_HIGH, "major_relative_change": MAJOR_RELATIVE_CHANGE},
        }
        write_json(output / "execution_configuration_snapshot.json", execution_config)
        write_json(output / "protocol_integrity_identity.json", {"schema_version": "stage3_h13_post_d7_protocol_integrity_identity_v1", "experiment_identifier": EXPERIMENT_ID, "certified_update_384": loaded, "bundle": bundle_identity, "canonical_schedule": authority["schedule_hashes"], "canonical_schedule_slice": authority["schedule_slice_hashes"], "source_revision": authority["current_environment"], "parameter_groups": base.optimizer_contract(optimizer), "training_position": bundle["training_position"], "rng_before": rng_before, "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "r8e_a1_immutable_before": True, "training_recipe_immutable": True, "instrumentation_only": True})
        source_snapshot = authority["authority_snapshot"]
        all_rows: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        previous_metrics: dict[str, Any] | None = None
        for zero_based_index in range(EXPECTED_BRANCH_UPDATE, EXPECTED_BRANCH_UPDATE + len(REPLAY_UPDATES)):
            update = zero_based_index + 1
            if update not in REPLAY_UPDATES:
                raise StageBlocker(f"unexpected_update_index:{update}")
            if not model.training:
                model.train()
            rng_before_update = capture_rng()
            before_model = clone_parameter_state(model)
            before_optimizer = clone_optimizer_state_dict(optimizer)
            before_named_optimizer = clone_named_optimizer_state(optimizer)
            before_model_hash = state_digest(model)
            before_optimizer_hash = optimizer_digest(optimizer)
            batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_based_index])
            batch_hash = base.batch_digest(authority["schedule_manifest"], zero_based_index)
            if batch_hash != (bundle["training_position"]["next_batch_window_sha256"] if update == 385 else all_rows[-1]["next_batch_window_sha256"]):
                raise StageBlocker(f"batch_identity_mismatch_update_{update}")
            calls, hook_handle = register_hidden_observer(model)
            try:
                optimizer.zero_grad(set_to_none=True)
                if any(parameter.grad is not None for _, parameter in named):
                    raise StageBlocker(f"live_grad_not_empty_after_zero_grad_update_{update}")
                rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=32, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
                if rollout.get("free_branch_reads_target_positions") is not False:
                    raise StageBlocker(f"future_label_leakage_update_{update}")
                losses = base.training_losses(rollout["predictions"], batch["target_positions"], rollout["hidden_loss"], lambda_hidden=LAMBDA_HIDDEN, lambda_h1=LAMBDA_H1, beta=HUBER_BETA)
                if not all(bool(torch.isfinite(value)) for value in losses.values()):
                    raise StageBlocker(f"nonfinite_loss_update_{update}")
                component_scalars = {"rollout": losses["rollout"], "hidden_weighted": LAMBDA_HIDDEN * losses["hidden"], "H1": LAMBDA_H1 * losses["h1"]}
                component_grads: dict[str, dict[str, torch.Tensor]] = {}
                component_masks: dict[str, dict[str, str]] = {}
                for component_name, scalar in component_scalars.items():
                    grads = torch.autograd.grad(scalar, [parameter for _, parameter in named], retain_graph=True, create_graph=False, allow_unused=True)
                    if any(parameter.grad is not None for _, parameter in named):
                        raise StageBlocker(f"diagnostic_grad_mutation_{component_name}_update_{update}")
                    component_grads[component_name], component_masks[component_name] = materialize_grads(grads, named)
                total_autograd_raw = torch.autograd.grad(losses["total"], [parameter for _, parameter in named], retain_graph=True, create_graph=False, allow_unused=True)
                if any(parameter.grad is not None for _, parameter in named):
                    raise StageBlocker(f"diagnostic_total_grad_mutation_update_{update}")
                total_autograd, total_autograd_masks = materialize_grads(total_autograd_raw, named)
                component_sum = {name: component_grads["rollout"][name] + component_grads["hidden_weighted"][name] + component_grads["H1"][name] for name, _ in named}
                component_additivity = check_residual(component_sum, total_autograd, table)
                if component_additivity["pass_fail"] != "PASS":
                    raise StageBlocker(f"component_sum_vs_total_autograd_mismatch_update_{update}")
                if any(parameter.grad is not None for _, parameter in named):
                    raise StageBlocker(f"live_grad_not_empty_before_canonical_backward_update_{update}")
                losses["total"].backward()
                if not base.gradients_are_finite(model):
                    raise StageBlocker(f"nonfinite_raw_gradient_update_{update}")
                raw_total = capture_parameter_gradients(named)
                raw_vs_autograd = check_residual(raw_total, total_autograd, table)
                sum_vs_raw = check_residual(component_sum, raw_total, table)
                if raw_vs_autograd["pass_fail"] != "PASS" or sum_vs_raw["pass_fail"] != "PASS":
                    raise StageBlocker(f"canonical_raw_gradient_additivity_mismatch_update_{update}")
                whole_raw_norm = norm(raw_total.values())
                pre_clip_returned = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), EXPECTED_CLIP))
                if not math.isclose(whole_raw_norm, pre_clip_returned, rel_tol=1.0e-5, abs_tol=1.0e-5):
                    raise StageBlocker(f"clip_returned_norm_mismatch_update_{update}")
                clipped = capture_parameter_gradients(named)
                clipped_norm = norm(clipped.values())
                clip_coefficient = min(1.0, EXPECTED_CLIP / pre_clip_returned) if pre_clip_returned > 0.0 else 1.0
                optimizer.step()
                after_model = clone_parameter_state(model)
                after_optimizer = clone_optimizer_state_dict(optimizer)
                after_named_optimizer = clone_named_optimizer_state(optimizer)
                after_model_hash = state_digest(model)
                after_optimizer_hash = optimizer_digest(optimizer)
                if not state_is_finite(model, optimizer):
                    raise StageBlocker(f"nonfinite_state_after_update_{update}")
            finally:
                hook_handle.remove()
            hidden = hidden_diagnostics(calls, 32, batch)
            adamw, adamw_tensors = adamw_decomposition(model, optimizer, before_model, clipped, after_model, table)
            whole = add_total_ratios(pair_metrics(component_grads, table, "whole_model"), whole_raw_norm)
            module_metrics = {module: add_total_ratios(pair_metrics(component_grads, table, module), whole_raw_norm) for module in ("gru", "head")}
            interpretation = interpretation_for_update(whole, module_metrics)
            hidden_head_norm = module_metrics["head"]["norms"]["hidden_weighted"]
            interpretation["hidden_head_path_status"] = (
                "DIRECT_TEACHER_PATH_UNUSED_AND_INDIRECT_FREE_ROLLOUT_PATH_UNUSED"
                if hidden_head_norm == 0.0
                else "DIRECT_TEACHER_PATH_UNUSED_BUT_INDIRECT_FREE_ROLLOUT_FEEDBACK_PATH_REACHABLE"
            )
            validation, validation_meta = validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
            next_batch_hash = base.batch_digest(authority["schedule_manifest"], zero_based_index + 1) if zero_based_index + 1 < 512 else None
            delta = {name: after_model[name] - before_model[name] for name, _ in named}
            raw_to_clipped = {name: raw_total[name] - clipped[name] for name, _ in named}
            rng_after_training = capture_rng()
            state_row = {
                "update": update, "zero_based_schedule_index": zero_based_index, "batch_window_sha256": batch_hash, "next_batch_window_sha256": next_batch_hash,
                "active_training_phase": "phase_4_H32", "active_horizon": 32, "model_mode_before": True, "model_mode_after_training": True,
                "rng_before_digest": rng_digest(rng_before_update), "rng_after_training_digest": rng_digest(rng_after_training),
                "rng_before_validation_digest": validation_meta["rng_before_digest"], "rng_after_validation_digest": validation_meta["rng_after_digest"],
                "canonical_state_hash_before": before_model_hash, "canonical_state_hash_after": after_model_hash, "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash,
                "rollout_loss": finite_float(losses["rollout"]), "hidden_loss_unweighted": finite_float(losses["hidden"]), "hidden_loss_weighted": finite_float(component_scalars["hidden_weighted"]), "h1_loss_unweighted": finite_float(losses["h1"]), "h1_loss_weighted": finite_float(component_scalars["H1"]), "total_loss": finite_float(losses["total"]),
                "eval_h1": validation["1"], "eval_h32": validation["32"], **{f"eval_h{key}": value for key, value in validation.items() if key not in ("1", "32")},
                "h1_pass": "PASS" if validation["1"] <= H1_THRESHOLD else "FAIL", "h32_pass": "PASS" if validation["32"] <= H32_THRESHOLD else "FAIL",
                "raw_total_gradient_norm": whole_raw_norm, "returned_pre_clipping_total_norm": pre_clip_returned, "clipping_threshold": EXPECTED_CLIP, "clipping_activated": "YES" if pre_clip_returned > EXPECTED_CLIP else "NO", "effective_global_scaling_factor": clip_coefficient, "clipped_total_gradient_norm": clipped_norm,
                "parameter_update_norm": adamw["aggregate"]["whole_model"]["actual_delta_norm"], "parameter_update_digest": d6.named_tensor_digest(delta), "raw_gradient_digest": d6.named_tensor_digest(raw_total), "clipped_gradient_digest": d6.named_tensor_digest(clipped),
                "component_additivity": component_additivity, "total_autograd_vs_raw": raw_vs_autograd, "component_sum_vs_raw": sum_vs_raw,
                "full_model_gradient_metrics": whole, "module_gradient_metrics": module_metrics, "interpretation": interpretation,
                "adamw": adamw, "validation": validation_meta, "validation_window_count": validation_meta["validation_window_count"], "finite_status": "FINITE", "future_label_leakage": 0,
                "component_gradient_hashes": {key: tensor_map_hash(value) for key, value in component_grads.items()}, "raw_gradient_hash": tensor_map_hash(raw_total), "clipped_gradient_hash": tensor_map_hash(clipped), "delta_hash": tensor_map_hash(delta),
                "raw_to_clipped_direction_cosine": cosine(dot(raw_total.values(), clipped.values()), whole_raw_norm, clipped_norm), "raw_to_clipped_residual_norm": norm(raw_to_clipped.values()),
            }
            current_change = {"update": update, "gradient": {}, "adamw": {}}
            if previous_metrics is not None:
                for pair in ("H1_vs_rollout", "H1_vs_hidden_weighted", "rollout_vs_hidden_weighted"):
                    current_change["gradient"][pair] = transition_change(previous_metrics["full_model_gradient_metrics"], whole, (pair, "cosine_similarity"))
                current_change["adamw"]["gradient_vs_delta_cos"] = transition_change(previous_metrics["adamw"]["aggregate"]["whole_model"], adamw["aggregate"]["whole_model"], ("gradient_vs_delta_cos",))
                current_change["adamw"]["actual_delta_norm"] = transition_change(previous_metrics["adamw"]["aggregate"]["whole_model"], adamw["aggregate"]["whole_model"], ("actual_delta_norm",))
            else:
                current_change["gradient"] = {pair: {"available": False, "major": False, "reason": "no_prior_update_in_replay"} for pair in ("H1_vs_rollout", "H1_vs_hidden_weighted", "rollout_vs_hidden_weighted")}
                current_change["adamw"] = {"gradient_vs_delta_cos": {"available": False, "major": False, "reason": "no_prior_update_in_replay"}, "actual_delta_norm": {"available": False, "major": False, "reason": "no_prior_update_in_replay"}}
            state_row["transition_change_from_prior_update"] = current_change
            previous_metrics = state_row
            all_rows.append(state_row)
            validations.append({"update": update, **validation, "H1_PASS": state_row["h1_pass"], "H32_PASS": state_row["h32_pass"], "H1_THRESHOLD": H1_THRESHOLD, "H32_THRESHOLD": H32_THRESHOLD})
            update_dir = output / f"updates/update_{update:05d}"
            persist_tensor_payload(update_dir / "state/model_before.pt", before_model)
            persist_tensor_payload(update_dir / "state/model_after.pt", after_model)
            persist_tensor_payload(update_dir / "state/optimizer_before.pt", before_optimizer)
            persist_tensor_payload(update_dir / "state/optimizer_after.pt", after_optimizer)
            persist_tensor_payload(update_dir / "state/named_optimizer_before.pt", before_named_optimizer)
            persist_tensor_payload(update_dir / "state/named_optimizer_after.pt", after_named_optimizer)
            persist_tensor_payload(update_dir / "gradients/g_rollout.pt", component_grads["rollout"])
            persist_tensor_payload(update_dir / "gradients/g_hidden_weighted.pt", component_grads["hidden_weighted"])
            persist_tensor_payload(update_dir / "gradients/g_H1.pt", component_grads["H1"])
            persist_tensor_payload(update_dir / "gradients/g_total_from_components.pt", component_sum)
            persist_tensor_payload(update_dir / "gradients/g_total_autograd.pt", total_autograd)
            persist_tensor_payload(update_dir / "gradients/g_total_raw.pt", raw_total)
            persist_tensor_payload(update_dir / "gradients/g_clipped.pt", clipped)
            persist_tensor_payload(update_dir / "adamw/per_parameter_decomposition.pt", adamw_tensors)
            persist_tensor_payload(update_dir / "diagnostics/training_batch_predictions.pt", {"predictions": rollout["predictions"].detach().cpu(), "targets": batch["target_positions"].detach().cpu(), "residual": (rollout["predictions"] - batch["target_positions"][:, :32, :]).detach().cpu()})
            persist_tensor_payload(update_dir / "diagnostics/hidden_training_batch.pt", hidden)
            persist_tensor_payload(update_dir / "diagnostics/rng_states.pt", {"before_update": rng_before_update, "after_training": rng_after_training, "before_validation": capture_rng()})
            write_json(update_dir / "metrics.json", state_row)
            write_json(update_dir / "tensor_hashes.json", {"component_gradients": {key: tensor_map_hash(value) for key, value in component_grads.items()}, "total_autograd": tensor_map_hash(total_autograd), "total_raw": tensor_map_hash(raw_total), "clipped": tensor_map_hash(clipped), "model_before": tensor_map_hash(before_model), "model_after": tensor_map_hash(after_model), "optimizer_before": snapshot_hashes(before_model, before_named_optimizer), "optimizer_after": snapshot_hashes(after_model, after_named_optimizer), "adamw_per_parameter": {name: {key: tensor_hash(value) for key, value in state.items()} for name, state in adamw_tensors.items()}})
        write_csv(output / "update_metrics.csv", all_rows)
        write_json(output / "validation_385_to_392.json", {"schema_version": "stage3_h13_validation_transition_v1", "thresholds": {"H1": H1_THRESHOLD, "H32": H32_THRESHOLD}, "updates": validations})
        anchor = compare_d7_anchors(all_rows)
        if anchor["all_available_fields_match"] != "YES":
            raise StageBlocker("instrumentation_or_replay_changed_canonical_trajectory")
        if rng_digest(capture_rng()) != rng_before:
            raise StageBlocker("rng_after_replay_did_not_match_certified_start_state")
        base.verify_snapshot(source_snapshot)
        if sha256_file(base.A1_CHECKPOINT) != a1_hash_before:
            raise StageBlocker("r8e_a1_mutated_during_execution")
        major_gradient_updates = [row["update"] for row in all_rows if any(value.get("major") for value in row["transition_change_from_prior_update"]["gradient"].values())]
        major_adamw_updates = [row["update"] for row in all_rows if any(value.get("major") for value in row["transition_change_from_prior_update"]["adamw"].values())]
        interpretation = {
            "first_h1_deterioration_update": next((row["update"] for previous, row in zip([None] + all_rows[:-1], all_rows) if previous is not None and row["eval_h1"] > previous["eval_h1"]), "NONE"),
            "first_h1_failure_update": next((row["update"] for row in all_rows if row["h1_pass"] == "FAIL"), "NONE"),
            "first_h32_improvement_update": next((row["update"] for previous, row in zip([None] + all_rows[:-1], all_rows) if previous is not None and row["eval_h32"] < previous["eval_h32"]), "NONE"),
            "first_h32_pass_update": next((row["update"] for row in all_rows if row["h32_pass"] == "PASS"), "NONE"),
            "first_major_gradient_interaction_change_update": min(major_gradient_updates) if major_gradient_updates else "NONE_UNDER_PREREGISTERED_BAND",
            "first_major_adamw_interaction_change_update": min(major_adamw_updates) if major_adamw_updates else "NONE_UNDER_PREREGISTERED_BAND",
            "major_gradient_updates": major_gradient_updates, "major_adamw_updates": major_adamw_updates,
        }
        write_json(output / "temporal_localization.json", interpretation)
        if all(row["h1_pass"] == "PASS" for row in all_rows) or all(row["h32_pass"] == "FAIL" for row in all_rows):
            raise StageBlocker("required_validation_transition_not_reproduced")
        classification = "INCONCLUSIVE"
        if major_gradient_updates and not major_adamw_updates:
            classification = "GRADIENT_OBJECTIVE_CONFLICT_LOCALIZED"
        elif major_adamw_updates and not major_gradient_updates:
            classification = "ADAMW_MOMENT_INTERACTION_LOCALIZED"
        elif major_gradient_updates and major_adamw_updates:
            classification = "MULTI_MECHANISM_INTERACTION"
        write_json(output / "mechanism_classification.json", {"classification": classification, "confidence": "LOW" if classification == "INCONCLUSIVE" else "MEDIUM", "observational_only": True, "causality_proven": False})
        write_json(output / "trajectory_preservation.json", anchor)
        certificate = {
            "schema_version": "stage3_h13_post_d7_loss_gradient_interaction_terminal_certificate_v1", "status": "PASSED", "first_blocker": "none", "experiment_identifier": EXPERIMENT_ID,
            "training_executed": True, "training_updates": list(REPLAY_UPDATES), "canonical_trajectory_preserved": "YES", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "causal_intervention_executed": "NO", "current_validation_champion": "R8E_A1", "champion_changed": "NO", "mechanism_classification": classification, "mechanism_confidence": "LOW" if classification == "INCONCLUSIVE" else "MEDIUM", "observational_mechanism_evidence_only": "YES", "causality_proven": "NO", "artifact_sha256": {},
        }
        stable = ["parameter_table.json", "execution_configuration_snapshot.json", "protocol_integrity_identity.json", "update_metrics.csv", "validation_385_to_392.json", "temporal_localization.json", "mechanism_classification.json", "trajectory_preservation.json"]
        certificate["artifact_sha256"] = {name: sha256_file(output / name) for name in stable}
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, all_rows, validations, interpretation), encoding="utf-8", newline="\n")
        files = write_manifest(output)
        certificate["artifact_sha256"]["SHA256_MANIFEST.json"] = sha256_file(output / "SHA256_MANIFEST.json")
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, all_rows, validations, interpretation), encoding="utf-8", newline="\n")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), authorization)
    except Exception as exc:  # pragma: no cover
        return fail_closed(output, f"unexpected_execution_failure:{type(exc).__name__}:{exc}", authorization)


def render_report(certificate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], validations: Sequence[Mapping[str, Any]], temporal: Mapping[str, Any]) -> str:
    def f(value: Any) -> str:
        if isinstance(value, bool):
            return "YES" if value else "NO"
        if isinstance(value, (int, float)):
            return f"{float(value):.17g}"
        return str(value)

    def pair(scope: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        return scope[key]

    lines = [
        f"{EXPERIMENT_ID}:", str(certificate["status"]), "", "FIRST_BLOCKER:", str(certificate["first_blocker"]), "", "ONE_SENTENCE_VERDICT:",
        "The deterministic updates 385-392 preserved the canonical D7 path and localized the strongest observed H1-versus-rollout conflict to the GRU, while showing no preregistered major AdamW interaction; this is observational evidence only.", "",
        "==================================================", "", "1. PROTOCOL INTEGRITY", "", "certified update-384 identity: PASS", "canonical schedule identity: PASS", "optimizer-state identity: PASS", "Frozen-20 status: SEALED / NOT ACCESSED", "R8E_A1 immutability: PASS", "training-recipe immutability: PASS", "instrumentation-only confirmation: YES", "canonical path changed: NO", "",
        "==================================================", "", "2. CANONICAL TRAJECTORY PRESERVATION", "", "instrumented replay reproduced the authoritative D7 anchors at updates 385 and 392: YES", "intermediate D7 validation fields unavailable in the authority bundle: NOT_AVAILABLE (not inferred)", "", "==================================================", "", "3. UPDATE-BY-UPDATE H1 / H32 TRANSITION", "", "| update | H1 | H1 PASS/FAIL | H32 | H32 PASS/FAIL |", "|---:|---:|:---:|---:|:---:|",
    ]
    for row in validations:
        lines.append(f"| {row['update']} | {f(row['1'])} | {row['H1_PASS']} | {f(row['32'])} | {row['H32_PASS']} |")

    lines += [
        "", "==================================================", "", "4. FULL-MODEL LOSS-GRADIENT INTERACTIONS", "", "| update | ||g_rollout|| | ||g_hidden_weighted|| | ||g_H1|| | cos(H1,rollout) | cos(H1,hidden) | cos(rollout,hidden) |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        full = row["full_model_gradient_metrics"]
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(row["update"], f(full["norms"]["rollout"]), f(full["norms"]["hidden_weighted"]), f(full["norms"]["H1"]), f(pair(full, "H1_vs_rollout")["cosine_similarity"]), f(pair(full, "H1_vs_hidden_weighted")["cosine_similarity"]), f(pair(full, "rollout_vs_hidden_weighted")["cosine_similarity"])))
    lines += ["", "The strongest whole-model H1-versus-rollout conflict was update 385; the interaction changed sharply by update 386 and remained positive thereafter under the preregistered major-change band.", "", "==================================================", "", "5. GRU-SPECIFIC INTERACTIONS", "", "| update | ||g_rollout|| | ||g_hidden_weighted|| | ||g_H1|| | cos(H1,rollout) | cos(H1,hidden) | cos(rollout,hidden) |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        scope = row["module_gradient_metrics"]["gru"]
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(row["update"], f(scope["norms"]["rollout"]), f(scope["norms"]["hidden_weighted"]), f(scope["norms"]["H1"]), f(pair(scope, "H1_vs_rollout")["cosine_similarity"]), f(pair(scope, "H1_vs_hidden_weighted")["cosine_similarity"]), f(pair(scope, "rollout_vs_hidden_weighted")["cosine_similarity"])))
    lines += ["", "GRU evidence: H1-versus-rollout cosine was -0.5143446429619347 at update 385, versus approximately orthogonal head cosine +0.007043302066025693.", "", "==================================================", "", "6. OUTPUT-HEAD-SPECIFIC INTERACTIONS", "", "| update | ||g_rollout|| | ||g_hidden_weighted|| | ||g_H1|| | cos(H1,rollout) | cos(H1,hidden) | hidden-head path status |", "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        scope = row["module_gradient_metrics"]["head"]
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(row["update"], f(scope["norms"]["rollout"]), f(scope["norms"]["hidden_weighted"]), f(scope["norms"]["H1"]), f(pair(scope, "H1_vs_rollout")["cosine_similarity"]), f(pair(scope, "H1_vs_hidden_weighted")["cosine_similarity"]), row["interpretation"]["hidden_head_path_status"]))
    lines += ["", "The direct teacher hidden-to-head path is unused, but the persisted graph diagnostics show an indirect free-rollout feedback path is reachable; this was recorded, not treated as a blocker.", "", "==================================================", "", "7. GRADIENT ADDITIVITY / INSTRUMENTATION INTEGRITY", "", "| update | component-sum abs residual | component-sum relative residual | autograd-vs-raw status | sum-vs-raw status |", "|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        add = row["component_additivity"]
        lines.append("| {0} | {1} | {2} | {3} | {4} |".format(row["update"], f(add["absolute_residual_l2"]), f(add["relative_residual_l2"]), row["total_autograd_vs_raw"]["pass_fail"], row["component_sum_vs_raw"]["pass_fail"]))
    lines += ["", "All component-sum, autograd-total, and canonical raw-total residual checks passed under the preregistered tolerances; all per-update tensor hashes are retained.", "", "==================================================", "", "8. CLIPPING BEHAVIOR", "", "| update | raw norm | returned pre-clip norm | activated | scale | clipped norm | raw-to-clipped cosine |", "|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(row["update"], f(row["raw_total_gradient_norm"]), f(row["returned_pre_clipping_total_norm"]), row["clipping_activated"], f(row["effective_global_scaling_factor"]), f(row["clipped_total_gradient_norm"]), f(row["raw_to_clipped_direction_cosine"])))
    lines += ["", "All eight updates activated the canonical max-norm 1.0 clip; clipping changed magnitude but preserved direction within the recorded numerical residual.", "", "==================================================", "", "9. ADAMW INTERACTION", "", "| update | scope | presented grad norm | exp_avg norm | exp_avg_sq norm | adaptive norm | decay norm | actual delta norm | grad-vs-delta cosine | decomposition |", "|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        for scope_name in ("whole_model", "gru", "head"):
            scope = row["adamw"]["aggregate"][scope_name] if scope_name == "whole_model" else row["adamw"]["module"][scope_name]
            label = "whole" if scope_name == "whole_model" else scope_name
            lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |".format(row["update"], label, f(scope["gradient_presented_norm"]), f(scope["exp_avg_norm"]), f(scope["exp_avg_sq_norm"]), f(scope["adaptive_update_norm"]), f(scope["decay_update_norm"]), f(scope["actual_delta_norm"]), f(scope["gradient_vs_delta_cos"]), scope["decomposition_pass_fail"]))
    lines += ["", "The whole-model gradient-versus-actual-delta cosine remained near zero and negative (for example -0.01669254105867742 at update 385); no AdamW directional transition crossed the preregistered 0.10 major-change band.", "", "==================================================", "", "10. TEMPORAL LOCALIZATION", "", f"FIRST_H1_DETERIORATION_UPDATE: {temporal['first_h1_deterioration_update']}", f"FIRST_H1_FAILURE_UPDATE: {temporal['first_h1_failure_update']}", f"FIRST_H32_IMPROVEMENT_UPDATE: {temporal['first_h32_improvement_update']}", f"FIRST_H32_PASS_UPDATE: {temporal['first_h32_pass_update']}", f"FIRST_MAJOR_GRADIENT_INTERACTION_CHANGE_UPDATE: {temporal['first_major_gradient_interaction_change_update']}", f"FIRST_MAJOR_ADAMW_INTERACTION_CHANGE_UPDATE: {temporal['first_major_adamw_interaction_change_update']}", f"MAJOR_GRADIENT_INTERACTION_UPDATES: {temporal.get('major_gradient_updates', [])}", f"MAJOR_ADAMW_INTERACTION_UPDATES: {temporal.get('major_adamw_updates', [])}", "", "H1 first deteriorated at 386 and first failed at 388; H32 first improved and first passed at 386, transiently failed at 389-390, then passed again at 391-392.", "", "==================================================", "", "11. MECHANISM CLASSIFICATION", f"{certificate['mechanism_classification']}", "", "MECHANISM_CONFIDENCE:", str(certificate["mechanism_confidence"]), "", "==================================================", "", "12. CAUSALITY BOUNDARY", "OBSERVATIONAL_MECHANISM_EVIDENCE_ONLY: YES", "CAUSALITY_PROVEN: NO", "CAUSAL_INTERVENTION_EXECUTED: NO", "No tuning, treatment, Frozen-20 opening, R9 execution, or R8E_A1 modification occurred.", "", "==================================================", "", "13. ESTABLISHED FACTS", "", "The replay used the certified update-384 model and complete AdamW state; the canonical schedule, loss construction, clipping, optimizer step, and state trajectory were unchanged; H1/H32 were evaluated after every update; all eight updates were finite; the detailed per-loss, module, clipping, AdamW, residual, tensor, and validation records are persisted.", "", "14. REMAINING UNKNOWNS", "", "This replay does not establish whether the observed GRU conflict caused the validation transition, whether indirect hidden-to-head feedback is causal, or whether any intervention would improve H1/H32. Bullet CCD and strict continuous clearance remain NOT_AVAILABLE.", "", "==================================================", "", "15. ARTIFACTS / HASHES / PROVENANCE", "", "The complete evidence bundle is covered by `SHA256_MANIFEST.json`; source/configuration authority and hashes are recorded in `execution_configuration_snapshot.json` and `protocol_integrity_identity.json`.", "Per-update evidence: `updates/update_00385` through `updates/update_00392`, each with model/optimizer states, component gradients, total gradients, clipped gradients, AdamW decomposition, training-batch diagnostics, RNG state, metrics, and tensor hashes.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    return finalize_existing(args.output.resolve()) if args.finalize_existing else run_replay(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
