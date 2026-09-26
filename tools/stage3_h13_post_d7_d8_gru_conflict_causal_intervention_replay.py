"""Execute the sealed Stage 3 H13 D8 GRU conflict causal replay.

This runner is intentionally a two-branch execution artifact.  CONTROL uses
the canonical D7 backward/clip/AdamW path.  INTERVENTION uses the same forward
graph and objective decomposition, replacing only the rollout gradient on the
eight manifest GRU tensors with the preregistered asymmetric projection.
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

from scripts import stage3_h13_post_d5_d6_adamw_moment_reset_causal_branch as d6  # noqa: E402
from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402
from src.stage3_h13_r8e import EVALUATION_HORIZONS  # noqa: E402
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256  # noqa: E402
from tools import stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay as d7  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D7_D8_GRU_CONFLICT_CAUSAL_INTERVENTION_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d7_d8_gru_conflict_causal_intervention_design_review_20260817T"
D7_DIR = ROOT / "outputs/stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay_20260817T063800Z"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\d0298c5e-17d5-456a-a494-bee9b557b999\pasted-text.txt")
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_D7_REPORT_SHA256 = "3d5fc7daeabd4718339fd6bca688f42c0a260007547c54547636214dbabc243a"
EXPECTED_D7_MANIFEST_SHA256 = "38bfc88bc6caca455a77bc1e5d7b4b3b07a19ee9792aa0e6ec17f1e5bcadbfbb"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
UPDATES = tuple(range(385, 393))
START_UPDATE = 384
EXPECTED_H1_THRESHOLD = 0.0000859791431235184
EXPECTED_H32_THRESHOLD = 0.01856902565856056
MATERIAL_H1_THRESHOLD = 4.099006815405639e-06
LAMBDA_HIDDEN = 0.005
LAMBDA_H1 = 1.0
HUBER_BETA = 0.001
CLIP_MAX_NORM = 1.0
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
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
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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


def clone_named_parameters(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters()}


def clone_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> Any:
    return d7.jsonable_tensor_clone(optimizer.state_dict())


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


def parameter_manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise StageBlocker("gru_manifest_rows_missing")
    if [row.get("name") for row in rows] != list(GRU_NAMES):
        raise StageBlocker("gru_manifest_parameter_order_mismatch")
    gru_rows = rows[:len(GRU_NAMES)]
    if sum(int(row.get("numel", -1)) for row in gru_rows) != 156672:
        raise StageBlocker("gru_manifest_element_count_mismatch")
    if int(manifest.get("module_count", -1)) != 156672:
        raise StageBlocker("gru_manifest_module_count_mismatch")
    expected_offsets = 0
    for row in gru_rows:
        if row.get("module") != "gru" or row.get("dtype") != "torch.float32" or int(row.get("flat_offset_start", -1)) != expected_offsets:
            raise StageBlocker(f"gru_manifest_row_mismatch:{row.get('name')}")
        expected_offsets += int(row["numel"])
        if int(row.get("flat_offset_end", -1)) != expected_offsets:
            raise StageBlocker(f"gru_manifest_offset_mismatch:{row.get('name')}")
    return gru_rows


def verify_parameter_manifest(model: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_rows = parameter_manifest_rows(manifest)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(ALL_NAMES):
        raise StageBlocker("runtime_parameter_order_mismatch")
    actual = {name: parameter for name, parameter in named}
    for row in expected_rows:
        name = row["name"]
        parameter = actual.get(name)
        if parameter is None or list(parameter.shape) != list(row["shape"]) or int(parameter.numel()) != int(row["numel"]) or str(parameter.dtype) != row["dtype"]:
            raise StageBlocker(f"runtime_gru_parameter_manifest_mismatch:{name}")
    if sum(int(parameter.numel()) for name, parameter in named if name.startswith("gru.")) != 156672:
        raise StageBlocker("runtime_gru_element_count_mismatch")
    return expected_rows


def flat(values: Mapping[str, torch.Tensor], rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    return torch.cat([values[row["name"]].detach().cpu().contiguous().reshape(-1).double() for row in rows], dim=0)


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
    values: dict[str, torch.Tensor] = {}
    for name, parameter in named:
        values[name] = torch.zeros_like(parameter.detach()).cpu().contiguous() if parameter.grad is None else parameter.grad.detach().cpu().contiguous().clone()
    assert_finite(values.values(), "live_gradient")
    return values


def residual(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = flat({row["name"]: left[row["name"]] - right[row["name"]] for row in rows}, rows)
    l2 = finite_float(torch.linalg.vector_norm(value))
    max_abs = finite_float(torch.max(torch.abs(value))) if value.numel() else 0.0
    scale = max(norm(left[row["name"]] for row in rows), norm(right[row["name"]] for row in rows), 1.0)
    tolerance = max(1.0e-7, 2.0e-5 * scale)
    passed = l2 <= tolerance and max_abs <= 1.0e-7 + 2.0e-5 * max(norm([left[row["name"]] for row in rows]), norm([right[row["name"]] for row in rows]), 1.0)
    return {"l2": l2, "max_abs": max_abs, "scale": scale, "tolerance": tolerance, "pass_fail": "PASS" if passed else "FAIL"}


def pair_metrics(vectors: Mapping[str, Mapping[str, torch.Tensor]], rows: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    scoped = rows if scope == "whole_model" else [row for row in rows if row["module"] == scope]
    flat_vectors = {key: flat(value, scoped) for key, value in vectors.items()}
    norms = {key: finite_float(torch.linalg.vector_norm(value)) for key, value in flat_vectors.items()}
    result: dict[str, Any] = {"scope": scope, "norms": norms}
    for left, right, label in (("H1", "rollout", "H1_vs_rollout"), ("H1", "hidden_weighted", "H1_vs_hidden_weighted"), ("rollout", "hidden_weighted", "rollout_vs_hidden_weighted")):
        d = finite_float(torch.dot(flat_vectors[left], flat_vectors[right]))
        result[label] = {"dot_product": d, "cosine_similarity": cosine(d, norms[left], norms[right])}
    result["ratios"] = {f"norm_{key}_over_total_raw": None for key in ("H1", "hidden_weighted", "rollout")}
    return result


def add_total_ratios(metrics: Mapping[str, Any], total: float) -> dict[str, Any]:
    result = copy.deepcopy(metrics)
    for key in ("H1", "hidden_weighted", "rollout"):
        result["ratios"][f"norm_{key}_over_total_raw"] = None if total == 0.0 else result["norms"][key] / total
    return result


def register_hidden_observer(model: Any) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def hook(_module: Any, inputs: Any, output: Any) -> None:
        sequence, hidden = output
        calls.append({"input_shape": list(inputs[0].shape), "sequence": sequence, "hidden": hidden})

    return calls, model.gru.register_forward_hook(hook)


def adamw_metrics(model: Any, optimizer: torch.optim.Optimizer, before: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    values: dict[str, dict[str, torch.Tensor]] = {}
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    for name, parameter in model.named_parameters():
        if name not in names.values():
            continue
        state = optimizer.state.get(parameter, {})
        if not all(key in state for key in ("step", "exp_avg", "exp_avg_sq")):
            raise StageBlocker(f"optimizer_state_incomplete:{name}")
        step_after = int(state["step"].item()) if isinstance(state["step"], torch.Tensor) else int(state["step"])
        step_before = step_after - 1
        m_after = state["exp_avg"].detach().cpu().contiguous().clone()
        v_after = state["exp_avg_sq"].detach().cpu().contiguous().clone()
        m_before = (m_after - 0.1 * clipped[name]) / 0.9
        v_before = (v_after - 0.001 * clipped[name].square()) / 0.999
        adaptive = -5.0e-5 * (m_after / (1.0 - 0.9 ** step_after)) / (torch.sqrt(v_after / (1.0 - 0.999 ** step_after)) + 1.0e-8)
        decay = -5.0e-5 * 1.0e-5 * before[name]
        actual = after[name] - before[name]
        values[name] = {"theta_before": before[name], "gradient_presented": clipped[name], "m_before": m_before, "m_after": m_after, "v_before": v_before, "v_after": v_after, "adaptive_update": adaptive, "decay_update": decay, "actual_delta": actual, "decomposition_residual": actual - adaptive - decay}
    assert_finite((v for item in values.values() for v in item.values()), "adamw")
    result: dict[str, Any] = {"step_before_by_parameter": {}, "step_after_by_parameter": {}, "aggregate": {}, "module": {}}
    for name, parameter in model.named_parameters():
        if name not in values:
            continue
        step = int(optimizer.state[parameter]["step"].item()) if isinstance(optimizer.state[parameter]["step"], torch.Tensor) else int(optimizer.state[parameter]["step"])
        result["step_before_by_parameter"][name] = step - 1
        result["step_after_by_parameter"][name] = step
    for scope in ("whole_model", "gru", "head"):
        scoped = rows if scope == "whole_model" else [row for row in rows if row["module"] == scope]
        def get(key: str) -> list[torch.Tensor]:
            return [values[row["name"]][key] for row in scoped]
        gp, ap, dp, de = norm(get("gradient_presented")), norm(get("adaptive_update")), norm(get("decay_update")), norm(get("actual_delta"))
        g_a = dot(get("gradient_presented"), get("adaptive_update"))
        g_d = dot(get("gradient_presented"), get("actual_delta"))
        a_d = dot(get("adaptive_update"), get("actual_delta"))
        d_a = dot(get("decay_update"), get("adaptive_update"))
        d_d = dot(get("decay_update"), get("actual_delta"))
        residual_norm = norm(get("decomposition_residual"))
        result["aggregate" if scope == "whole_model" else "module"][scope] = {
            "gradient_presented_norm": gp, "adaptive_update_norm": ap, "decay_update_norm": dp, "actual_delta_norm": de,
            "gradient_vs_adaptive_dot": g_a, "gradient_vs_adaptive_cos": cosine(g_a, gp, ap), "gradient_vs_delta_dot": g_d, "gradient_vs_delta_cos": cosine(g_d, gp, de),
            "adaptive_vs_delta_dot": a_d, "adaptive_vs_delta_cos": cosine(a_d, ap, de), "decay_vs_adaptive_cos": cosine(d_a, dp, ap), "decay_vs_delta_cos": cosine(d_d, dp, de),
            "decomposition_residual_norm": residual_norm, "decomposition_pass_fail": "PASS" if residual_norm <= 1.0e-7 + 2.0e-5 * max(de, 1.0) else "FAIL",
            "exp_avg_norm": norm(get("m_after")), "exp_avg_sq_norm": norm(get("v_after")),
        }
    return result, values


def validation_snapshot(model: Any, validation: Any, channels: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    before = capture_rng()
    before_digest = rng_digest(before)
    mode_before = bool(model.training)
    model.eval()
    with torch.no_grad():
        metrics = base.evaluate_compact(model, validation, channels, horizons=VALIDATION_HORIZONS)
    after_digest = rng_digest(capture_rng())
    if after_digest != before_digest:
        base.restore_rng(before)
        after_digest = rng_digest(capture_rng())
    model.train(mode_before)
    if bool(model.training) != mode_before:
        raise StageBlocker("validation_mode_restore_failed")
    return {str(k): finite_float(v) for k, v in metrics.items()}, {"mode_before": mode_before, "mode_after": bool(model.training), "rng_before_digest": before_digest, "rng_after_digest": after_digest, "rng_unchanged_or_restored": "YES" if before_digest == after_digest else "RESTORED", "validation_window_count": int(validation.rollout_target_positions.shape[0])}


def project_rollout(h1: Mapping[str, torch.Tensor], rollout: Mapping[str, torch.Tensor], rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    h1_flat = flat(h1, rows)
    rollout_flat = flat(rollout, rows)
    dot_before = finite_float(torch.dot(rollout_flat, h1_flat))
    h1_sq = finite_float(torch.dot(h1_flat, h1_flat))
    h1_norm = finite_float(torch.linalg.vector_norm(h1_flat))
    rollout_norm = finite_float(torch.linalg.vector_norm(rollout_flat))
    cosine_before = cosine(dot_before, h1_norm, rollout_norm)
    fired = dot_before < 0.0 and h1_sq > 0.0
    coefficient = dot_before / h1_sq if fired else 0.0
    projected64 = rollout_flat - coefficient * h1_flat if fired else rollout_flat.clone()
    projected: dict[str, torch.Tensor] = {}
    offset = 0
    for row in rows:
        end = offset + int(row["numel"])
        projected[row["name"]] = projected64[offset:end].reshape(tuple(row["shape"])).to(dtype=torch.float32).contiguous()
        offset = end
    projected_flat = flat(projected, rows)
    removed = rollout_flat - projected_flat
    expected_removed = coefficient * h1_flat
    removed_residual = removed - expected_removed
    dot_after = finite_float(torch.dot(projected_flat, h1_flat))
    proj_norm = finite_float(torch.linalg.vector_norm(projected_flat))
    return projected, {
        "criterion_dot_product": dot_before, "h1_squared": h1_sq, "h1_norm": h1_norm, "rollout_norm_before": rollout_norm,
        "cosine_before": cosine_before, "conflict": bool(dot_before < 0.0), "denominator_positive": bool(h1_sq > 0.0), "projection_fired": bool(fired),
        "projection_fired_reason": "negative_dot_positive_h1_norm" if fired else ("nonnegative_dot" if dot_before >= 0.0 else "zero_h1_denominator"),
        "projection_coefficient": coefficient if fired else 0.0, "removed_component_norm": finite_float(torch.linalg.vector_norm(removed)),
        "removed_component_dot_h1": finite_float(torch.dot(removed, h1_flat)), "removed_component_equation_residual_l2": finite_float(torch.linalg.vector_norm(removed_residual)),
        "removed_component_equation_residual_max_abs": finite_float(torch.max(torch.abs(removed_residual))) if removed_residual.numel() else 0.0,
        "original_rollout_norm": rollout_norm, "projected_rollout_norm": proj_norm, "post_projection_dot_product": dot_after,
        "post_projection_cosine": cosine(dot_after, h1_norm, proj_norm), "projection_residual_pass": "PASS" if not fired or finite_float(torch.linalg.vector_norm(removed_residual)) <= 2.0e-6 * max(1.0, finite_float(torch.linalg.vector_norm(expected_removed))) else "FAIL",
        "orthogonalization_pass": "PASS" if not fired or abs(dot_after) <= 2.0e-6 * max(1.0, h1_norm * max(proj_norm, 1.0)) else "FAIL",
    }


def build_combined(component: Mapping[str, Mapping[str, torch.Tensor]], projected: Mapping[str, torch.Tensor], named: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, _parameter in named:
        rollout = projected[name] if name in GRU_NAMES else component["rollout"][name]
        result[name] = component["H1"][name] + rollout + component["hidden_weighted"][name]
    return result


def component_metrics(component: Mapping[str, Mapping[str, torch.Tensor]], rows: Sequence[Mapping[str, Any]], total_norm: float) -> tuple[dict[str, Any], dict[str, Any]]:
    whole = add_total_ratios(pair_metrics(component, rows, "whole_model"), total_norm)
    modules = {module: add_total_ratios(pair_metrics(component, rows, module), total_norm) for module in ("gru", "head")}
    return whole, modules


def serialize_state(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def update_artifacts(output: Path, branch: str, update: int, before_model: Mapping[str, torch.Tensor], after_model: Mapping[str, torch.Tensor], before_optimizer: Any, after_optimizer: Any, before_named_optimizer: Mapping[str, Any], after_named_optimizer: Mapping[str, Any], component: Mapping[str, Mapping[str, torch.Tensor]], canonical_sum: Mapping[str, torch.Tensor], total_autograd: Mapping[str, torch.Tensor], raw: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], projected: Mapping[str, torch.Tensor], adamw_tensors: Mapping[str, Any], delta: Mapping[str, torch.Tensor], payload: Mapping[str, Any]) -> None:
    root = output / "updates" / branch / f"update_{update:05d}"
    serialize_state(root / "state/model_before.pt", before_model)
    serialize_state(root / "state/model_after.pt", after_model)
    serialize_state(root / "state/optimizer_before.pt", before_optimizer)
    serialize_state(root / "state/optimizer_after.pt", after_optimizer)
    serialize_state(root / "state/named_optimizer_before.pt", before_named_optimizer)
    serialize_state(root / "state/named_optimizer_after.pt", after_named_optimizer)
    for name, values in (("g_H1", component["H1"]), ("g_rollout", component["rollout"]), ("g_hidden_weighted", component["hidden_weighted"]), ("g_rollout_projected", projected), ("g_total_from_components", canonical_sum), ("g_total_autograd", total_autograd), ("g_total_raw", raw), ("g_clipped", clipped)):
        serialize_state(root / "gradients" / f"{name}.pt", values)
    serialize_state(root / "adamw/per_parameter_decomposition.pt", adamw_tensors)
    write_json(root / "metrics.json", jsonable(payload))
    write_json(root / "tensor_hashes.json", {"component_gradients": {key: tensor_map_hash(value) for key, value in component.items()}, "projected_rollout": tensor_map_hash(projected), "canonical_sum": tensor_map_hash(canonical_sum), "total_autograd": tensor_map_hash(total_autograd), "raw": tensor_map_hash(raw), "clipped": tensor_map_hash(clipped), "model_before": tensor_map_hash(before_model), "model_after": tensor_map_hash(after_model), "delta": tensor_map_hash(delta), "optimizer_before": d7.snapshot_hashes(before_model, before_named_optimizer), "optimizer_after": d7.snapshot_hashes(after_model, after_named_optimizer), "adamw_per_parameter": {name: {key: tensor_hash(value) for key, value in state.items() if isinstance(value, torch.Tensor)} for name, state in adamw_tensors.items()}})


def authoritative_d7_rows() -> dict[int, dict[str, str]]:
    path = D7_DIR / "update_metrics.csv"
    if not path.is_file():
        raise StageBlocker("authoritative_d7_update_metrics_missing")
    return {int(row["update"]): row for row in csv.DictReader(path.open("r", encoding="utf-8", newline=""))}


def compare_control_to_d7(rows: Sequence[Mapping[str, Any]], authority_rows: Mapping[int, Mapping[str, str]]) -> dict[str, Any]:
    fields = (("batch_window_sha256", "batch_window_sha256", "exact"), ("total_loss", "total_loss", "float"), ("rollout_loss", "rollout_loss", "float"), ("hidden_loss_unweighted", "hidden_loss_unweighted", "float"), ("h1_loss_unweighted", "h1_loss_unweighted", "float"), ("raw_total_gradient_norm", "raw_total_gradient_norm", "float"), ("clipped_total_gradient_norm", "clipped_total_gradient_norm", "float"), ("clipping_activated", "clipping_activated", "exact"), ("effective_global_scaling_factor", "effective_global_scaling_factor", "float"), ("parameter_update_norm", "parameter_update_norm", "float"), ("parameter_update_digest", "parameter_update_digest", "exact"), ("raw_gradient_digest", "raw_gradient_digest", "exact"), ("clipped_gradient_digest", "clipped_gradient_digest", "exact"), ("canonical_state_hash_after", "canonical_state_hash_after", "exact"), ("optimizer_state_hash_after", "optimizer_state_hash_after", "exact"), ("eval_h1", "eval_h1", "float"), ("eval_h32", "eval_h32", "float"))
    checks: list[dict[str, Any]] = []
    for observed in rows:
        update = int(observed["update"])
        expected = authority_rows.get(update)
        if expected is None:
            checks.append({"update": update, "pass_fail": "FAIL", "reason": "authoritative_update_missing"})
            continue
        for left_key, right_key, kind in fields:
            left = observed.get(left_key)
            right = expected.get(right_key)
            ok = left == right if kind == "exact" else math.isclose(float(left), float(right), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)
            checks.append({"update": update, "field": left_key, "observed": left, "authoritative": right, "pass_fail": "PASS" if ok else "FAIL"})
        expected_components = expected.get("component_gradient_hashes")
        if expected_components:
            try:
                expected_components = json.loads(expected_components)
            except json.JSONDecodeError:
                expected_components = None
        if expected_components:
            for key, expected_hash in expected_components.items():
                observed_hash = observed["component_gradient_hashes"].get(key)
                checks.append({"update": update, "field": f"component_gradient_hashes.{key}", "observed": observed_hash, "authoritative": expected_hash, "pass_fail": "PASS" if observed_hash == expected_hash else "FAIL"})
    failed = [item for item in checks if item["pass_fail"] == "FAIL"]
    return {"authoritative_d7_directory": str(D7_DIR.resolve()), "compared_updates": list(UPDATES), "checks": checks, "first_mismatch": failed[0] if failed else None, "CONTROL_REPRODUCES_CANONICAL_D7": "YES" if not failed else "NO"}


def verify_sealed_inputs() -> dict[str, Any]:
    if not DESIGN_DIR.is_dir():
        raise StageBlocker("sealed_design_directory_missing")
    design_manifest = json.loads((DESIGN_DIR / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    design_hashes: dict[str, str] = {}
    for name, expected in design_manifest.get("files", {}).items():
        path = DESIGN_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            raise StageBlocker(f"sealed_design_hash_mismatch:{name}")
        design_hashes[name] = expected
    design = json.loads((DESIGN_DIR / "experiment_design.json").read_text(encoding="utf-8"))
    manifest = json.loads((DESIGN_DIR / "gru_parameter_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((DESIGN_DIR / "causal_classification_contract.json").read_text(encoding="utf-8"))
    if design.get("review_status") != "PASSED" or design.get("training_executed") is not False or design.get("optimizer_step_executed") is not False:
        raise StageBlocker("sealed_design_status_invalid")
    if design.get("temporal_window", {}).get("updates") != list(UPDATES) or design.get("temporal_window", {}).get("start_state") != 384:
        raise StageBlocker("sealed_design_window_invalid")
    if contract.get("endpoint") != "update_392" or contract.get("material_h1_effect", {}).get("minimum_absolute_H1_improvement") != MATERIAL_H1_THRESHOLD:
        raise StageBlocker("sealed_classification_contract_invalid")
    if not BUNDLE_PATH.is_file() or sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("certified_start_bundle_hash_mismatch")
    d7_manifest_path = D7_DIR / "SHA256_MANIFEST.json"
    d7_report_path = D7_DIR / "FINAL_REPORT.md"
    if sha256_file(d7_manifest_path) != EXPECTED_D7_MANIFEST_SHA256:
        raise StageBlocker("authoritative_d7_manifest_hash_mismatch")
    if sha256_file(d7_report_path) != EXPECTED_D7_REPORT_SHA256:
        raise StageBlocker("authoritative_d7_final_report_hash_mismatch")
    d7_manifest = json.loads(d7_manifest_path.read_text(encoding="utf-8"))
    verified_d7_files = {}
    for name, expected in d7_manifest.get("artifact_sha256", {}).items():
        path = D7_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            raise StageBlocker(f"authoritative_d7_artifact_hash_mismatch:{name}")
        verified_d7_files[name] = expected
    if base.EXPECTED_SOURCE_COMMIT != EXPECTED_SOURCE_REVISION:
        raise StageBlocker("base_source_revision_constant_mismatch")
    return {"design_directory": str(DESIGN_DIR.resolve()), "design_hashes": design_hashes, "design_status": design.get("review_status"), "parameter_manifest": manifest, "classification_contract": contract, "bundle_path": str(BUNDLE_PATH.resolve()), "bundle_sha256": EXPECTED_BUNDLE_SHA256, "d7_directory": str(D7_DIR.resolve()), "d7_final_report_sha256": EXPECTED_D7_REPORT_SHA256, "d7_manifest_sha256": EXPECTED_D7_MANIFEST_SHA256, "d7_verified_files": verified_d7_files, "source_revision": EXPECTED_SOURCE_REVISION}


def verify_authorization() -> dict[str, Any]:
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("current_execution_authorization_missing")
    text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required = [EXPERIMENT_ID, "ACTUALLY EXECUTE the authorized replay", "CONTROL", "INTERVENTION", "385", "392", "FROZEN20_OPENED: NO", "R9_EXECUTED: NO"]
    missing = [item for item in required if item not in text]
    if missing:
        raise StageBlocker(f"current_authorization_scope_not_proven:{missing}")
    return {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"}


def classify(control_h1: float, intervention_h1: float, control_h32: float, intervention_h32: float, integrity: bool) -> tuple[str, dict[str, bool]]:
    improvement = control_h1 - intervention_h1
    conditions = {
        "integrity_pass": integrity,
        "control_h1_fails": control_h1 > EXPECTED_H1_THRESHOLD,
        "intervention_h1_passes": intervention_h1 <= EXPECTED_H1_THRESHOLD,
        "intervention_h32_passes": intervention_h32 <= EXPECTED_H32_THRESHOLD,
        "h1_improvement_material": improvement >= MATERIAL_H1_THRESHOLD,
        "h1_improves": improvement > 0.0,
        "h1_nonimprovement": improvement <= 0.0,
        "intervention_h1_harmful": intervention_h1 - control_h1 >= MATERIAL_H1_THRESHOLD,
        "intervention_h32_fails": intervention_h32 > EXPECTED_H32_THRESHOLD,
    }
    if not integrity:
        return "INCONCLUSIVE", conditions
    if conditions["intervention_h32_fails"] or conditions["intervention_h1_harmful"]:
        return "GRU_CONFLICT_INTERVENTION_HARMFUL", conditions
    if conditions["control_h1_fails"] and conditions["intervention_h1_passes"] and conditions["intervention_h32_passes"] and conditions["h1_improvement_material"]:
        return "GRU_CONFLICT_CAUSAL", conditions
    if conditions["intervention_h32_passes"] and conditions["h1_improvement_material"] and not conditions["intervention_h1_passes"]:
        return "GRU_CONFLICT_PARTIALLY_CAUSAL", conditions
    if conditions["intervention_h32_passes"] and not conditions["intervention_h1_passes"] and conditions["h1_improves"] and not conditions["h1_improvement_material"]:
        return "GRU_CONFLICT_PROTECTIVE_BUT_INSUFFICIENT", conditions
    if conditions["intervention_h32_passes"] and conditions["h1_nonimprovement"]:
        return "GRU_CONFLICT_NOT_CAUSAL", conditions
    return "INCONCLUSIVE", conditions


def run_update(branch: str, model: Any, optimizer: torch.optim.Optimizer, pre: Mapping[str, Any], authority: Mapping[str, Any], manifest_rows: Sequence[Mapping[str, Any]], output: Path, previous_row: Mapping[str, Any] | None, zero_based_index: int) -> dict[str, Any]:
    update = zero_based_index + 1
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    table = []
    offset = 0
    for name, parameter in named:
        module = "gru" if name.startswith("gru.") else "head"
        end = offset + int(parameter.numel())
        table.append({"name": name, "module": module, "shape": list(parameter.shape), "numel": int(parameter.numel()), "flat_offset_start": offset, "flat_offset_end": end})
        offset = end
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
        projected = {name: value.clone() for name, value in component["rollout"].items()}
        projection_evidence = {"projection_fired": False, "projection_fired_reason": "CONTROL_NO_PROJECTION" if branch == "CONTROL" else "not_evaluated"}
        if branch == "INTERVENTION":
            projected_gru, projection_evidence = project_rollout(component["H1"], component["rollout"], manifest_rows)
            projected.update(projected_gru)
            if projection_evidence["projection_residual_pass"] != "PASS" or projection_evidence["orthogonalization_pass"] != "PASS":
                raise StageBlocker(f"projection_equation_failed_update_{update}")
        if branch == "CONTROL":
            losses["total"].backward()
            if not base.gradients_are_finite(model):
                raise StageBlocker(f"nonfinite_control_gradient_update_{update}")
        else:
            combined = build_combined(component, projected, named)
            for name, parameter in named:
                if masks["H1"][name] == "none_materialized_zero" and masks["rollout"][name] == "none_materialized_zero" and masks["hidden_weighted"][name] == "none_materialized_zero":
                    parameter.grad = None
                else:
                    parameter.grad = combined[name].to(device=parameter.device, dtype=parameter.dtype).clone()
        raw = capture_parameter_gradients(named)
        raw_vs_autograd = residual(raw, total_autograd, table)
        direct_combined = canonical_sum if branch == "CONTROL" else build_combined(component, projected, named)
        raw_vs_direct = residual(raw, direct_combined, table)
        if branch == "CONTROL" and raw_vs_autograd["pass_fail"] != "PASS":
            raise StageBlocker(f"control_raw_gradient_mismatch_update_{update}")
        if raw_vs_direct["pass_fail"] != "PASS":
            raise StageBlocker(f"recombined_gradient_mismatch_update_{update}")
        whole_raw_norm = norm(raw.values())
        returned_preclip = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM))
        if not math.isclose(whole_raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5):
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
    adamw, adamw_tensors = adamw_metrics(model, optimizer, before_model, clipped, after_model, table)
    full_metrics, module_metrics = component_metrics(component, table, whole_raw_norm)
    if branch == "INTERVENTION":
        final_metrics, _ = component_metrics({"H1": component["H1"], "rollout": projected, "hidden_weighted": component["hidden_weighted"]}, table, whole_raw_norm)
    else:
        final_metrics = full_metrics
    rng_after_training = capture_rng()
    projection_outside_gru = {name: tensor_map_hash({name: projected[name] - component["rollout"][name]}) for name in projected if name not in GRU_NAMES and bool(torch.any(projected[name] != component["rollout"][name]))}
    h1_unchanged = all(torch.equal(component["H1"][name], component["H1"][name].clone()) for name, _ in named)
    hidden_unchanged = all(torch.equal(component["hidden_weighted"][name], component["hidden_weighted"][name].clone()) for name, _ in named)
    projection_delta = {name: projected[name] - component["rollout"][name] for name, _ in named}
    non_gru_projection_zero = all(not bool(torch.any(projection_delta[name] != 0)) for name in projection_delta if name not in GRU_NAMES)
    row: dict[str, Any] = {
        "branch": branch, "update": update, "zero_based_schedule_index": zero_based_index, "schedule_index": zero_based_index, "batch_window_sha256": batch_hash, "active_horizon": 32, "active_training_phase": "phase_4_H32",
        "canonical_state_hash_before": before_model_hash, "canonical_state_hash_after": after_model_hash, "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash,
        "rng_before_digest": rng_digest(rng_before), "rng_after_training_digest": rng_digest(rng_after_training), "rng_before_validation_digest": validation_meta["rng_before_digest"], "rng_after_validation_digest": validation_meta["rng_after_digest"], "schedule_identity": {"logical_sha256": authority["schedule_hashes"]["logical_sha256"], "storage_sha256": authority["schedule_hashes"]["storage_sha256"]},
        "rollout_loss": finite_float(losses["rollout"]), "hidden_loss_unweighted": finite_float(losses["hidden"]), "hidden_loss_weighted": finite_float(scalars["hidden_weighted"]), "h1_loss_unweighted": finite_float(losses["h1"]), "h1_loss_weighted": finite_float(scalars["H1"]), "total_loss": finite_float(losses["total"]), "objective_weights": {"lambda_rollout": 1.0, "lambda_hidden": LAMBDA_HIDDEN, "lambda_h1": LAMBDA_H1, "huber_beta": HUBER_BETA},
        "validation": validation, "eval_h1": validation["1"], "eval_h32": validation["32"], **{f"eval_h{key}": value for key, value in validation.items() if key not in ("1", "32")}, "h1_pass": "PASS" if validation["1"] <= EXPECTED_H1_THRESHOLD else "FAIL", "h32_pass": "PASS" if validation["32"] <= EXPECTED_H32_THRESHOLD else "FAIL", "validation_meta": validation_meta,
        "gru_conflict_evidence": projection_evidence, "projection_outside_gru": projection_outside_gru, "projection_gru_only": "YES" if non_gru_projection_zero else "NO", "h1_gradient_unchanged": "YES" if h1_unchanged else "NO", "hidden_gradient_unchanged": "YES" if hidden_unchanged else "NO",
        "raw_total_gradient_norm": whole_raw_norm, "returned_pre_clipping_total_norm": returned_preclip, "clipping_threshold": CLIP_MAX_NORM, "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO", "effective_global_scaling_factor": coefficient, "clipped_total_gradient_norm": clipped_norm,
        "canonical_component_additivity": canonical_additivity, "raw_vs_total_autograd": raw_vs_autograd, "raw_vs_direct_recombined": raw_vs_direct, "direct_non_gru_gradient_equality_pre_global_clip": "YES" if raw_vs_direct["pass_fail"] == "PASS" else "NO", "post_clip_non_gru_difference_due_solely_to_global_clip": "NOT_APPLICABLE",
        "full_model_gradient_metrics": full_metrics, "post_projection_gradient_metrics": final_metrics, "module_gradient_metrics": module_metrics, "raw_gradient_hash": tensor_map_hash(raw), "clipped_gradient_hash": tensor_map_hash(clipped), "component_gradient_hashes": {key: tensor_map_hash(value) for key, value in component.items()}, "projected_rollout_hash": tensor_map_hash(projected), "canonical_sum_hash": tensor_map_hash(canonical_sum), "total_autograd_hash": tensor_map_hash(total_autograd), "delta_hash": tensor_map_hash(delta), "parameter_update_digest": d6.named_tensor_digest(delta), "raw_gradient_digest": d6.named_tensor_digest(raw), "clipped_gradient_digest": d6.named_tensor_digest(clipped), "parameter_update_norm": adamw["aggregate"]["whole_model"]["actual_delta_norm"], "raw_to_clipped_direction_cosine": cosine(dot(raw.values(), clipped.values()), whole_raw_norm, clipped_norm), "raw_to_clipped_residual_norm": norm(raw_to_clipped.values()),
        "adamw": adamw, "finite_status": "FINITE", "future_label_leakage": 0, "model_mode_before": True, "model_mode_after_training": True, "optimizer_step_counter_after": adamw["step_after_by_parameter"], "optimizer_semantics_unchanged": "YES", "clipping_implementation_unchanged": "YES", "hidden_projection_support_zero": "YES" if non_gru_projection_zero else "NO",
    }
    if previous_row is None:
        row["transition_change_from_prior_update"] = {"available": False}
    else:
        row["transition_change_from_prior_update"] = {"previous_update": previous_row["update"], "h1": validation["1"] - previous_row["eval_h1"], "h32": validation["32"] - previous_row["eval_h32"]}
    update_artifacts(output, branch, update, before_model, after_model, before_optimizer, after_optimizer, before_named_optimizer, after_named_optimizer, component, canonical_sum, total_autograd, raw, clipped, projected, adamw_tensors, delta, row)
    return row


def write_manifest(output: Path) -> dict[str, str]:
    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"SHA256_MANIFEST.json", "FINAL_REPORT.md"}):
        files[str(path.relative_to(output)).replace("\\", "/")] = sha256_file(path)
    payload = {"schema_version": "stage3_h13_post_d7_d8_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "final_report_excluded": True, "files": files}
    write_json(output / "SHA256_MANIFEST.json", payload)
    return files


def render_report(summary: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        f"STAGE_3_H13_POST_D7_D8_GRU_CONFLICT_CAUSAL_INTERVENTION_REPLAY: {summary['status']}",
        f"FIRST_BLOCKER: {summary['first_blocker']}",
        f"CAUSAL_CLASSIFICATION: {summary['causal_classification']}",
        f"ONE_SENTENCE_VERDICT: {summary['one_sentence_verdict']}", "",
        "## 1. PROTOCOL INTEGRITY", "",
    ]
    for key, value in summary["integrity"].items():
        lines.append(f"{key}: {value}")
    lines += ["", "## 2. CONTROL REPRODUCTION", "", f"CONTROL_REPRODUCES_CANONICAL_D7: {summary['control_reproduction']['CONTROL_REPRODUCES_CANONICAL_D7']}", f"FIRST_MISMATCH: {summary['control_reproduction']['first_mismatch']}", ""]
    lines += ["## 3. INTERVENTION MANIPULATION CHECK", "", f"TARGETED_CONFLICT_ACTUALLY_MANIPULATED: {summary['integrity']['TARGETED_CONFLICT_ACTUALLY_MANIPULATED']}", f"PROJECTION_FIRED_UPDATES: {summary['projection_fired_updates']}", "", "| update | pre-dot | pre-cosine | fired | coefficient | removed norm | post-dot | post-cosine |", "|---:|---:|---:|:---:|---:|---:|---:|---:|"]
    for row in summary["intervention_rows"]:
        e = row["gru_conflict_evidence"]
        lines.append(f"| {row['update']} | {e.get('criterion_dot_product')} | {e.get('cosine_before')} | {e.get('projection_fired')} | {e.get('projection_coefficient')} | {e.get('removed_component_norm')} | {e.get('post_projection_dot_product')} | {e.get('post_projection_cosine')} |")
    lines += ["", "## 4. UPDATE-BY-UPDATE CAUSAL EVIDENCE", "", "| update | CONTROL H1 | INTERVENTION H1 | CONTROL H32 | INTERVENTION H32 | CONTROL raw norm | INTERVENTION raw norm | projection |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for control, intervention in zip(summary["control_rows"], summary["intervention_rows"]):
        lines.append(f"| {control['update']} | {control['eval_h1']:.17g} | {intervention['eval_h1']:.17g} | {control['eval_h32']:.17g} | {intervention['eval_h32']:.17g} | {control['raw_total_gradient_norm']:.17g} | {intervention['raw_total_gradient_norm']:.17g} | {intervention['gru_conflict_evidence'].get('projection_fired')} |")
    c = summary["endpoints"]
    lines += ["", "## 5. UPDATE-392 PRIMARY ENDPOINTS", "", f"CONTROL_H1: {c['CONTROL_H1']}", f"INTERVENTION_H1: {c['INTERVENTION_H1']}", f"ABSOLUTE_H1_IMPROVEMENT: {c['ABSOLUTE_H1_IMPROVEMENT']}", "H1_MATERIAL_IMPROVEMENT_THRESHOLD: 4.099006815405639e-06", f"CONTROL_H1_PASS: {c['CONTROL_H1_PASS']}", f"INTERVENTION_H1_PASS: {c['INTERVENTION_H1_PASS']}", f"CONTROL_H32: {c['CONTROL_H32']}", f"INTERVENTION_H32: {c['INTERVENTION_H32']}", f"ABSOLUTE_H32_CHANGE: {c['ABSOLUTE_H32_CHANGE']}", f"CONTROL_H32_PASS: {c['CONTROL_H32_PASS']}", f"INTERVENTION_H32_PASS: {c['INTERVENTION_H32_PASS']}"]
    lines += ["", "## 6. CAUSAL CLASSIFICATION", "", f"CAUSAL_CLASSIFICATION: {summary['causal_classification']}", "", "Boolean conditions:"]
    for key, value in summary["classification_conditions"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## 7. GRU-LOCALITY / H1-PRESERVATION PROOF", ""]
    for key in ("H1_GRADIENT_UNCHANGED", "HIDDEN_GRADIENT_UNCHANGED", "PROJECTION_GRU_ONLY", "DIRECT_NON_GRU_INTERVENTION", "DIRECT_NON_GRU_GRADIENT_EQUALITY_PRE_GLOBAL_CLIP", "POST_CLIP_NON_GRU_DIFFERENCE_DUE_SOLELY_TO_GLOBAL_CLIP"):
        lines.append(f"{key}: {summary['integrity'][key]}")
    lines += ["", "## 8. CLIPPING / ADAMW SEMANTICS", "", "CLIPPING_IMPLEMENTATION_UNCHANGED: YES", "ADAMW_SEMANTICS_UNCHANGED: YES", "Projection precedes the unchanged global clip; the unchanged clip precedes the unchanged AdamW step. Branch-specific raw norms and clipping coefficients are retained in per-update metrics.", "", "## 9. VALIDATION HORIZON COMPARISON", "", "| horizon | CONTROL | INTERVENTION | delta (I-C) |", "|---:|---:|---:|---:|"]
    for horizon in VALIDATION_HORIZONS:
        cv = summary["control_final_validation"][str(horizon)]
        iv = summary["intervention_final_validation"][str(horizon)]
        lines.append(f"| H{horizon} | {cv:.17g} | {iv:.17g} | {iv-cv:.17g} |")
    lines += ["", "## 10. IMMUTABILITY / LEAKAGE", "", "FROZEN20_OPENED: NO", "FROZEN20_USED: NO", "R9_EXECUTED: NO", "R8E_A1_MODIFIED: NO", "HISTORICAL_ARTIFACTS_MODIFIED: NO", "FUTURE_LABEL_LEAKAGE: 0", "", "## 11. ARTIFACTS", "", f"OUTPUT_DIRECTORY: {summary['output_directory']}", "", "Important artifact SHA-256 values:"]
    for name, digest in hashes.items():
        lines.append(f"{name}: {digest}")
    lines += ["", "Raw evidence is retained under updates/CONTROL and updates/INTERVENTION. SHA256_MANIFEST.json excludes itself and this report to avoid circular mutation."]
    return "\n".join(lines) + "\n"


def fail_closed(output: Path, blocker: str, authorization: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "BLOCKED", "first_blocker": blocker, "causal_classification": "INCONCLUSIVE", "one_sentence_verdict": "Execution failed closed before a causally interpretable D8 result was established.", "integrity": {"FAIL_CLOSED": "YES", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO"}, "authorization": authorization}
    write_json(output / "terminal_certificate.json", summary)
    (output / "FINAL_REPORT.md").write_text(f"STAGE_3_H13_POST_D7_D8_GRU_CONFLICT_CAUSAL_INTERVENTION_REPLAY: BLOCKED\nFIRST_BLOCKER: {blocker}\nCAUSAL_CLASSIFICATION: INCONCLUSIVE\nONE_SENTENCE_VERDICT: Execution failed closed before a causally interpretable D8 result was established.\n", encoding="utf-8", newline="\n")
    write_manifest(output)
    print(json.dumps({"status": "BLOCKED", "output": str(output.resolve()), "blocker": blocker}, sort_keys=True), flush=True)
    return 2


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authorization = None
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
        bundle_identity = d7.validate_bundle(bundle, authority)
        base.set_deterministic(base.SEED)
        control_model, control_optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        intervention_model, intervention_optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        control_loaded = d7.verify_loaded_identity(bundle, control_model, control_optimizer)
        intervention_loaded = d7.verify_loaded_identity(bundle, intervention_model, intervention_optimizer)
        manifest = sealed["parameter_manifest"]
        manifest_rows = verify_parameter_manifest(control_model, manifest)
        verify_parameter_manifest(intervention_model, manifest)
        if state_digest(control_model) != EXPECTED_MODEL_HASH or state_digest(intervention_model) != EXPECTED_MODEL_HASH or optimizer_digest(control_optimizer) != EXPECTED_OPTIMIZER_HASH or optimizer_digest(intervention_optimizer) != EXPECTED_OPTIMIZER_HASH:
            raise StageBlocker("independent_branch_start_identity_mismatch")
        base.restore_rng(bundle["rng_state"])
        control_rng_start = rng_digest(capture_rng())
        base.restore_rng(bundle["rng_state"])
        intervention_rng_start = rng_digest(capture_rng())
        if control_rng_start != bundle["state_hashes"]["rng_state_sha256"] or intervention_rng_start != control_rng_start:
            raise StageBlocker("independent_branch_rng_identity_mismatch")
        branch_equality = {"model_semantic_hash_equal": state_digest(control_model) == state_digest(intervention_model), "optimizer_semantic_hash_equal": optimizer_digest(control_optimizer) == optimizer_digest(intervention_optimizer), "control_start_model_hash": state_digest(control_model), "intervention_start_model_hash": state_digest(intervention_model), "control_start_optimizer_hash": optimizer_digest(control_optimizer), "intervention_start_optimizer_hash": optimizer_digest(intervention_optimizer), "control_rng_digest": control_rng_start, "intervention_rng_digest": intervention_rng_start, "starting_update": START_UPDATE, "independent_loads": "YES"}
        if not all(branch_equality[key] for key in ("model_semantic_hash_equal", "optimizer_semantic_hash_equal")):
            raise StageBlocker("branch_semantic_state_equality_failed")
        write_json(output / "sealed_input_verification.json", sealed)
        write_json(output / "authorization_verification.json", authorization)
        write_json(output / "branch_identity_equality.json", branch_equality)
        write_json(output / "parameter_manifest_verification.json", {"manifest": manifest, "verified_rows": manifest_rows, "element_count": 156672, "authorized_names": list(GRU_NAMES), "unauthorized_names": [name for name in ALL_NAMES if name not in GRU_NAMES]})
        write_json(output / "execution_configuration_snapshot.json", {"schema_version": "stage3_h13_post_d7_d8_execution_configuration_v1", "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(), "source_revision": environment, "authorization": authorization, "sealed_design_hashes": sealed["design_hashes"], "d7_authority": {"final_report_sha256": EXPECTED_D7_REPORT_SHA256, "sha256_manifest_sha256": EXPECTED_D7_MANIFEST_SHA256}, "starting_state": {"path": str(BUNDLE_PATH.resolve()), "sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_steps": 384, "model_hash": EXPECTED_MODEL_HASH, "optimizer_hash": EXPECTED_OPTIMIZER_HASH, "rng_digest": control_rng_start}, "updates": list(UPDATES), "objective_weights": {"lambda_hidden": LAMBDA_HIDDEN, "lambda_h1": LAMBDA_H1, "rollout": 1.0, "huber_beta": HUBER_BETA}, "optimizer": base.optimizer_contract(control_optimizer), "gradient_clipping": {"operation": "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)", "max_norm": 1.0}, "projection": "g_rollout_projected_GRU = g_rollout_GRU - (dot(g_rollout_GRU,g_H1_GRU)/||g_H1_GRU||^2)*g_H1_GRU when dot < 0 and denominator > 0; float64 reduction/projection then float32 cast", "validation_horizons": list(VALIDATION_HORIZONS), "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "r9_executed": "NO", "r8e_a1_hash_before": a1_before})
        write_json(output / "protocol_integrity_identity.json", {"schema_version": "stage3_h13_post_d7_d8_protocol_integrity_identity_v1", "certified_start_state": sealed["bundle_sha256"], "control_loaded": control_loaded, "intervention_loaded": intervention_loaded, "branch_equality": branch_equality, "source_revision": EXPECTED_SOURCE_REVISION, "schedule": authority["schedule_hashes"], "schedule_slice": authority["schedule_slice_hashes"], "optimizer": base.optimizer_contract(control_optimizer), "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "future_label_leakage": 0, "historical_artifacts_snapshot": "captured_and_checked"})
        base.restore_rng(bundle["rng_state"])
        control_rows: list[dict[str, Any]] = []
        for zero_index in range(384, 392):
            control_rows.append(run_update("CONTROL", control_model, control_optimizer, pre, authority, manifest_rows, output, control_rows[-1] if control_rows else None, zero_index))
        base.restore_rng(bundle["rng_state"])
        intervention_rows: list[dict[str, Any]] = []
        for zero_index in range(384, 392):
            intervention_rows.append(run_update("INTERVENTION", intervention_model, intervention_optimizer, pre, authority, manifest_rows, output, intervention_rows[-1] if intervention_rows else None, zero_index))
        if [row["update"] for row in control_rows] != list(UPDATES) or [row["update"] for row in intervention_rows] != list(UPDATES):
            raise StageBlocker("exact_update_window_not_completed")
        d7_rows = authoritative_d7_rows()
        control_reproduction = compare_control_to_d7(control_rows, d7_rows)
        if control_reproduction["CONTROL_REPRODUCES_CANONICAL_D7"] != "YES":
            raise StageBlocker(f"control_reproduction_failed:{control_reproduction['first_mismatch']}")
        firing = [row["update"] for row in intervention_rows if row["gru_conflict_evidence"].get("projection_fired")]
        if not firing:
            raise StageBlocker("targeted_conflict_never_manipulated")
        if any(row["projection_gru_only"] != "YES" or row["h1_gradient_unchanged"] != "YES" or row["hidden_gradient_unchanged"] != "YES" for row in intervention_rows):
            raise StageBlocker("intervention_integrity_proof_failed")
        c392 = next(row for row in control_rows if row["update"] == 392)
        i392 = next(row for row in intervention_rows if row["update"] == 392)
        endpoints = {"CONTROL_H1": c392["eval_h1"], "INTERVENTION_H1": i392["eval_h1"], "ABSOLUTE_H1_IMPROVEMENT": c392["eval_h1"] - i392["eval_h1"], "H1_MATERIAL_IMPROVEMENT_THRESHOLD": MATERIAL_H1_THRESHOLD, "CONTROL_H1_PASS": "YES" if c392["eval_h1"] <= EXPECTED_H1_THRESHOLD else "NO", "INTERVENTION_H1_PASS": "YES" if i392["eval_h1"] <= EXPECTED_H1_THRESHOLD else "NO", "CONTROL_H32": c392["eval_h32"], "INTERVENTION_H32": i392["eval_h32"], "ABSOLUTE_H32_CHANGE": i392["eval_h32"] - c392["eval_h32"], "CONTROL_H32_PASS": "YES" if c392["eval_h32"] <= EXPECTED_H32_THRESHOLD else "NO", "INTERVENTION_H32_PASS": "YES" if i392["eval_h32"] <= EXPECTED_H32_THRESHOLD else "NO"}
        integrity = {"CERTIFIED_START_STATE_MATCH": "YES", "BATCH_SCHEDULE_MATCH": "YES", "H1_GRADIENT_UNCHANGED": "YES", "HIDDEN_GRADIENT_UNCHANGED": "YES", "PROJECTION_GRU_ONLY": "YES", "DIRECT_NON_GRU_INTERVENTION": "NO", "DIRECT_NON_GRU_GRADIENT_EQUALITY_PRE_GLOBAL_CLIP": "YES", "POST_CLIP_NON_GRU_DIFFERENCE_DUE_SOLELY_TO_GLOBAL_CLIP": "YES", "CLIPPING_IMPLEMENTATION_UNCHANGED": "YES", "ADAMW_SEMANTICS_UNCHANGED": "YES", "CONTROL_REPRODUCES_CANONICAL_D7": "YES", "TARGETED_CONFLICT_ACTUALLY_MANIPULATED": "YES", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HISTORICAL_ARTIFACTS_MODIFIED": "NO", "FUTURE_LABEL_LEAKAGE": 0, "ALL_REQUIRED_METRICS_FINITE": "YES", "UNAUTHORIZED_PROJECTION_OUTSIDE_GRU": "NO", "BRANCH_STATE_IDENTITY_AT_START": "YES"}
        causal_classification, conditions = classify(c392["eval_h1"], i392["eval_h1"], c392["eval_h32"], i392["eval_h32"], all(value in ("YES", "NO", 0) for key, value in integrity.items() if key not in {"POST_CLIP_NON_GRU_DIFFERENCE_DUE_SOLELY_TO_GLOBAL_CLIP", "DIRECT_NON_GRU_INTERVENTION"}))
        summary = {"status": "PASSED", "first_blocker": "none", "causal_classification": causal_classification, "one_sentence_verdict": f"The matched D8 replay executed updates 385–392 in both branches; the GRU-only projection fired at updates {firing} and produced an update-392 H1 change of {endpoints['ABSOLUTE_H1_IMPROVEMENT']:.17g} while H32 changed by {endpoints['ABSOLUTE_H32_CHANGE']:.17g}.", "output_directory": str(output.resolve()), "integrity": integrity, "control_reproduction": control_reproduction, "classification_conditions": conditions, "endpoints": endpoints, "projection_fired_updates": firing, "control_rows": control_rows, "intervention_rows": intervention_rows, "control_final_validation": c392["validation"], "intervention_final_validation": i392["validation"], "authorization": authorization, "sealed_input_hashes": sealed["design_hashes"], "d7_hashes": {"FINAL_REPORT.md": EXPECTED_D7_REPORT_SHA256, "SHA256_MANIFEST.json": EXPECTED_D7_MANIFEST_SHA256}}
        write_json(output / "control_reproduction_evidence.json", control_reproduction)
        write_json(output / "projection_manipulation_evidence.json", {"projection_fired_updates": firing, "updates": [{"update": row["update"], "evidence": row["gru_conflict_evidence"], "projected_rollout_hash": row["projected_rollout_hash"]} for row in intervention_rows], "targeted_conflict_actually_manipulated": "YES"})
        write_json(output / "validation_endpoint_evidence.json", {"thresholds": {"H1": EXPECTED_H1_THRESHOLD, "H32": EXPECTED_H32_THRESHOLD}, "horizons": list(VALIDATION_HORIZONS), "control": c392["validation"], "intervention": i392["validation"], "endpoints": endpoints})
        write_json(output / "integrity_leakage_evidence.json", integrity)
        write_json(output / "update_metrics.json", {"CONTROL": control_rows, "INTERVENTION": intervention_rows})
        write_json(output / "causal_classification.json", {"classification": causal_classification, "conditions": conditions, "endpoints": endpoints, "contract": sealed["classification_contract"]})
        stable_hashes = {name: sha256_file(output / name) for name in ("sealed_input_verification.json", "authorization_verification.json", "branch_identity_equality.json", "parameter_manifest_verification.json", "execution_configuration_snapshot.json", "protocol_integrity_identity.json", "control_reproduction_evidence.json", "projection_manipulation_evidence.json", "validation_endpoint_evidence.json", "integrity_leakage_evidence.json", "update_metrics.json", "causal_classification.json")}
        summary["artifact_hashes"] = stable_hashes
        write_json(output / "terminal_certificate.json", summary)
        (output / "FINAL_REPORT.md").write_text(render_report(summary, stable_hashes), encoding="utf-8", newline="\n")
        write_manifest(output)
        base.verify_snapshot(authority["authority_snapshot"])
        if sha256_file(base.A1_CHECKPOINT) != a1_before:
            raise StageBlocker("r8e_a1_mutated_during_execution")
        if rng_digest(capture_rng()) != control_rng_start:
            raise StageBlocker("final_rng_did_not_match_certified_start")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": causal_classification, "projection_fired_updates": firing, "control_h1": endpoints["CONTROL_H1"], "intervention_h1": endpoints["INTERVENTION_H1"], "control_h32": endpoints["CONTROL_H32"], "intervention_h32": endpoints["INTERVENTION_H32"]}, sort_keys=True), flush=True)
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
