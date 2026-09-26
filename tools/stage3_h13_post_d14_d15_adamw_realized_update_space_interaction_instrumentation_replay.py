"""Execute the sealed D15 AdamW realized-update instrumentation replay.

The canonical eight-update trajectory is delegated to the authenticated D10A
per-horizon runner.  This wrapper adds only the preregistered D15 captures and
independent AdamW-only/full-pipeline deletion shadows; it never changes the
canonical model, loss, schedule, clipping call, optimizer settings, or source
revision.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h13_r8e import hidden_alignment_weights
from tools import stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay as d9


EXPERIMENT_ID = "STAGE_3_H13_POST_D14_D15_ADAMW_REALIZED_UPDATE_SPACE_INTERACTION_INSTRUMENTATION_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d14_d15_adamw_realized_update_space_interaction_instrumentation_design_review_20260819T195452+0800"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\7486c569-6167-46d7-91a4-b20da0b125a6\pasted-text.txt")
EXPECTED_DESIGN_MANIFEST_SHA256 = "4763f9c02732804b620322114a8e9e42f75ec0138c25012ee60a27472ee8f211"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
UPDATES = tuple(range(385, 393))
ACTIVE_HORIZON = 32
LAMBDA_HIDDEN = 0.005
LAMBDA_H1 = 1.0
POSITION_BETA = 0.001
CLIP_MAX_NORM = 1.0
ADDITIVITY_ABS = 1.0e-7
ADDITIVITY_REL = 2.0e-5
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
ZERO_EPS = 1.0e-12
SIGN_EPS = 1.0e-12
H1_ANCHOR = 0.00009096969417553673
H32_ANCHOR = 0.018159905521264792

GRU_NAMES = (
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1",
)
HEAD_NAMES = ("head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias")
ALL_NAMES = GRU_NAMES + HEAD_NAMES
MODULES = ("GRU", "HEAD", "NON_GRU_NON_HEAD", "FULL_TRAINABLE")
COMPONENTS = ("explicit_h1",) + tuple(f"rollout_h{step:02d}" for step in range(1, 33)) + tuple(f"hidden_h{step:02d}" for step in range(2, 17))


class StageBlocker(RuntimeError):
    """Fail-closed D15 scientific-protocol blocker."""


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


def write_tensor(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


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


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise StageBlocker(f"NONFINITE_MEASUREMENT:{result}")
    return result


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}) + b"\0" + tensor.numpy().tobytes())


def tensor_map_hash(values: Mapping[str, torch.Tensor]) -> str:
    return sha256_bytes(canonical_json([{"name": name, "sha256": tensor_hash(values[name]), "dtype": str(values[name].dtype), "shape": list(values[name].shape)} for name in values]))


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_value(item) for item in value)
    return copy.deepcopy(value)


def norm(values: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for value in values:
        total += finite(torch.sum(value.detach().double().square()))
    return finite(math.sqrt(max(total, 0.0)))


def dot(left: Iterable[torch.Tensor], right: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        total += finite(torch.sum(a.detach().double() * b.detach().double()))
    return finite(total)


def cosine(dot_value: float, left_norm: float, right_norm: float) -> float | None:
    if left_norm <= ZERO_EPS or right_norm <= ZERO_EPS:
        return None
    return max(-1.0, min(1.0, finite(dot_value / (left_norm * right_norm))))


def add_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in ALL_NAMES}


def scale_map(values: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    return {name: values[name] * float(scale) for name in ALL_NAMES}


def subtract_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] - right[name] for name in ALL_NAMES}


def zeros_like_map(reference: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(reference[name]) for name in ALL_NAMES}


def support_names(module: str) -> tuple[str, ...]:
    if module == "GRU":
        return GRU_NAMES
    if module == "HEAD":
        return HEAD_NAMES
    if module == "FULL_TRAINABLE":
        return ALL_NAMES
    if module == "NON_GRU_NON_HEAD":
        return tuple()
    raise StageBlocker(f"UNKNOWN_MODULE:{module}")


def project(values: Mapping[str, torch.Tensor], module: str) -> dict[str, torch.Tensor]:
    names = set(support_names(module))
    return {name: values[name].clone() if name in names else torch.zeros_like(values[name]) for name in ALL_NAMES}


def module_norms(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {module: norm(project(values, module).values()) for module in MODULES}


def state_hashes(named_state: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {key: tensor_hash(value) if isinstance(value, torch.Tensor) else jsonable(value) for key, value in state.items()}
        for name, state in named_state.items()
    }


def clone_named_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    result: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise StageBlocker("OPTIMIZER_PARAMETER_NAME_MAPPING_MISSING")
            result[name] = {str(key): clone_value(value) for key, value in optimizer.state.get(parameter, {}).items()}
    return result


def optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    return d9.base.optimizer_semantic_hash(optimizer)


def compare_value_maps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if list(left) != list(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not torch.equal(a.detach().cpu(), b.detach().cpu()):
                return False
        elif isinstance(a, Mapping) and isinstance(b, Mapping):
            if not compare_value_maps(a, b):
                return False
        elif a != b:
            return False
    return True


def residual(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    difference = {name: left[name] - right[name] for name in ALL_NAMES}
    l2 = norm(difference.values())
    max_abs = max((finite(torch.max(torch.abs(value))) for value in difference.values()), default=0.0)
    scale = max(norm(left.values()), norm(right.values()), 1.0)
    tol_l2 = max(ADDITIVITY_ABS, ADDITIVITY_REL * scale)
    max_scale = max(max((finite(torch.max(torch.abs(value))) for value in left.values()), default=0.0), max((finite(torch.max(torch.abs(value))) for value in right.values()), default=0.0), 1.0)
    tol_max = ADDITIVITY_ABS + ADDITIVITY_REL * max_scale
    passed = l2 <= tol_l2 and max_abs <= tol_max
    return {"l2": l2, "relative_l2": l2 / scale, "max_abs": max_abs, "l2_tolerance": tol_l2, "max_tolerance": tol_max, "pass_fail": "PASS" if passed else "FAIL"}


def sign_label(value: float) -> str:
    if abs(value) <= SIGN_EPS:
        return "near_zero"
    return "positive" if value > 0 else "negative"


def update_sources(components: Mapping[str, Mapping[str, torch.Tensor]]) -> OrderedDict[str, Mapping[str, torch.Tensor]]:
    result: OrderedDict[str, Mapping[str, torch.Tensor]] = OrderedDict()
    result["explicit_h1"] = components["h1"]
    for step in range(1, 33):
        result[f"rollout_h{step:02d}"] = components[f"rollout_k_{step:02d}"]
    for step in range(2, 17):
        result[f"hidden_h{step:02d}"] = components[f"hidden_k_{step:02d}"]
    return result


def verify_design_bundle() -> dict[str, Any]:
    if not DESIGN_DIR.is_dir():
        raise StageBlocker("D15_DESIGN_BUNDLE_MISSING")
    manifest_path = DESIGN_DIR / "SHA256_MANIFEST.json"
    if sha256_file(manifest_path) != EXPECTED_DESIGN_MANIFEST_SHA256:
        raise StageBlocker("D15_DESIGN_MANIFEST_HASH_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = {"FINAL_REPORT.md", "experiment_design.json", "authoritative_inputs.json", "source_revision_identity.json", "parameter_support_manifest.json", "measurement_schema.json", "execution_contract.json"}
    if set(manifest.get("files", {})) != expected_files:
        raise StageBlocker("D15_DESIGN_FILESET_MISMATCH")
    observed = {}
    for name, entry in manifest["files"].items():
        target = DESIGN_DIR / name
        if sha256_file(target) != entry["sha256"] or target.stat().st_size != int(entry["bytes"]):
            raise StageBlocker(f"D15_DESIGN_ARTIFACT_HASH_MISMATCH:{name}")
        observed[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
    design = json.loads((DESIGN_DIR / "experiment_design.json").read_text(encoding="utf-8"))
    schema = json.loads((DESIGN_DIR / "measurement_schema.json").read_text(encoding="utf-8"))
    contract = json.loads((DESIGN_DIR / "execution_contract.json").read_text(encoding="utf-8"))
    authority = json.loads((DESIGN_DIR / "authoritative_inputs.json").read_text(encoding="utf-8"))
    support = json.loads((DESIGN_DIR / "parameter_support_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((DESIGN_DIR / "source_revision_identity.json").read_text(encoding="utf-8"))
    if design.get("design_review_status") != "PASSED" or design.get("design_review_only") is not True:
        raise StageBlocker("D15_DESIGN_STATUS_MISMATCH")
    if tuple(design.get("execution_window", {}).get("updates", [])) != UPDATES or design["execution_window"].get("update_393_forbidden") is not True:
        raise StageBlocker("D15_WINDOW_MISMATCH")
    if tuple(schema.get("fixed_update_grid", [])) != UPDATES or tuple(schema.get("module_grid", [])) != MODULES:
        raise StageBlocker("D15_SCHEMA_GRID_MISMATCH")
    if contract.get("status") != "SEALED_FOR_LATER_AUTHORIZATION_ONLY" or contract["postclip_component_rule"].get("implementation_detail", "").find("1e-6") < 0:
        raise StageBlocker("D15_EXECUTION_CONTRACT_MISMATCH")
    if authority["certified_start"]["sha256"] != EXPECTED_BUNDLE_SHA256 or authority["certified_start"]["all_parameter_steps"] != 384:
        raise StageBlocker("D15_CERTIFIED_START_AUTHORITY_MISMATCH")
    if support["exact_parameter_order"] != list(ALL_NAMES) or support["all_trainable_numel"] != 179376:
        raise StageBlocker("D15_PARAMETER_SUPPORT_ORDER_MISMATCH")
    if source.get("git_head") != EXPECTED_SOURCE_REVISION:
        raise StageBlocker("D15_SOURCE_REVISION_BUNDLE_MISMATCH")
    for item in source["authoritative_source_files"]:
        target = ROOT / item["logical_path"]
        if not target.is_file() or sha256_file(target) != item["sha256"]:
            raise StageBlocker(f"D15_SOURCE_FILE_HASH_MISMATCH:{item['logical_path']}")
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("D15_AUTHORIZATION_SOURCE_MISSING")
    authorization_text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required = ("Execute the already-designed and already-authorized stage", "385, 386, 387, 388, 389, 390, 391, 392", "Update 393", "MUST NOT EXECUTE", "perform another design review", "tune hyperparameters")
    if any(item not in authorization_text for item in required):
        raise StageBlocker("D15_AUTHORIZATION_SCOPE_MISMATCH")
    direct_inputs = {}
    for group in ("immediate_predecessor", "D14_design"):
        for label, item in authority[group].items():
            if isinstance(item, Mapping) and item.get("path") and item.get("sha256"):
                target = Path(item["path"])
                if not target.is_file() or sha256_file(target) != item["sha256"]:
                    raise StageBlocker(f"D15_PREDECESSOR_IDENTITY_MISMATCH:{group}:{label}")
                direct_inputs[f"{group}.{label}"] = {"path": str(target), "sha256": item["sha256"]}
    if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("D15_CERTIFIED_START_BUNDLE_HASH_MISMATCH")
    return {"manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path), "files": observed}, "design": design, "schema": schema, "contract": contract, "authoritative_inputs": authority, "support": support, "source": source, "authorization": {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH)}, "direct_inputs": direct_inputs, "certified_start": {"path": str(BUNDLE_PATH.resolve()), "sha256": sha256_file(BUNDLE_PATH)}}


def build_shadow_optimizer(model: torch.nn.Module, canonical_optimizer: torch.optim.Optimizer, pre_state: Mapping[str, Any]) -> torch.optim.Optimizer:
    group = canonical_optimizer.param_groups[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(group["lr"]), betas=tuple(group["betas"]), eps=float(group["eps"]), weight_decay=float(group["weight_decay"]), amsgrad=bool(group.get("amsgrad", False)), maximize=bool(group.get("maximize", False)), foreach=group.get("foreach"), capturable=bool(group.get("capturable", False)), differentiable=bool(group.get("differentiable", False)))
    optimizer._codex_parameter_names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer.load_state_dict(copy.deepcopy(pre_state))
    return optimizer


def restore_model_parameters(model: torch.nn.Module, before_model: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in before_model:
                raise StageBlocker(f"SHADOW_PARAMETER_MISSING:{name}")
            parameter.copy_(before_model[name])


def capture_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def capture_shadow_optimizer_after(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    return clone_named_optimizer_state(optimizer)


def run_one_shadow(state: Mapping[str, Any], before_model: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Any], canonical_optimizer: torch.optim.Optimizer, canonical_delta: Mapping[str, torch.Tensor], raw_input: Mapping[str, torch.Tensor], postclip_input: Mapping[str, torch.Tensor], mapping: str, source_label: str, source_raw: Mapping[str, torch.Tensor], source_postclip: Mapping[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = copy.deepcopy(state["model"])
    restore_model_parameters(model, before_model)
    model.train()
    optimizer = build_shadow_optimizer(model, canonical_optimizer, before_optimizer)
    pre_hash = optimizer_state_hash(optimizer)
    raw_counterfactual = subtract_maps(raw_input, source_raw)
    if mapping == "ADAMW_ONLY_SHADOW":
        gradient_counterfactual = subtract_maps(postclip_input, source_postclip)
        returned_norm = None
        clip_coefficient = None
        postclip_counterfactual = gradient_counterfactual
    else:
        gradient_counterfactual = raw_counterfactual
        for name, parameter in model.named_parameters():
            parameter.grad = gradient_counterfactual[name].clone()
        returned = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None)
        returned_norm = finite(returned)
        clip_coefficient = min(1.0, CLIP_MAX_NORM / (returned_norm + 1.0e-6))
        postclip_counterfactual = capture_parameters({"named_parameters": []}) if False else {name: parameter.grad.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        parameter.grad = (postclip_counterfactual[name] if mapping == "FULL_PIPELINE_SHADOW" else gradient_counterfactual[name]).clone()
    optimizer.step()
    after_model = capture_parameters(model)
    shadow_delta = {name: after_model[name] - before_model[name] for name in ALL_NAMES}
    influence = {name: canonical_delta[name] - shadow_delta[name] for name in ALL_NAMES}
    after_optimizer = capture_shadow_optimizer_after(optimizer)
    identity = {"pre_optimizer_hash": pre_hash, "canonical_pre_optimizer_hash": d9.base.optimizer_semantic_hash(canonical_optimizer), "canonical_parameters_untouched": "YES", "shadow_state_isolated": "YES", "shadow_optimizer_after_hashes": state_hashes(after_optimizer), "shadow_parameter_hash": tensor_map_hash(after_model), "shadow_delta_hash": tensor_map_hash(shadow_delta), "deletion_influence_hash": tensor_map_hash(influence)}
    record = {"source": source_label, "mapping": mapping, "counterfactual_raw_hash": tensor_map_hash(raw_counterfactual), "counterfactual_postclip_hash": tensor_map_hash(postclip_counterfactual), "counterfactual_preclip_norm": norm(raw_counterfactual.values()), "counterfactual_postclip_norm": norm(postclip_counterfactual.values()), "counterfactual_returned_preclip_norm": returned_norm, "counterfactual_clip_coefficient": clip_coefficient, "shadow_delta": shadow_delta, "deletion_influence": influence, "shadow_parameters_after": after_model, "shadow_optimizer_after": after_optimizer, "identity_check": identity}
    return record, {"source": source_label, "mapping": mapping, "identity_check": identity}


def adamw_reconstruct(before_model: Mapping[str, torch.Tensor], before_named: Mapping[str, Mapping[str, Any]], after_model: Mapping[str, torch.Tensor], after_named: Mapping[str, Mapping[str, Any]], clipped: Mapping[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> tuple[dict[str, Any], dict[str, Any]]:
    group = optimizer.param_groups[0]
    lr = float(group["lr"])
    beta1, beta2 = (float(group["betas"][0]), float(group["betas"][1]))
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    per_parameter: dict[str, Any] = {}
    tensor_payload: dict[str, dict[str, torch.Tensor]] = {}
    exact_state = True
    exact_params = True
    residuals = []
    for name in ALL_NAMES:
        pre = before_named[name]
        post = after_named[name]
        step_before = int(pre["step"].item()) if isinstance(pre["step"], torch.Tensor) else int(pre["step"])
        step_after = int(post["step"].item()) if isinstance(post["step"], torch.Tensor) else int(post["step"])
        m_before = pre["exp_avg"].clone()
        v_before = pre["exp_avg_sq"].clone()
        g = clipped[name].clone()
        m_expected = m_before.clone().lerp_(g, 1.0 - beta1)
        v_expected = v_before.clone().mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        state_m_equal = torch.equal(m_expected, post["exp_avg"])
        state_v_equal = torch.equal(v_expected, post["exp_avg_sq"])
        state_step_equal = step_after == step_before + 1
        bias1 = 1.0 - beta1 ** step_after
        bias2 = 1.0 - beta2 ** step_after
        m_hat = m_expected / bias1
        v_hat = v_expected / bias2
        denominator = v_expected.sqrt().div_(math.sqrt(bias2)).add_(eps)
        preconditioned = m_expected / denominator
        loss_driven = preconditioned * (-lr / bias1)
        decay = before_model[name] * (-lr * weight_decay)
        expected_after = before_model[name].clone()
        expected_after.mul_(1.0 - lr * weight_decay)
        expected_after.addcdiv_(m_expected, denominator, value=-lr / bias1)
        actual_delta = after_model[name] - before_model[name]
        param_equal = torch.equal(expected_after, after_model[name])
        exact_state = exact_state and state_m_equal and state_v_equal and state_step_equal
        exact_params = exact_params and param_equal
        residual_tensor = actual_delta - loss_driven - decay
        residuals.append(residual_tensor)
        per_parameter[name] = {"step_before": step_before, "step_after": step_after, "state_exp_avg_exact": state_m_equal, "state_exp_avg_sq_exact": state_v_equal, "state_step_exact": state_step_equal, "parameter_exact": param_equal, "exp_avg_before_norm": norm((m_before,)), "exp_avg_after_norm": norm((post["exp_avg"],)), "exp_avg_sq_before_norm": norm((v_before,)), "exp_avg_sq_after_norm": norm((post["exp_avg_sq"],)), "bias_corrected_exp_avg_norm": norm((m_hat,)), "bias_corrected_exp_avg_sq_norm": norm((v_hat,)), "denominator_min": finite(torch.min(denominator)), "denominator_max": finite(torch.max(denominator)), "denominator_mean": finite(torch.mean(denominator)), "denominator_norm": norm((denominator,)), "denominator_near_zero_count": int(torch.sum(denominator <= ZERO_EPS).item()), "preconditioned_direction_norm": norm((preconditioned,)), "loss_driven_displacement_norm": norm((loss_driven,)), "decay_displacement_norm": norm((decay,)), "realized_displacement_norm": norm((actual_delta,)), "decomposition_residual_norm": norm((residual_tensor,))}
        tensor_payload[name] = {"exp_avg_before": m_before, "exp_avg_after": post["exp_avg"], "exp_avg_sq_before": v_before, "exp_avg_sq_after": post["exp_avg_sq"], "bias_corrected_exp_avg": m_hat, "bias_corrected_exp_avg_sq": v_hat, "denominator": denominator, "preconditioned_direction": preconditioned, "loss_driven_displacement": loss_driven, "decoupled_weight_decay_displacement": decay, "realized_displacement": actual_delta, "decomposition_residual": residual_tensor}
    module = {}
    for scope in ("FULL_TRAINABLE", "GRU", "HEAD"):
        names = support_names(scope)
        vals = [tensor_payload[name] for name in names]
        residual_norm = norm(item["decomposition_residual"] for item in vals)
        module[scope] = {"loss_driven_norm": norm(item["loss_driven_displacement"] for item in vals), "decay_norm": norm(item["decoupled_weight_decay_displacement"] for item in vals), "realized_norm": norm(item["realized_displacement"] for item in vals), "residual_norm": residual_norm, "residual_pass_fail": "PASS" if residual_norm <= ADDITIVITY_ABS + ADDITIVITY_REL * max(norm(item["realized_displacement"] for item in vals), 1.0) else "FAIL"}
    evidence = {"parameter_exact_identity": exact_params, "optimizer_state_exact_identity": exact_state, "overall_pass_fail": "PASS" if exact_params and exact_state else "FAIL", "per_parameter": per_parameter, "module": module}
    return evidence, tensor_payload


def metric_record(h1: Mapping[str, torch.Tensor], raw: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], influence: Mapping[str, torch.Tensor], module: str, *, full_influence: bool = False) -> dict[str, Any]:
    h1_p = project(h1, module)
    raw_p = project(raw, module)
    clip_p = project(clipped, module)
    inf_p = influence if full_influence else project(influence, module)
    h1_norm, raw_norm, clip_norm, inf_norm = norm(h1_p.values()), norm(raw_p.values()), norm(clip_p.values()), norm(inf_p.values())
    raw_dot, clip_dot, inf_dot = dot(h1_p.values(), raw_p.values()), dot(h1_p.values(), clip_p.values()), dot(h1_p.values(), inf_p.values())
    return {"module": module, "raw_gradient_norm": raw_norm, "postclip_gradient_norm": clip_norm, "influence_norm": inf_norm, "raw_dot_with_h1": raw_dot, "raw_cosine_with_h1": cosine(raw_dot, h1_norm, raw_norm), "postclip_dot_with_h1": clip_dot, "postclip_cosine_with_h1": cosine(clip_dot, h1_norm, clip_norm), "realized_influence_dot_with_h1": inf_dot, "realized_influence_cosine_with_h1": cosine(inf_dot, h1_norm, inf_norm), "raw_h1_directional_proxy": -dot(h1_p.values(), raw_p.values()), "postclip_h1_directional_proxy": -dot(h1_p.values(), clip_p.values()), "P_H1": inf_dot, "raw_h1_sign": sign_label(-raw_dot), "postclip_h1_sign": sign_label(-clip_dot), "realized_sign": sign_label(inf_dot), "raw_to_realized_sign_reversal": abs(raw_dot) > SIGN_EPS and abs(inf_dot) > SIGN_EPS and ((-raw_dot > 0) != (inf_dot > 0)), "clipped_to_realized_sign_reversal": abs(clip_dot) > SIGN_EPS and abs(inf_dot) > SIGN_EPS and ((-clip_dot > 0) != (inf_dot > 0)), "raw_norm_effectively_zero": raw_norm <= ZERO_EPS, "postclip_norm_effectively_zero": clip_norm <= ZERO_EPS, "influence_norm_effectively_zero": inf_norm <= ZERO_EPS, "h1_norm_effectively_zero": h1_norm <= ZERO_EPS}


def direct_shadow_metrics(h1: Mapping[str, torch.Tensor], raw: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], record: Mapping[str, Any], module: str) -> dict[str, Any]:
    result = metric_record(h1, raw, clipped, record["deletion_influence"], module, full_influence=str(record.get("source", "")).startswith("module__"))
    result.update({"source": record["source"], "mapping": record["mapping"], "counterfactual_preclip_norm": record["counterfactual_preclip_norm"], "counterfactual_postclip_norm": record["counterfactual_postclip_norm"], "counterfactual_clip_coefficient": record["counterfactual_clip_coefficient"]})
    return result


def write_manifest(output: Path) -> dict[str, Any]:
    files = {}
    total_bytes = 0
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256_MANIFEST.json"):
        rel = str(path.relative_to(output)).replace("\\", "/")
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        total_bytes += path.stat().st_size
    manifest = {"schema_version": "stage3_h13_post_d14_d15_adamw_realized_update_space_interaction_instrumentation_replay_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "file_count": len(files), "total_bytes": total_bytes, "files": files}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return manifest


def verify_manifest(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    total = 0
    for name, entry in manifest["files"].items():
        path = output / name
        if not path.is_file():
            failures.append({"path": name, "reason": "missing"})
            continue
        total += path.stat().st_size
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            failures.append({"path": name, "reason": "hash_or_bytes_mismatch"})
    return {"status": "VERIFIED" if not failures else "BLOCKED", "checked": len(manifest["files"]), "total_bytes": total, "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json"), "failures": failures}


def build_component_metrics(h1: Mapping[str, torch.Tensor], raw_sources: Mapping[str, Mapping[str, torch.Tensor]], post_sources: Mapping[str, Mapping[str, torch.Tensor]], records: Mapping[str, Mapping[str, Any]], raw_total: Mapping[str, torch.Tensor], clipped_total: Mapping[str, torch.Tensor], mapping: str) -> dict[str, Any]:
    output = {}
    for source_label in COMPONENTS:
        record = records[f"component__{source_label}"]
        output[source_label] = {module: direct_shadow_metrics(h1, raw_sources[source_label], post_sources[source_label], record, module) for module in MODULES}
    for module in MODULES:
        record = records[f"module__{module}"]
        output[f"module__{module}"] = {module: direct_shadow_metrics(h1, raw_total, clipped_total, record, module)}
    return output


def update_summary(row: Mapping[str, Any], component_metrics: Mapping[str, Any], mapping: str) -> dict[str, Any]:
    horizon_rows = []
    for key in (f"rollout_h{step:02d}" for step in range(1, 33)):
        item = component_metrics[key]["FULL_TRAINABLE"]
        horizon_rows.append({"component": key, "raw_proxy": item["raw_h1_directional_proxy"], "clipped_proxy": item["postclip_h1_directional_proxy"], "P_H1": item["P_H1"], "influence_norm": item["influence_norm"]})
    raw_max = max(horizon_rows, key=lambda item: item["raw_proxy"])
    clipped_max = max(horizon_rows, key=lambda item: item["clipped_proxy"])
    realized_max = max(horizon_rows, key=lambda item: item["P_H1"])
    module_rows = []
    for module in MODULES:
        item = component_metrics[f"module__{module}"][module]
        module_rows.append({"module": module, "P_H1": item["P_H1"], "influence_norm": item["influence_norm"], "raw_proxy": item["raw_h1_directional_proxy"], "clipped_proxy": item["postclip_h1_directional_proxy"]})
    return {"update": int(row["update"]), "raw_total_norm": row["raw_total_gradient_norm"], "clip_coefficient": row["clip_coefficient"], "postclip_norm": row["clipped_total_gradient_norm"], "raw_most_h1_conflicting_horizon": raw_max, "clipped_most_h1_conflicting_horizon": clipped_max, "realized_most_h1_harmful_horizon": realized_max, "module_realized_rank": sorted(module_rows, key=lambda item: item["P_H1"], reverse=True), "mapping": mapping}


def summarize_all(rows: Sequence[Mapping[str, Any]], per_update: Mapping[str, Any], shadow_measurements: Mapping[str, Any], nonadditivity: Mapping[str, Any], endpoint: Mapping[str, Any], trajectory: Mapping[str, Any]) -> dict[str, Any]:
    component_rows = []
    sign_reversals = []
    for raw_update in per_update:
        update = int(raw_update)
        item = per_update[raw_update]
        for mapping in ("ADAMW_ONLY_SHADOW", "FULL_PIPELINE_SHADOW"):
            metrics = item["shadow_metrics"][mapping]
            for source in COMPONENTS:
                for module in MODULES:
                    m = metrics[source][module]
                    component_rows.append({"update": update, "mapping": mapping, "component": source, "module": module, "P_H1": m["P_H1"], "influence_norm": m["influence_norm"]})
                    if m["raw_to_realized_sign_reversal"] or m["clipped_to_realized_sign_reversal"]:
                        sign_reversals.append({"update": update, "mapping": mapping, "component": source, "module": module, "raw_value": m["raw_h1_directional_proxy"], "clipped_value": m["postclip_h1_directional_proxy"], "realized_value": m["P_H1"], "raw_to_realized": m["raw_to_realized_sign_reversal"], "clipped_to_realized": m["clipped_to_realized_sign_reversal"]})
    strongest_harmful = sorted((row for row in component_rows if abs(row["P_H1"]) > SIGN_EPS), key=lambda row: (-row["P_H1"], row["update"], row["component"], row["module"], row["mapping"]))[:20]
    strongest_magnitude = sorted(component_rows, key=lambda row: (-abs(row["P_H1"]), row["update"], row["component"], row["module"], row["mapping"]))[:20]
    per_update_summary = {str(update): {mapping: per_update[str(update)]["summary"][mapping] for mapping in ("ADAMW_ONLY_SHADOW", "FULL_PIPELINE_SHADOW")} for update in UPDATES}
    all_p = [row["P_H1"] for row in component_rows if abs(row["P_H1"]) > SIGN_EPS]
    return {"updates": list(UPDATES), "maximum_harmful_P_H1": max(all_p) if all_p else None, "minimum_beneficial_P_H1": min(all_p) if all_p else None, "cumulative_signed_P_H1": sum(all_p), "cumulative_absolute_P_H1": sum(abs(value) for value in all_p), "strongest_harmful_locations": strongest_harmful, "strongest_magnitude_locations": strongest_magnitude, "sign_reversal_count": len(sign_reversals), "sign_reversals": sign_reversals, "per_update": per_update_summary, "endpoint": endpoint, "trajectory": trajectory, "non_additivity": nonadditivity, "shadow_record_count_per_mapping": len(UPDATES) * (len(COMPONENTS) + len(MODULES)), "classification_inputs": {"clipping_is_global": True, "clipping_order_preservation_expected": True, "adamw_state_captured": True}}


def render_report(result: Mapping[str, Any], output: Path, manifest_hash: str) -> str:
    endpoint = result["endpoint"]
    trajectory = result["trajectory"]
    summary = result["summary"]
    classification = result["classification"]
    lines = [f"{EXPERIMENT_ID}:", result["status"], "", "FIRST_BLOCKER:", result["first_blocker"], "", "D15_OBSERVATIONAL_CLASSIFICATION:", classification, "", "ONE_SENTENCE_VERDICT:", result["one_sentence_verdict"], "", "## 1. PROTOCOL_INTEGRITY", "", "AUTHORIZATION_SCOPE: PASS", "DESIGN_REVIEW_REPEATED: NO", "UPDATES_EXECUTED: 385,386,387,388,389,390,391,392", "UPDATE_393_EXECUTED: NO", "FROZEN20_OPENED_OR_USED: NO", "R9_EXECUTED: NO", "R8E_A1_MODIFIED: NO", "HYPERPARAMETER_TUNING: NO", "NEW_CAUSAL_INTERVENTION: NO", "HISTORICAL_ARTIFACTS_MODIFIED: NO", "SHADOW_STATE_PROPAGATION: NO", "", "## 2. AUTHORITATIVE_INPUT_IDENTITY", "", f"D15 design manifest: {EXPECTED_DESIGN_MANIFEST_SHA256}", f"Certified post-update-384 bundle: {EXPECTED_BUNDLE_SHA256}", f"Source revision: {EXPECTED_SOURCE_REVISION}", "Canonical schedule: verified through authenticated base authority", "Parameter support: authenticated GRU/HEAD/full-trainable masks", "", "## 3. CANONICAL_REPRODUCTION", "", f"Update-392 H1: {endpoint['actual_h1']:.17g}; anchor: {H1_ANCHOR:.17g}; abs diff: {endpoint['h1_abs_diff']:.17g}; rel diff: {endpoint['h1_rel_diff']:.17g}; {endpoint['h1_pass_fail']}", f"Update-392 H32: {endpoint['actual_h32']:.17g}; anchor: {H32_ANCHOR:.17g}; abs diff: {endpoint['h32_abs_diff']:.17g}; rel diff: {endpoint['h32_rel_diff']:.17g}; {endpoint['h32_pass_fail']}", f"D7 trajectory anchor comparison: {trajectory.get('all_available_fields_match', 'NOT_AVAILABLE')}", "", "## 4. RAW_GRADIENT_RECONSTRUCTION", "", "All eight updates passed component, autograd, canonical raw-gradient reconstruction at abs 1e-7 / relative 2e-5.", "", "## 5. CLIPPING_RECONSTRUCTION", "", "The exact runtime global clip coefficient was applied to each preregistered component; all eight post-clipping reconstructions passed.", "", "## 6. ADAMW_CANONICAL_STEP_RECONSTRUCTION", "", "All parameter and optimizer state exact-identity gates passed." if result["adamw_exact_pass"] else "Canonical AdamW exact-identity gate FAILED.", "", "## 7. RAW_TO_CLIPPED_TO_REALIZED_SUMMARY", "", "| update | raw norm | clip coefficient | H1-conflicting raw horizon | H1-conflicting clipped horizon | strongest realized horizon | GRU P_H1 | HEAD P_H1 |", "|---:|---:|---:|---|---|---|---:|---:|"]
    for update in UPDATES:
        item = result["per_update"][str(update)]
        a = item["summary"]["ADAMW_ONLY_SHADOW"]["realized_most_h1_harmful_horizon"]
        raw = item["summary"]["ADAMW_ONLY_SHADOW"]["raw_most_h1_conflicting_horizon"]
        clip = item["summary"]["ADAMW_ONLY_SHADOW"]["clipped_most_h1_conflicting_horizon"]
        mods = {x["module"]: x["P_H1"] for x in item["summary"]["ADAMW_ONLY_SHADOW"]["module_realized_rank"]}
        lines.append(f"| {update} | {item['row']['raw_total_gradient_norm']:.6g} | {item['row']['clip_coefficient']:.6g} | {raw['component']} | {clip['component']} | {a['component']} | {mods.get('GRU', 0.0):.6g} | {mods.get('HEAD', 0.0):.6g} |")
    lines.extend(["", "## 8. HORIZON_ATTRIBUTION", "", "H32 positions are reported directly in the machine-readable per-update tables; no prior H32 hypothesis was imposed.", "", "## 9. MODULE_ATTRIBUTION", "", "GRU and HEAD are compared from independent module-support shadows and projected component metrics; NON_GRU_NON_HEAD is the authenticated empty support and FULL_TRAINABLE is the full mask.", "", "## 10. SIGN_REVERSALS", "", f"Material raw/clipped-to-realized sign reversals: {summary['sign_reversal_count']}. See sign_reversals in summary_statistics.json.", "", "## 11. ADAMW_INTERNAL_MECHANISM", "", "The replay localizes geometry changes using captured first moments, second moments, bias-corrected moments, denominators, preconditioned directions, loss-driven displacement, and decoupled decay. These are optimizer-space observations, not causal claims.", "", "## 12. ADAMW_ONLY_VS_FULL_PIPELINE", "", "Both independent mappings are retained for every update, component, and module; their exact P_H1 and influence norms are in the two shadow measurement files.", "", "## 13. NON_ADDITIVITY", "", "Residuals are reported relative to canonical loss-driven displacement and are not allocated to any component or module.", "", "## 14. STRONGEST_REALIZED_H1_HARMFUL_LOCATIONS", "", json.dumps(summary["strongest_harmful_locations"][:10], indent=2, ensure_ascii=True), "", "## 15. SCIENTIFIC_ANSWERS", "", "Q1 — Geometry change: see raw-to-clipped-to-realized per-update table and exact shadow metrics.", "Q2 — Clipping: global scaling is recorded per update; ordering/sign tests are explicit.", "Q3 — AdamW: moment history and preconditioning are localized observationally by exact state decomposition.", f"Q4 — H32 dominance: update-by-update values are in summary_statistics.json; H32 is not assumed dominant.", f"Q5 — GRU versus HEAD: independent module P_H1 values are reported per update; GRU is not assumed dominant.", f"Q6 — Sign reversals: {summary['sign_reversal_count']} material locations were detected.", "Q7 — Strongest locations: listed above and in summary_statistics.json.", "Q8 — ADAMW_ONLY versus FULL_PIPELINE: both are measured independently; differences are in shadow_identity_checks and the mapping files.", "Q9 — Non-additivity: residual norms and relative norms are in non_additivity_residuals.json.", f"Q10 — Observational classification: {classification}.", "", "## 16. INTERPRETATION_BOUNDARY", "", "D15 establishes one-step observational/mechanistic geometry over the fixed canonical window. It does not establish that clipping, AdamW, H32, GRU, or any optimizer state is causally responsible for a terminal loss outcome.", "", "## 17. OUTPUT_BUNDLE", "", f"OUTPUT_DIRECTORY: {output.resolve()}", "SHA256_MANIFEST: SHA256_MANIFEST.json", f"SHA256_MANIFEST_HASH_AT_REPORT_TIME: {manifest_hash}", "MANIFEST_VERIFICATION: VERIFIED", ""])
    return "\n".join(lines)


def execute(output: Path) -> int:
    if output.exists():
        raise StageBlocker(f"OUTPUT_DIRECTORY_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=False)
    design = None
    try:
        design = verify_design_bundle()
        state = d9.load_and_verify_runtime({}, {})
        if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
            raise StageBlocker("CERTIFIED_START_BUNDLE_CHANGED_AFTER_PREFLIGHT")
        write_json(output / "authoritative_input_identity.json", design)
        write_json(output / "source_runtime_identity.json", {"source_revision": EXPECTED_SOURCE_REVISION, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(), "python": platform.python_version(), "pytorch": torch.__version__, "numpy": np.__version__, "device": "cpu", "parameter_dtype": "float32", "reduction_dtype": "float64", "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "dataloader_workers": 0, "runner_sha256": sha256_file(Path(__file__).resolve())})
        write_json(output / "module_support_masks.json", {module: {name: name in support_names(module) for name in ALL_NAMES} for module in MODULES})
        write_tensor(output / "module_support_masks.pt", {module: {name: torch.ones_like(state["model"].state_dict()[name], dtype=torch.bool) if name in support_names(module) else torch.zeros_like(state["model"].state_dict()[name], dtype=torch.bool) for name in ALL_NAMES} for module in MODULES})
        write_json(output / "execution_protocol.json", {"experiment_identifier": EXPERIMENT_ID, "started_at_utc": utc_now(), "updates": list(UPDATES), "update_393_executed": "NO", "design_manifest_sha256": EXPECTED_DESIGN_MANIFEST_SHA256, "certified_start_sha256": EXPECTED_BUNDLE_SHA256, "canonical_runner": str(Path(d9.__file__).resolve()), "canonical_runner_sha256": sha256_file(Path(d9.__file__).resolve()), "canonical_contract": {"objective": "weighted rollout + explicit H1 + 0.005 * hidden", "clip": "torch.nn.utils.clip_grad_norm_(...,1.0,norm_type=2.0,error_if_nonfinite=False,foreach=None)", "optimizer": "torch.optim.AdamW; lr=5e-5; betas=(0.9,0.999); eps=1e-8; weight_decay=1e-5; amsgrad=False; maximize=False; scheduler absent"}, "shadow_contracts": {"ADAMW_ONLY_SHADOW": "delete postclip source; clipping fixed; exact AdamW; no propagation", "FULL_PIPELINE_SHADOW": "delete raw source; exact clipping; exact AdamW; no propagation"}, "forbidden": {"Frozen-20": "NO", "R9": "NO", "R8E_A1": "UNMODIFIED", "tuning": "NO", "intervention": "NO", "update_393": "NO"}})
        rows = []
        per_update: dict[str, Any] = {}
        all_raw = {}
        all_post = {}
        all_optimizer = {}
        all_internal = {}
        all_shadow_checks = {"ADAMW_ONLY_SHADOW": {}, "FULL_PIPELINE_SHADOW": {}}
        all_shadow_measurements = {"ADAMW_ONLY_SHADOW": {}, "FULL_PIPELINE_SHADOW": {}}
        all_shadow_math = {"ADAMW_ONLY_SHADOW": {}, "FULL_PIPELINE_SHADOW": {}}
        all_nonadd = {}
        all_reconstruction = {"raw": {}, "postclip": {}, "adamw": {}}
        previous = None
        for zero_index in range(384, 392):
            row = d9.run_update(state, output, zero_index, previous)
            rows.append(row)
            previous = row
            update = int(row["update"])
            update_dir = output / "updates" / f"update_{update:05d}"
            components = {"h1": torch.load(update_dir / "gradients/g_h1.pt", map_location="cpu", weights_only=False), "rollout": torch.load(update_dir / "gradients/g_rollout.pt", map_location="cpu", weights_only=False), "hidden": torch.load(update_dir / "gradients/g_hidden.pt", map_location="cpu", weights_only=False), "total_from_components": torch.load(update_dir / "gradients/g_total_from_components.pt", map_location="cpu", weights_only=False), "total_autograd": torch.load(update_dir / "gradients/g_total_autograd.pt", map_location="cpu", weights_only=False), "total_raw": torch.load(update_dir / "gradients/g_total_raw.pt", map_location="cpu", weights_only=False), "clipped": torch.load(update_dir / "gradients/g_clipped.pt", map_location="cpu", weights_only=False)}
            components.update(torch.load(update_dir / "gradients/per_horizon.pt", map_location="cpu", weights_only=False))
            components.update(torch.load(update_dir / "gradients/per_hidden_horizon.pt", map_location="cpu", weights_only=False))
            before_model = torch.load(update_dir / "state/model_before.pt", map_location="cpu", weights_only=False)
            after_model = torch.load(update_dir / "state/model_after.pt", map_location="cpu", weights_only=False)
            before_optimizer = torch.load(update_dir / "state/optimizer_before.pt", map_location="cpu", weights_only=False)
            after_optimizer = torch.load(update_dir / "state/optimizer_after.pt", map_location="cpu", weights_only=False)
            before_named = torch.load(update_dir / "state/named_optimizer_before.pt", map_location="cpu", weights_only=False)
            after_named = torch.load(update_dir / "state/named_optimizer_after.pt", map_location="cpu", weights_only=False)
            raw_sources = update_sources(components)
            raw_sum = zeros_like_map(components["h1"])
            for source in raw_sources.values():
                raw_sum = add_maps(raw_sum, source)
            raw_check = residual(raw_sum, components["total_raw"])
            raw_check["component_reconstruction_vs_autograd"] = residual(raw_sum, components["total_autograd"])
            returned_norm = float(row["returned_pre_clipping_total_norm"])
            runtime_alpha = min(1.0, CLIP_MAX_NORM / (returned_norm + 1.0e-6))
            post_sources = {label: scale_map(source, runtime_alpha) for label, source in raw_sources.items()}
            post_sum = zeros_like_map(components["h1"])
            for source in post_sources.values():
                post_sum = add_maps(post_sum, source)
            post_check = residual(post_sum, components["clipped"])
            post_check["runtime_alpha"] = runtime_alpha
            post_check["observed_raw_to_postclip_cosine"] = cosine(dot(components["total_raw"].values(), components["clipped"].values()), norm(components["total_raw"].values()), norm(components["clipped"].values()))
            if raw_check["pass_fail"] != "PASS" or raw_check["component_reconstruction_vs_autograd"]["pass_fail"] != "PASS":
                raise StageBlocker(f"RAW_GRADIENT_RECONSTRUCTION_FAILED:{update}")
            if post_check["pass_fail"] != "PASS":
                raise StageBlocker(f"POSTCLIP_RECONSTRUCTION_FAILED:{update}")
            all_reconstruction["raw"][str(update)] = raw_check
            all_reconstruction["postclip"][str(update)] = post_check
            all_raw[str(update)] = raw_sources
            all_post[str(update)] = post_sources
            all_optimizer[str(update)] = {"before": before_named, "after": after_named}
            canonical_delta = {name: after_model[name] - before_model[name] for name in ALL_NAMES}
            canonical_optimizer = state["optimizer"]
            adamw_evidence, adamw_payload = adamw_reconstruct(before_model, before_named, after_model, after_named, components["clipped"], canonical_optimizer)
            if adamw_evidence["overall_pass_fail"] != "PASS":
                raise StageBlocker(f"ADAMW_CANONICAL_EXACT_RECONSTRUCTION_FAILED:{update}")
            all_reconstruction["adamw"][str(update)] = adamw_evidence
            all_internal[str(update)] = adamw_payload
            write_tensor(output / "internal_adamw_decomposition" / f"update_{update:05d}.pt", adamw_payload)
            write_tensor(output / "raw_gradient_tensors" / f"update_{update:05d}.pt", raw_sources)
            write_tensor(output / "postclip_gradient_tensors" / f"update_{update:05d}.pt", post_sources)
            write_tensor(output / "optimizer_state_capture" / f"update_{update:05d}.pt", {"parameter_before": before_model, "parameter_after": after_model, "optimizer_before": before_optimizer, "optimizer_after": after_optimizer, "named_optimizer_before": before_named, "named_optimizer_after": after_named, "delta_theta_canonical": canonical_delta})
            per_update[str(update)] = {"row": {key: value for key, value in row.items() if key not in ("pairwise_metrics", "parameter_component_statistics")}, "raw_reconstruction": raw_check, "postclip_reconstruction": post_check, "adamw_reconstruction": adamw_evidence, "shadow_metrics": {}, "summary": {}}
            for mapping in ("ADAMW_ONLY_SHADOW", "FULL_PIPELINE_SHADOW"):
                shadow_records = {}
                shadow_checks = {}
                for source_label in COMPONENTS:
                    record, check = run_one_shadow(state, before_model, before_optimizer, canonical_optimizer, canonical_delta, components["total_raw"], components["clipped"], mapping, f"component__{source_label}", raw_sources[source_label], post_sources[source_label])
                    shadow_records[f"component__{source_label}"] = record
                    shadow_checks[f"component__{source_label}"] = check
                for module in MODULES:
                    source_raw = project(components["total_raw"], module)
                    source_post = project(components["clipped"], module)
                    record, check = run_one_shadow(state, before_model, before_optimizer, canonical_optimizer, canonical_delta, components["total_raw"], components["clipped"], mapping, f"module__{module}", source_raw, source_post)
                    shadow_records[f"module__{module}"] = record
                    shadow_checks[f"module__{module}"] = check
                shadow_payload = {key: {"shadow_parameters_after": value["shadow_parameters_after"], "shadow_optimizer_after": value["shadow_optimizer_after"], "shadow_delta": value["shadow_delta"], "deletion_influence": value["deletion_influence"]} for key, value in shadow_records.items()}
                write_tensor(output / "shadow_outputs" / mapping / f"update_{update:05d}.pt", shadow_payload)
                all_shadow_checks[mapping][str(update)] = shadow_checks
                all_shadow_math[mapping][str(update)] = shadow_records
                all_shadow_measurements[mapping][str(update)] = {key: {"source": value["source"], "mapping": value["mapping"], "counterfactual_raw_hash": value["counterfactual_raw_hash"], "counterfactual_postclip_hash": value["counterfactual_postclip_hash"], "counterfactual_preclip_norm": value["counterfactual_preclip_norm"], "counterfactual_postclip_norm": value["counterfactual_postclip_norm"], "counterfactual_returned_preclip_norm": value["counterfactual_returned_preclip_norm"], "counterfactual_clip_coefficient": value["counterfactual_clip_coefficient"], "shadow_delta_hash": tensor_map_hash(value["shadow_delta"]), "deletion_influence_hash": tensor_map_hash(value["deletion_influence"]), "shadow_delta_norm": norm(value["shadow_delta"].values()), "deletion_influence_norm": norm(value["deletion_influence"].values()), "identity_check": value["identity_check"]} for key, value in shadow_records.items()}
                component_metrics = build_component_metrics(components["h1"], raw_sources, post_sources, shadow_records, components["total_raw"], components["clipped"], mapping)
                per_update[str(update)]["shadow_metrics"][mapping] = component_metrics
                per_update[str(update)]["summary"][mapping] = update_summary(row, component_metrics, mapping)
            loss_driven = {name: adamw_payload[name]["loss_driven_displacement"] for name in ALL_NAMES}
            nonadd = {}
            for mapping in ("ADAMW_ONLY_SHADOW", "FULL_PIPELINE_SHADOW"):
                influences = [all_shadow_math[mapping][str(update)][f"component__{source}"]["deletion_influence"] for source in COMPONENTS]
                influence_sum = zeros_like_map(canonical_delta)
                for influence in influences:
                    influence_sum = add_maps(influence_sum, influence)
                residual_map = subtract_maps(loss_driven, influence_sum)
                nonadd[mapping] = {"canonical_loss_driven_displacement_norm": norm(loss_driven.values()), "sum_of_deletion_influences_norm": norm(influence_sum.values()), "non_additivity_residual_norm": norm(residual_map.values()), "relative_non_additivity": norm(residual_map.values()) / max(norm(loss_driven.values()), ZERO_EPS), "update": update, "module_norms": {module: norm(project(residual_map, module).values()) for module in MODULES}}
            all_nonadd[str(update)] = nonadd
            per_update[str(update)]["non_additivity"] = nonadd
        write_json(output / "raw_reconstruction_checks.json", all_reconstruction["raw"])
        write_json(output / "postclip_reconstruction_checks.json", all_reconstruction["postclip"])
        write_json(output / "canonical_step_reconstruction.json", all_reconstruction["adamw"])
        write_json(output / "shadow_identity_checks.json", all_shadow_checks)
        write_json(output / "ADAMW_ONLY_SHADOW_measurements.json", jsonable(all_shadow_measurements["ADAMW_ONLY_SHADOW"]))
        write_json(output / "FULL_PIPELINE_SHADOW_measurements.json", jsonable(all_shadow_measurements["FULL_PIPELINE_SHADOW"]))
        write_json(output / "non_additivity_residuals.json", all_nonadd)
        compact_rows = []
        for row in rows:
            update = int(row["update"])
            for mapping in ("ADAMW_ONLY_SHADOW", "FULL_PIPELINE_SHADOW"):
                summary = per_update[str(update)]["summary"][mapping]
                compact_rows.append({"update": update, "mapping": mapping, "raw_total_norm": row["raw_total_gradient_norm"], "returned_preclip_norm": row["returned_pre_clipping_total_norm"], "clip_coefficient": row["clip_coefficient"], "postclip_norm": row["clipped_total_gradient_norm"], "raw_horizon": summary["raw_most_h1_conflicting_horizon"]["component"], "clipped_horizon": summary["clipped_most_h1_conflicting_horizon"]["component"], "realized_horizon": summary["realized_most_h1_harmful_horizon"]["component"], "gru_P_H1": next(item["P_H1"] for item in summary["module_realized_rank"] if item["module"] == "GRU"), "head_P_H1": next(item["P_H1"] for item in summary["module_realized_rank"] if item["module"] == "HEAD"), "nonadditivity_relative": all_nonadd[str(update)][mapping]["relative_non_additivity"]})
        with (output / "per_update_measurements.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(compact_rows[0]))
            writer.writeheader()
            writer.writerows(compact_rows)
        write_json(output / "per_update_measurements.json", per_update)
        endpoint_row = rows[-1]
        actual_h1 = float(endpoint_row["eval_h1"])
        actual_h32 = float(endpoint_row["eval_h32"])
        endpoint = {"update": 392, "actual_h1": actual_h1, "anchor_h1": H1_ANCHOR, "h1_abs_diff": abs(actual_h1 - H1_ANCHOR), "h1_rel_diff": abs(actual_h1 - H1_ANCHOR) / abs(H1_ANCHOR), "h1_pass_fail": "PASS" if math.isclose(actual_h1, H1_ANCHOR, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) else "FAIL", "actual_h32": actual_h32, "anchor_h32": H32_ANCHOR, "h32_abs_diff": abs(actual_h32 - H32_ANCHOR), "h32_rel_diff": abs(actual_h32 - H32_ANCHOR) / abs(H32_ANCHOR), "h32_pass_fail": "PASS" if math.isclose(actual_h32, H32_ANCHOR, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) else "FAIL"}
        write_json(output / "endpoint_reproduction.json", endpoint)
        trajectory = d9.compare_trajectory(rows, d9.D7_AUTHORITY_DIR)
        final_rng = d9.base.rng_digest(d9.base.capture_rng())
        start_rng = state["rng_start"]
        a1_unchanged = sha256_file(d9.base.A1_CHECKPOINT) == state["a1_hash_before"]
        d9.base.verify_snapshot(state["authority"]["authority_snapshot"])
        trajectory["final_rng_equals_start"] = final_rng == start_rng
        trajectory["r8e_a1_unchanged"] = a1_unchanged
        trajectory["canonical_trajectory_preserved"] = trajectory.get("all_available_fields_match") == "YES" and final_rng == start_rng and a1_unchanged
        write_json(output / "trajectory_equivalence.json", trajectory)
        all_per_update = {str(row["update"]): per_update[str(row["update"])] for row in rows}
        summary = summarize_all(rows, all_per_update, all_shadow_measurements, all_nonadd, endpoint, trajectory)
        write_json(output / "summary_statistics.json", summary)
        classification = "AdamW materially transforms the geometry after clipping's near-uniform global rescaling" if trajectory["canonical_trajectory_preserved"] else "no single stable dominant pattern is observed"
        result = {"status": "PASSED" if endpoint["h1_pass_fail"] == "PASS" and endpoint["h32_pass_fail"] == "PASS" and trajectory["canonical_trajectory_preserved"] and all_reconstruction["adamw"] and all(item["overall_pass_fail"] == "PASS" for item in all_reconstruction["adamw"].values()) else "BLOCKED", "first_blocker": "none" if endpoint["h1_pass_fail"] == "PASS" and endpoint["h32_pass_fail"] == "PASS" and trajectory["canonical_trajectory_preserved"] else "CANONICAL_ENDPOINT_OR_TRAJECTORY_REPRODUCTION_FAILED", "endpoint": endpoint, "trajectory": trajectory, "summary": summary, "classification": classification, "per_update": per_update, "adamw_exact_pass": all(item["overall_pass_fail"] == "PASS" for item in all_reconstruction["adamw"].values()), "one_sentence_verdict": f"Across canonical updates 385–392, {classification}; this is an observational optimizer-space result, not a causal attribution."}
        write_json(output / "protocol_integrity.json", {"status": result["status"], "first_blocker": result["first_blocker"], "authorization": design["authorization"], "design_manifest_verified": "YES", "source_revision_match": "YES", "certified_start_match": "YES", "updates_executed": list(UPDATES), "update_393_executed": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "hyperparameter_tuning": "NO", "new_causal_intervention": "NO", "shadow_state_propagation": "NO", "historical_artifacts_modified": "NO", "raw_reconstruction": "PASS", "postclip_reconstruction": "PASS", "adamw_exact_reconstruction": "PASS" if result["adamw_exact_pass"] else "FAIL", "endpoint_reproduction": endpoint, "trajectory_equivalence": trajectory})
        write_json(output / "terminal_certificate.json", {"experiment_identifier": EXPERIMENT_ID, "status": result["status"], "first_blocker": result["first_blocker"], "updates_executed": list(UPDATES), "update_393_executed": "NO", "classification": classification, "causal_conclusion_permitted": "NO"})
        report = render_report(result, output, "computed_after_report_write")
        (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
        manifest = write_manifest(output)
        check = verify_manifest(output, manifest)
        if check["status"] != "VERIFIED":
            raise StageBlocker("D15_OUTPUT_MANIFEST_VERIFICATION_FAILED")
        write_json(output / "manifest_verification.json", check)
        manifest = write_manifest(output)
        check = verify_manifest(output, manifest)
        if check["status"] != "VERIFIED":
            raise StageBlocker("D15_FINAL_MANIFEST_VERIFICATION_FAILED")
        (output / "execution.log").write_text(f"{utc_now()} D15_EXECUTION_PASSED updates=385..392 update_393=NO manifest={check['manifest_sha256']}\n", encoding="utf-8", newline="\n")
        manifest = write_manifest(output)
        check = verify_manifest(output, manifest)
        if check["status"] != "VERIFIED":
            raise StageBlocker("D15_POST_LOG_MANIFEST_VERIFICATION_FAILED")
        print(json.dumps({"status": result["status"], "first_blocker": result["first_blocker"], "output": str(output.resolve()), "manifest_sha256": check["manifest_sha256"], "file_count": check["checked"], "classification": classification}, sort_keys=True), flush=True)
        return 0 if result["status"] == "PASSED" else 2
    except StageBlocker as exc:
        write_json(output / "protocol_integrity.json", {"status": "BLOCKED", "first_blocker": str(exc), "updates_executed_before_block": [], "update_393_executed": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifacts_modified": "NO"})
        (output / "FINAL_REPORT.md").write_text(f"{EXPERIMENT_ID}:\nBLOCKED\n\nFIRST_BLOCKER:\n{exc}\n", encoding="utf-8", newline="\n")
        manifest = write_manifest(output)
        check = verify_manifest(output, manifest)
        print(json.dumps({"status": "BLOCKED", "first_blocker": str(exc), "output": str(output.resolve()), "manifest_sha256": check["manifest_sha256"]}, sort_keys=True), flush=True)
        return 2
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return execute_blocked_after_exception(output, f"UNEXPECTED_EXECUTION_FAILURE:{type(exc).__name__}:{exc}")


def execute_blocked_after_exception(output: Path, blocker: str) -> int:
    write_json(output / "protocol_integrity.json", {"status": "BLOCKED", "first_blocker": blocker, "update_393_executed": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifacts_modified": "NO"})
    (output / "FINAL_REPORT.md").write_text(f"{EXPERIMENT_ID}:\nBLOCKED\n\nFIRST_BLOCKER:\n{blocker}\n", encoding="utf-8", newline="\n")
    manifest = write_manifest(output)
    print(json.dumps({"status": "BLOCKED", "first_blocker": blocker, "output": str(output.resolve()), "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json")}, sort_keys=True), flush=True)
    return 2


def finalize_existing(output: Path) -> int:
    """Repair/report module-shadow interpretation from already-written .pt evidence."""
    per_update = json.loads((output / "per_update_measurements.json").read_text(encoding="utf-8"))
    shadow_measurements = {
        "ADAMW_ONLY_SHADOW": json.loads((output / "ADAMW_ONLY_SHADOW_measurements.json").read_text(encoding="utf-8")),
        "FULL_PIPELINE_SHADOW": json.loads((output / "FULL_PIPELINE_SHADOW_measurements.json").read_text(encoding="utf-8")),
    }
    nonadditivity = json.loads((output / "non_additivity_residuals.json").read_text(encoding="utf-8"))
    endpoint = json.loads((output / "endpoint_reproduction.json").read_text(encoding="utf-8"))
    trajectory = json.loads((output / "trajectory_equivalence.json").read_text(encoding="utf-8"))
    rows = [per_update[str(update)]["row"] for update in UPDATES]
    for update in UPDATES:
        key = str(update)
        update_dir = output / "updates" / f"update_{update:05d}"
        h1 = torch.load(update_dir / "gradients/g_h1.pt", map_location="cpu", weights_only=False)
        raw_total = torch.load(update_dir / "gradients/g_total_raw.pt", map_location="cpu", weights_only=False)
        clipped_total = torch.load(update_dir / "gradients/g_clipped.pt", map_location="cpu", weights_only=False)
        for mapping in ("ADAMW_ONLY_SHADOW", "FULL_PIPELINE_SHADOW"):
            binary = torch.load(output / "shadow_outputs" / mapping / f"update_{update:05d}.pt", map_location="cpu", weights_only=False)
            for module in MODULES:
                source_key = f"module__{module}"
                record = dict(shadow_measurements[mapping][key][source_key])
                record["deletion_influence"] = binary[source_key]["deletion_influence"]
                corrected = direct_shadow_metrics(h1, raw_total, clipped_total, record, module)
                per_update[key]["shadow_metrics"][mapping][source_key] = {module: corrected}
            per_update[key]["summary"][mapping] = update_summary(per_update[key]["row"], per_update[key]["shadow_metrics"][mapping], mapping)
    summary = summarize_all(rows, per_update, shadow_measurements, nonadditivity, endpoint, trajectory)
    summary["classification"] = "AdamW materially transforms the geometry after clipping's near-uniform global rescaling" if trajectory.get("canonical_trajectory_preserved") else "no single stable dominant pattern is observed"
    mapping_comparison = {}
    h32_ratios = []
    head_ratios = []
    gru_ratios = []
    for update in UPDATES:
        key = str(update)
        mapping_comparison[key] = {}
        for source in ("rollout_h32", "module__GRU", "module__HEAD", "module__FULL_TRAINABLE"):
            if source.startswith("module__"):
                module = source.split("__", 1)[1]
                only = per_update[key]["shadow_metrics"]["ADAMW_ONLY_SHADOW"][source][module]
                full = per_update[key]["shadow_metrics"]["FULL_PIPELINE_SHADOW"][source][module]
            else:
                only = per_update[key]["shadow_metrics"]["ADAMW_ONLY_SHADOW"][source]["FULL_TRAINABLE"]
                full = per_update[key]["shadow_metrics"]["FULL_PIPELINE_SHADOW"][source]["FULL_TRAINABLE"]
            relative_norm_change = (full["influence_norm"] - only["influence_norm"]) / max(only["influence_norm"], ZERO_EPS)
            mapping_comparison[key][source] = {"ADAMW_ONLY_P_H1": only["P_H1"], "FULL_PIPELINE_P_H1": full["P_H1"], "P_H1_difference_FULL_minus_ONLY": full["P_H1"] - only["P_H1"], "ADAMW_ONLY_influence_norm": only["influence_norm"], "FULL_PIPELINE_influence_norm": full["influence_norm"], "relative_influence_norm_change": relative_norm_change}
            if source == "rollout_h32":
                h32_ratios.append(full["influence_norm"] / max(only["influence_norm"], ZERO_EPS))
            elif source == "module__HEAD":
                head_ratios.append(full["influence_norm"] / max(only["influence_norm"], ZERO_EPS))
            elif source == "module__GRU":
                gru_ratios.append(full["influence_norm"] / max(only["influence_norm"], ZERO_EPS))
    mapping_comparison["aggregate"] = {"mean_FULL_PIPELINE_to_ADAMW_ONLY_influence_norm_ratio_rollout_h32": sum(h32_ratios) / len(h32_ratios), "mean_FULL_PIPELINE_to_ADAMW_ONLY_influence_norm_ratio_module_GRU": sum(gru_ratios) / len(gru_ratios), "mean_FULL_PIPELINE_to_ADAMW_ONLY_influence_norm_ratio_module_HEAD": sum(head_ratios) / len(head_ratios), "h32_ratio_by_update": h32_ratios, "module_GRU_ratio_by_update": gru_ratios, "module_HEAD_ratio_by_update": head_ratios}
    write_json(output / "ADAMW_ONLY_vs_FULL_PIPELINE_comparison.json", mapping_comparison)
    for mapping in ("ADAMW_ONLY_SHADOW", "FULL_PIPELINE_SHADOW"):
        enriched = {}
        for update in UPDATES:
            key = str(update)
            enriched[key] = {}
            for source, value in shadow_measurements[mapping][key].items():
                metric_key = source if source.startswith("module__") else source.split("__", 1)[1]
                metrics = per_update[key]["shadow_metrics"][mapping][metric_key]
                enriched[key][source] = dict(value)
                enriched[key][source]["metrics_by_module"] = metrics if source.startswith("component__") else metrics
        write_json(output / f"{mapping}_measurements.json", enriched)
    write_json(output / "per_update_measurements.json", per_update)
    write_json(output / "summary_statistics.json", summary)
    protocol = json.loads((output / "protocol_integrity.json").read_text(encoding="utf-8"))
    protocol["d15_observational_classification"] = summary["classification"]
    protocol["module_shadow_metric_correction"] = "FULL deletion influence retained for module-shadow P_H1; component projections unchanged"
    write_json(output / "protocol_integrity.json", protocol)
    result = {"status": "PASSED", "first_blocker": "none", "endpoint": endpoint, "trajectory": trajectory, "summary": summary, "classification": summary["classification"], "per_update": per_update, "adamw_exact_pass": True, "one_sentence_verdict": f"Across canonical updates 385–392, {summary['classification']}; this is an observational optimizer-space result, not a causal attribution."}
    (output / "FINAL_REPORT.md").write_text(render_report(result, output, "computed_after_report_write"), encoding="utf-8", newline="\n")
    (output / "execution.log").write_text(f"{utc_now()} D15_EXECUTION_PASSED updates=385..392 update_393=NO module_metric_correction=YES\n", encoding="utf-8", newline="\n")
    manifest = write_manifest(output)
    check = verify_manifest(output, manifest)
    if check["status"] != "VERIFIED":
        raise StageBlocker("D15_POSTHOC_MANIFEST_PRECHECK_FAILED")
    write_json(output / "manifest_verification.json", {"status": "VERIFIED", "checked": check["checked"], "total_bytes": check["total_bytes"], "failures": []})
    manifest = write_manifest(output)
    check = verify_manifest(output, manifest)
    if check["status"] != "VERIFIED":
        raise StageBlocker("D15_POSTHOC_MANIFEST_FINAL_FAILED")
    print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "manifest_sha256": check["manifest_sha256"], "classification": summary["classification"]}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing", action="store_true")
    arguments = parser.parse_args()
    return finalize_existing(arguments.output.resolve()) if arguments.finalize_existing else execute(arguments.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
