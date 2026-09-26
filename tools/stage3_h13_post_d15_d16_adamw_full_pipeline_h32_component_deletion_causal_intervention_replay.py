"""Execute the sealed D16 full-pipeline H32 deletion replay.

This runner creates two independent model-plus-optimizer branches from the
authenticated post-update-384 bundle.  CONTROL uses the sealed objective;
INTERVENTION removes only the weighted rollout-H32 scalar before backward.
Both branches then perform their own backward pass, exact global clipping,
real AdamW step, state propagation, and endpoint validation for updates
385--392.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402
from src.stage3_h13_r8e import rollout_horizon_weights  # noqa: E402
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256  # noqa: E402
from tools import stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay as d7  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D15_D16_ADAMW_FULL_PIPELINE_H32_COMPONENT_DELETION_CAUSAL_INTERVENTION_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_design_review_20260819T234116+0800"
D15_DIR = ROOT / "outputs/stage3_h13_post_d14_d15_adamw_realized_update_space_interaction_instrumentation_replay_20260819T203424+0800_retry3"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
SCHEDULE_IDENTITY = ROOT / "outputs/stage3_h13_post_d4_canonical_batch_robustness_20260816T101500Z/canonical_schedule_identity.json"
SCHEDULE_STORAGE = ROOT / "outputs/stage3_h13_post_d4_canonical_batch_robustness_20260816T101500Z/canonical_schedule_manifest.json.gz"
CLASSIFICATION_CONTRACT = ROOT / "outputs/stage3_h13_post_d13_d14_gru_h32_temporal_conflict_accumulation_causal_intervention_design_review_20260819T183000+0800/classification_contract.json"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\5ab0f32c-89fc-469f-a6f7-c69742fd3a7a\pasted-text.txt")

EXPECTED_DESIGN_MANIFEST_SHA256 = "cd63ff0ab20eb78c3966d7c44640457d7673ce28961e7272434e7d12838e8412"
EXPECTED_D15_MANIFEST_SHA256 = "fdc0219f6b036e3097dc79aebee3c8fedb93844d2fef7ed2a0b34f91080edcca"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_EXP_AVG_HASH = "7226e7ae693c1083d6928df6968d67d32d33477c16a8770adcf3050fc9492cdb"
EXPECTED_EXP_AVG_SQ_HASH = "eaf5ebdd6aaddefec25950e80d1a14baeb92a06dddcd11b246546e44c8489d89"
EXPECTED_RNG_HASH = "857d5ddd6c4271019b55a53774ad68c70e435abd38cf430b7f0d57ed047848d8"
EXPECTED_SCHEDULE_IDENTITY_SHA256 = "db72f9de0c4f3e4fbbfec0c313b62184b1147645bc94c23c7f4b59b9e05d9d1f"
EXPECTED_SCHEDULE_STORAGE_SHA256 = "631f3a10e6e2ce4497e4fa08be708ace0528a7073b8e3d245fa4b97518882a9f"
EXPECTED_SCHEDULE_LOGICAL_SHA256 = "e1e0d681d84c75348c656f306eedfd61bed849a1ff1c0b966f5ff3a63145ebcd"
EXPECTED_CLASSIFICATION_CONTRACT_SHA256 = "16497f895a4fac479ff6f946fd5bd4e6869eef68859e2af94c4f22229f9dfbe7"

UPDATES = tuple(range(385, 393))
START_UPDATE = 384
ACTIVE_HORIZON = 32
LAMBDA_ROLLOUT = 1.0
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
POSITION_BETA = 0.001
CLIP_MAX_NORM = 1.0
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
H1_MATERIAL_EFFECT_THRESHOLD = 4.099006815405639e-6
H1_GUARDRAIL = 0.0000859791431235184
H32_PASS_THRESHOLD = 0.01856902565856056
ADDITIVITY_ABS = 1.0e-7
ADDITIVITY_REL = 2.0e-5

ALL_NAMES = (
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1",
    "head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias",
)
GRU_NAMES = ALL_NAMES[:8]
HEAD_NAMES = ALL_NAMES[8:]


class StageBlocker(RuntimeError):
    """Fail-closed D16 execution blocker."""


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


def finite_float(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise StageBlocker(f"NONFINITE_MEASUREMENT:{result}")
    return result


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}) + b"\0" + tensor.numpy().tobytes())


def tensor_map_hash(values: Mapping[str, torch.Tensor]) -> str:
    return sha256_bytes(canonical_json([
        {"name": name, "dtype": str(values[name].dtype), "shape": list(values[name].shape), "sha256": tensor_hash(values[name])}
        for name in values
    ]))


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_value(item) for item in value)
    return copy.deepcopy(value)


def clone_parameters(model: Any) -> dict[str, torch.Tensor]:
    values = {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if list(values) != list(ALL_NAMES):
        raise StageBlocker("PARAMETER_ORDER_MISMATCH")
    return values


def clone_named_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    result: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise StageBlocker("OPTIMIZER_PARAMETER_NAME_MAPPING_MISSING")
            result[name] = {str(key): clone_value(value) for key, value in optimizer.state.get(parameter, {}).items()}
    if list(result) != list(ALL_NAMES):
        raise StageBlocker("OPTIMIZER_PARAMETER_ORDER_MISMATCH")
    return result


def norm(values: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for value in values:
        total += finite_float(torch.sum(value.detach().double().square()))
    return finite_float(math.sqrt(max(total, 0.0)))


def difference_norm(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return norm(left[name] - right[name] for name in ALL_NAMES)


def block_norms(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        "GRU": norm(values[name] for name in GRU_NAMES),
        "HEAD": norm(values[name] for name in HEAD_NAMES),
        "FULL_TRAINABLE": norm(values[name] for name in ALL_NAMES),
    }


def capture_gradients(named: Sequence[tuple[str, torch.nn.Parameter]]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    values: dict[str, torch.Tensor] = {}
    flags: dict[str, str] = {}
    for name, parameter in named:
        if parameter.grad is None:
            values[name] = torch.zeros_like(parameter.detach()).cpu().contiguous()
            flags[name] = "None"
        else:
            values[name] = parameter.grad.detach().cpu().contiguous().clone()
            flags[name] = "connected"
    for value in values.values():
        if not bool(torch.all(torch.isfinite(value))):
            raise StageBlocker("NONFINITE_GRADIENT")
    return values, flags


def diagnostic_gradients(scalar: torch.Tensor, named: Sequence[tuple[str, torch.nn.Parameter]]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    grads = torch.autograd.grad(scalar, [parameter for _, parameter in named], retain_graph=True, create_graph=False, allow_unused=True)
    values: dict[str, torch.Tensor] = {}
    flags: dict[str, str] = {}
    for (name, parameter), gradient in zip(named, grads):
        if gradient is None:
            values[name] = torch.zeros_like(parameter.detach()).cpu().contiguous()
            flags[name] = "None"
        else:
            values[name] = gradient.detach().cpu().contiguous().clone()
            flags[name] = "connected"
    return values, flags


def optimizer_steps(named_state: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in ALL_NAMES:
        raw = named_state[name].get("step")
        if raw is None:
            raise StageBlocker(f"OPTIMIZER_STEP_MISSING:{name}")
        result[name] = int(raw.item()) if isinstance(raw, torch.Tensor) else int(raw)
    return result


def moment_map(named_state: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, torch.Tensor]:
    result = {}
    for name in ALL_NAMES:
        value = named_state[name].get(key)
        if not isinstance(value, torch.Tensor):
            raise StageBlocker(f"OPTIMIZER_MOMENT_MISSING:{key}:{name}")
        result[name] = value.detach().cpu().contiguous().clone()
    return result


def state_finite(model: Any, optimizer: torch.optim.Optimizer) -> bool:
    if any(not bool(torch.all(torch.isfinite(parameter.detach()))) for parameter in model.parameters()):
        return False
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.all(torch.isfinite(value))):
                return False
    return True


def batch_identity(schedule_manifest: Mapping[str, Any], index: int) -> dict[str, Any]:
    item = schedule_manifest["ordered_batches"][index]
    ids = [int(value) for value in item["window_ids"]]
    return {
        "update": index + 1,
        "schedule_index": index,
        "window_ids": ids,
        "window_id_count": len(ids),
        "batch_window_sha256": base.batch_digest(schedule_manifest, index),
        "source_cycle_spans": item.get("source_cycle_spans"),
    }


def verify_manifest_file(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise StageBlocker(f"MANIFEST_MISSING:{path}")
    observed_manifest_sha256 = sha256_file(path)
    if expected_sha256 is not None and observed_manifest_sha256 != expected_sha256:
        raise StageBlocker(f"MANIFEST_HASH_MISMATCH:{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    for name, entry in manifest.get("files", {}).items():
        target = path.parent / name
        if not target.is_file():
            failures.append({"path": name, "reason": "missing"})
            continue
        expected = entry["sha256"] if isinstance(entry, Mapping) else entry
        expected_bytes = entry.get("bytes") if isinstance(entry, Mapping) else None
        if sha256_file(target) != expected or expected_bytes is not None and target.stat().st_size != int(expected_bytes):
            failures.append({"path": name, "reason": "hash_or_size_mismatch"})
    if failures:
        raise StageBlocker(f"MANIFEST_CONTENT_MISMATCH:{path}:{failures}")
    return {"path": str(path.resolve()), "sha256": observed_manifest_sha256, "file_count": len(manifest.get("files", {})), "files": manifest.get("files", {})}


def verify_design() -> dict[str, Any]:
    if not DESIGN_DIR.is_dir():
        raise StageBlocker("D16_DESIGN_BUNDLE_MISSING")
    design_manifest = verify_manifest_file(DESIGN_DIR / "SHA256_MANIFEST.json", EXPECTED_DESIGN_MANIFEST_SHA256)
    expected_files = {"FINAL_REPORT.md", "authoritative_inputs.json", "execution_contract.json", "experiment_design.json", "measurement_schema.json", "source_revision_identity.json"}
    if set(design_manifest["files"]) != expected_files:
        raise StageBlocker("D16_DESIGN_FILESET_MISMATCH")
    design = json.loads((DESIGN_DIR / "experiment_design.json").read_text(encoding="utf-8"))
    contract = json.loads((DESIGN_DIR / "execution_contract.json").read_text(encoding="utf-8"))
    authority = json.loads((DESIGN_DIR / "authoritative_inputs.json").read_text(encoding="utf-8"))
    schema = json.loads((DESIGN_DIR / "measurement_schema.json").read_text(encoding="utf-8"))
    source = json.loads((DESIGN_DIR / "source_revision_identity.json").read_text(encoding="utf-8"))
    if design.get("design_review_status") != "PASSED" or design.get("design_review_only") is not True or design.get("execution_authorized") is not False:
        raise StageBlocker("D16_DESIGN_STATUS_MISMATCH")
    if tuple(design.get("window", {}).get("updates", [])) != UPDATES or design["window"].get("update_393") != "FORBIDDEN":
        raise StageBlocker("D16_UPDATE_WINDOW_MISMATCH")
    if design.get("objective", {}).get("treatment_flag") != "drop_rollout_h32_from_training_objective=true":
        raise StageBlocker("D16_TREATMENT_FLAG_MISMATCH")
    if design.get("objective", {}).get("renormalization") != "NONE":
        raise StageBlocker("D16_RENORMALIZATION_CONTRACT_MISMATCH")
    if tuple(schema.get("per_update_required", {}).get("identity", []))[:3] != ("branch", "update", "schedule_index"):
        raise StageBlocker("D16_MEASUREMENT_SCHEMA_MISMATCH")
    if contract.get("status") != "SEALED_FOR_SEPARATELY_AUTHORIZED_EXECUTION_ONLY":
        raise StageBlocker("D16_EXECUTION_CONTRACT_STATUS_MISMATCH")
    if tuple(contract.get("branch_execution_order", {}).get("update_sequence", [])) != UPDATES or contract.get("branch_execution_order", {}).get("no_extra_update") is not True:
        raise StageBlocker("D16_BRANCH_EXECUTION_ORDER_MISMATCH")
    classification = contract.get("classification_precedence", [])
    labels = [item.get("label") for item in classification]
    expected_labels = [
        "EXECUTION_INVALID_BLOCKED",
        "ADAMW_FULL_PIPELINE_H32_CAUSALLY_PROTECTIVE_FOR_H1",
        "ADAMW_FULL_PIPELINE_H32_CAUSAL_AND_COMPATIBLE",
        "ADAMW_FULL_PIPELINE_H32_CAUSAL_BUT_H32_TRADEOFF",
        "ADAMW_FULL_PIPELINE_H32_NOT_CAUSAL_FOR_H1_BLOCKER",
    ]
    if labels != expected_labels:
        raise StageBlocker(f"D16_CLASSIFICATION_LABELS_MISMATCH:{labels}")
    if authority["certified_start"]["sha256"] != EXPECTED_BUNDLE_SHA256 or authority["certified_start"]["completed_optimizer_step"] != 384:
        raise StageBlocker("D16_CERTIFIED_START_AUTHORITY_MISMATCH")
    if authority["schedule"]["identity_file_sha256"] != EXPECTED_SCHEDULE_IDENTITY_SHA256 or authority["schedule"]["storage_sha256"] != EXPECTED_SCHEDULE_STORAGE_SHA256 or authority["schedule"]["logical_sha256"] != EXPECTED_SCHEDULE_LOGICAL_SHA256:
        raise StageBlocker("D16_SCHEDULE_AUTHORITY_MISMATCH")
    if authority["frozen_decision_contract"]["sha256"] != EXPECTED_CLASSIFICATION_CONTRACT_SHA256:
        raise StageBlocker("D16_CLASSIFICATION_CONTRACT_AUTHORITY_MISMATCH")
    if source.get("git_head") != EXPECTED_SOURCE_REVISION:
        raise StageBlocker("D16_SOURCE_REVISION_BUNDLE_MISMATCH")
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("D16_AUTHORIZATION_SOURCE_MISSING")
    auth_text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required_auth = (EXPERIMENT_ID, "PROJECT-OWNER EXECUTION AUTHORIZATION", "updates `385–392`", "Do not execute Frozen-20", "Do not execute R9")
    if any(marker not in auth_text for marker in required_auth):
        raise StageBlocker("D16_AUTHORIZATION_SCOPE_MISMATCH")
    if "Do not execute update 393 or later." not in auth_text and "Do not execute update 393+" not in auth_text:
        raise StageBlocker("D16_AUTHORIZATION_UPDATE_393_SCOPE_MISMATCH")
    for item in source.get("inspected_source_files", []):
        target = ROOT / item["logical_path"]
        if not target.is_file() or sha256_file(target) != item["sha256"]:
            raise StageBlocker(f"D16_SOURCE_FILE_HASH_MISMATCH:{item['logical_path']}")
    if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("D16_CERTIFIED_START_BUNDLE_HASH_MISMATCH")
    if sha256_file(SCHEDULE_IDENTITY) != EXPECTED_SCHEDULE_IDENTITY_SHA256 or sha256_file(SCHEDULE_STORAGE) != EXPECTED_SCHEDULE_STORAGE_SHA256:
        raise StageBlocker("D16_SCHEDULE_STORAGE_HASH_MISMATCH")
    if sha256_file(CLASSIFICATION_CONTRACT) != EXPECTED_CLASSIFICATION_CONTRACT_SHA256:
        raise StageBlocker("D16_CLASSIFICATION_CONTRACT_HASH_MISMATCH")
    predecessor = authority["immediate_predecessor"]
    d15_manifest_path = Path(predecessor["final_manifest_path"])
    d15_manifest = verify_manifest_file(d15_manifest_path, EXPECTED_D15_MANIFEST_SHA256)
    if d15_manifest["sha256"] != predecessor["final_manifest_sha256"]:
        raise StageBlocker("D15_FINAL_MANIFEST_AUTHORITY_MISMATCH")
    for label in ("final_report_path",):
        target = Path(predecessor[label])
        if not target.is_file() or sha256_file(target) != predecessor["final_report_sha256"]:
            raise StageBlocker(f"D15_PREDECESSOR_IDENTITY_MISMATCH:{label}")
    return {
        "design_manifest": design_manifest,
        "design": design,
        "execution_contract": contract,
        "authoritative_inputs": authority,
        "measurement_schema": schema,
        "source_revision_identity": source,
        "authorization": {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH)},
        "d15_manifest": d15_manifest,
        "classification_contract": {"path": str(CLASSIFICATION_CONTRACT.resolve()), "sha256": sha256_file(CLASSIFICATION_CONTRACT)},
    }


def verify_runtime_and_load(design: Mapping[str, Any]) -> dict[str, Any]:
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git_head != EXPECTED_SOURCE_REVISION:
        raise StageBlocker(f"SOURCE_REVISION_MISMATCH:{git_head}")
    base.set_deterministic(base.SEED)
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
    pre = base.preflight(True)
    authority = base.verify_authority(pre)
    environment = authority["current_environment"]
    expected_environment = {"python_version": "3.12.1", "pytorch_version": "2.13.0+cpu", "numpy_version": "2.1.3", "device": "cpu", "dtype": "float32", "source_revision": EXPECTED_SOURCE_REVISION}
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise StageBlocker(f"RUNTIME_IDENTITY_MISMATCH:{key}:{environment.get(key)}:{expected}")
    if not bool(torch.are_deterministic_algorithms_enabled()):
        raise StageBlocker("DETERMINISTIC_ALGORITHMS_NOT_ENABLED")
    model, optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
    loaded = d7.verify_loaded_identity(bundle, model, optimizer)
    if loaded["model_semantic_hash"] != EXPECTED_MODEL_HASH or loaded["optimizer_semantic_hash"] != EXPECTED_OPTIMIZER_HASH:
        raise StageBlocker("CERTIFIED_START_MODEL_OR_OPTIMIZER_HASH_MISMATCH")
    if loaded["exp_avg_semantic_hash"] != EXPECTED_EXP_AVG_HASH or loaded["exp_avg_sq_semantic_hash"] != EXPECTED_EXP_AVG_SQ_HASH:
        raise StageBlocker("CERTIFIED_START_MOMENT_HASH_MISMATCH")
    base.restore_rng(bundle["rng_state"])
    rng_start = base.rng_digest(base.capture_rng())
    if rng_start != EXPECTED_RNG_HASH:
        raise StageBlocker("CERTIFIED_START_RNG_HASH_MISMATCH")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(ALL_NAMES):
        raise StageBlocker("CERTIFIED_PARAMETER_ORDER_MISMATCH")
    initial_steps = optimizer_steps(clone_named_optimizer_state(optimizer))
    if set(initial_steps.values()) != {384}:
        raise StageBlocker(f"CERTIFIED_OPTIMIZER_STEP_MISMATCH:{initial_steps}")
    if base.optimizer_contract(optimizer) != {
        "class": "AdamW", "learning_rate": 5.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8,
        "weight_decay": 1.0e-5, "amsgrad": False, "foreach": None, "fused": None,
        "capturable": False, "differentiable": False, "maximize": False, "decoupled_weight_decay": True,
    }:
        raise StageBlocker("ADAMW_RUNTIME_CONTRACT_MISMATCH")
    return {
        "bundle": bundle,
        "pre": pre,
        "authority": authority,
        "environment": environment,
        "model": model,
        "optimizer": optimizer,
        "named": named,
        "loaded": loaded,
        "rng_start": rng_start,
        "git_head": git_head,
        "runtime_source_hash": sha256_file(Path(__file__).resolve()),
    }


def new_branch(bundle: Mapping[str, Any]) -> tuple[Any, torch.optim.Optimizer]:
    model, optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
    model.train()
    if [name for name, parameter in model.named_parameters() if parameter.requires_grad] != list(ALL_NAMES):
        raise StageBlocker("BRANCH_PARAMETER_ORDER_MISMATCH")
    return model, optimizer


def run_branch_update(
    branch: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
    zero_index: int,
    rng_before: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if zero_index + 1 not in UPDATES:
        raise StageBlocker(f"UNAUTHORIZED_UPDATE:{zero_index + 1}")
    base.restore_rng(rng_before)
    pre = state["pre"]
    authority = state["authority"]
    # Bind diagnostics to this branch's live parameters.  The authenticated
    # loader model in ``state`` is only a preflight identity object; it must
    # never be used as the gradient target for either branch.
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(ALL_NAMES):
        raise StageBlocker(f"BRANCH_LIVE_PARAMETER_ORDER_MISMATCH:{branch}:{zero_index + 1}")
    batch_record = batch_identity(authority["schedule_manifest"], zero_index)
    batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    before_model = clone_parameters(model)
    before_named_optimizer = clone_named_optimizer_state(optimizer)
    before_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    before_model_hash = canonical_model_state_sha256(model.state_dict())
    before_steps = optimizer_steps(before_named_optimizer)
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for _, parameter in named):
        raise StageBlocker(f"LIVE_GRADIENT_NOT_EMPTY:{branch}:{zero_index + 1}")
    rollout = base.causal_paired_rollout(
        model,
        batch["inputs"],
        batch["history_positions"],
        batch["history_times"],
        batch["target_times"],
        batch["starts"],
        batch["ends"],
        pre["stats"]["channels"],
        horizon=ACTIVE_HORIZON,
        target_positions_for_teacher=batch["target_positions"],
        hidden_consistency_enabled=True,
    )
    if rollout.get("free_branch_reads_target_positions") is not False or rollout.get("teacher_reference_stop_gradient") is not True:
        raise StageBlocker(f"GRAPH_SEMANTICS_GATE_FAILED:{branch}:{zero_index + 1}")
    predictions = rollout["predictions"]
    targets = batch["target_positions"][:, :ACTIVE_HORIZON, :]
    step_losses = F.smooth_l1_loss(predictions, targets, beta=POSITION_BETA, reduction="none").mean(dim=(0, 2))
    weights = rollout_horizon_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
    full_rollout_loss = torch.sum(step_losses * weights)
    partial_rollout_loss = torch.sum(step_losses[: ACTIVE_HORIZON - 1] * weights[: ACTIVE_HORIZON - 1])
    h32_term = weights[-1] * step_losses[-1]
    h1_loss = step_losses[0]
    hidden_loss = rollout["hidden_loss"]
    if branch == "CONTROL":
        rollout_loss = full_rollout_loss
        total_loss = rollout_loss + LAMBDA_H1 * h1_loss + LAMBDA_HIDDEN * hidden_loss
        deletion_identity = "NOT_APPLICABLE_CONTROL"
    elif branch == "INTERVENTION":
        rollout_loss = partial_rollout_loss
        total_loss = rollout_loss + LAMBDA_H1 * h1_loss + LAMBDA_HIDDEN * hidden_loss
        deletion_identity = "DELETED_weighted_rollout_h32_before_backward"
    else:
        raise StageBlocker(f"UNKNOWN_BRANCH:{branch}")
    losses = (full_rollout_loss, partial_rollout_loss, h32_term, h1_loss, hidden_loss, total_loss)
    if any(not bool(torch.isfinite(value)) for value in losses):
        raise StageBlocker(f"NONFINITE_OBJECTIVE:{branch}:{zero_index + 1}")
    h1_gradients, h1_flags = diagnostic_gradients(h1_loss, named)
    h32_gradients, h32_flags = diagnostic_gradients(h32_term, named)
    rng_after_diagnostics = base.capture_rng()
    if base.rng_digest(rng_after_diagnostics) != base.rng_digest(rng_before):
        raise StageBlocker(f"DIAGNOSTIC_RNG_CONTAMINATION:{branch}:{zero_index + 1}")
    total_loss.backward()
    if not base.gradients_are_finite(model):
        raise StageBlocker(f"NONFINITE_RAW_GRADIENT:{branch}:{zero_index + 1}")
    raw_gradients, none_flags = capture_gradients(named)
    raw_norm = norm(raw_gradients.values())
    returned_preclip = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    if not math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5):
        raise StageBlocker(f"CLIP_RETURNED_NORM_MISMATCH:{branch}:{zero_index + 1}:raw={raw_norm:.17g}:returned={returned_preclip:.17g}:difference={abs(raw_norm - returned_preclip):.17g}")
    postclip_gradients, _ = capture_gradients(named)
    postclip_norm = norm(postclip_gradients.values())
    exact_clip_coefficient = 1.0 if returned_preclip <= CLIP_MAX_NORM else min(1.0, CLIP_MAX_NORM / (returned_preclip + 1.0e-6))
    reconstructed = {name: raw_gradients[name] * exact_clip_coefficient for name in ALL_NAMES}
    reconstruction_residual = norm(postclip_gradients[name] - reconstructed[name] for name in ALL_NAMES)
    reconstruction_tolerance = ADDITIVITY_ABS + ADDITIVITY_REL * max(norm(postclip_gradients.values()), 1.0)
    if reconstruction_residual > reconstruction_tolerance:
        raise StageBlocker(f"POSTCLIP_RECONSTRUCTION_FAILED:{branch}:{zero_index + 1}")
    rng_before_optimizer = base.capture_rng()
    optimizer.step()
    after_model = clone_parameters(model)
    after_named_optimizer = clone_named_optimizer_state(optimizer)
    after_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    after_model_hash = canonical_model_state_sha256(model.state_dict())
    if not state_finite(model, optimizer):
        raise StageBlocker(f"NONFINITE_STATE_AFTER_UPDATE:{branch}:{zero_index + 1}")
    rng_after_training = base.capture_rng()
    if base.rng_digest(rng_before_optimizer) != base.rng_digest(rng_after_training):
        raise StageBlocker(f"OPTIMIZER_RNG_CONSUMPTION_UNMATCHED:{branch}:{zero_index + 1}")
    adamw_metrics, adamw_tensors = d7.adamw_decomposition(model, optimizer, before_model, postclip_gradients, after_model, state["table"] if "table" in state else d7.parameter_table(model)[0])
    validation, validation_meta = d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    rng_after_validation = base.capture_rng()
    if validation_meta["rng_unchanged_or_restored"] not in ("YES", "RESTORED"):
        raise StageBlocker(f"VALIDATION_RNG_GATE_FAILED:{branch}:{zero_index + 1}")
    after_steps = optimizer_steps(after_named_optimizer)
    parameter_delta = {name: after_model[name] - before_model[name] for name in ALL_NAMES}
    delta_norm = norm(parameter_delta.values())
    record = {
        "branch": branch,
        "update": zero_index + 1,
        "schedule_index": zero_index,
        "batch_identity": batch_record,
        "batch_window_sha256": batch_record["batch_window_sha256"],
        "model_state_hash_before": before_model_hash,
        "model_state_hash_after": after_model_hash,
        "optimizer_state_hash_before": before_optimizer_hash,
        "optimizer_state_hash_after": after_optimizer_hash,
        "rng_before_digest": base.rng_digest(rng_before),
        "rng_after_diagnostics_digest": base.rng_digest(rng_after_diagnostics),
        "rng_after_training_digest": base.rng_digest(rng_after_training),
        "rng_after_validation_digest": base.rng_digest(rng_after_validation),
        "rollout_loss_full": finite_float(full_rollout_loss),
        "rollout_loss_1_to_31": finite_float(partial_rollout_loss),
        "rollout_loss_used_for_backward": finite_float(rollout_loss),
        "weighted_rollout_h32_term": finite_float(h32_term),
        "explicit_h1_loss": finite_float(h1_loss),
        "hidden_loss_unweighted": finite_float(hidden_loss),
        "weighted_hidden_loss": finite_float(LAMBDA_HIDDEN * hidden_loss),
        "total_loss": finite_float(total_loss),
        "objective_deletion_identity": deletion_identity,
        "objective_renormalization": "NO",
        "raw_total_gradient_norm": raw_norm,
        "returned_pre_clipping_total_norm": returned_preclip,
        "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO",
        "exact_clip_coefficient": exact_clip_coefficient,
        "post_clipping_gradient_norm": postclip_norm,
        "postclip_reconstruction_residual_norm": reconstruction_residual,
        "raw_gradient_digest": d7.d6.named_tensor_digest(raw_gradients),
        "post_clipping_gradient_digest": d7.d6.named_tensor_digest(postclip_gradients),
        "gradient_none_pattern": none_flags,
        "gradient_none_pattern_hash": sha256_bytes(canonical_json(none_flags)),
        "h1_gradient_norm": norm(h1_gradients.values()),
        "h32_gradient_norm": norm(h32_gradients.values()),
        "weighted_h32_gradient_norm": norm(h32_gradients[name] * float(weights[-1]) for name in ALL_NAMES),
        "h1_gradient_norm_by_block": block_norms(h1_gradients),
        "h32_gradient_norm_by_block": block_norms(h32_gradients),
        "raw_gradient_norm_by_block": block_norms(raw_gradients),
        "postclip_gradient_norm_by_block": block_norms(postclip_gradients),
        "parameter_update_norm": delta_norm,
        "parameter_update_norm_by_block": block_norms(parameter_delta),
        "parameter_update_digest": d7.d6.named_tensor_digest(parameter_delta),
        "loss_driven_adamw_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["adaptive_update_norm"],
        "decoupled_weight_decay_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["decay_update_norm"],
        "adamw_decomposition_residual_norm": adamw_metrics["aggregate"]["whole_model"]["decomposition_residual_norm"],
        "adamw_decomposition_status": adamw_metrics["aggregate"]["whole_model"]["decomposition_pass_fail"],
        "step_counter_before": before_steps,
        "step_counter_after": after_steps,
        "exp_avg_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_norm"],
        "exp_avg_sq_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_sq_norm"],
        "adamw": adamw_metrics,
        "validation": validation,
        "per_update_H1": validation["1"],
        "per_update_H32": validation["32"],
        "finite_status": "FINITE",
        "branch_rng_contract_status": "PENDING_MATCH_CHECK",
    }
    capture = {
        "branch": branch,
        "update": zero_index + 1,
        "model_before": before_model,
        "model_after": after_model,
        "optimizer_before": before_named_optimizer,
        "optimizer_after": after_named_optimizer,
        "raw_gradients": raw_gradients,
        "postclip_gradients": postclip_gradients,
        "parameter_delta": parameter_delta,
        "h1_gradients_diagnostic": h1_gradients,
        "h32_gradients_diagnostic": h32_gradients,
        "adamw_decomposition_tensors": adamw_tensors,
    }
    rng_after = copy.deepcopy(rng_after_validation)
    return record, capture, rng_after


def add_longitudinal_divergence(control: Mapping[str, Any], intervention: Mapping[str, Any], control_capture: Mapping[str, Any], intervention_capture: Mapping[str, Any]) -> None:
    c_before = control_capture["model_before"]
    i_before = intervention_capture["model_before"]
    c_after = control_capture["model_after"]
    i_after = intervention_capture["model_after"]
    c_opt_before = control_capture["optimizer_before"]
    i_opt_before = intervention_capture["optimizer_before"]
    c_opt_after = control_capture["optimizer_after"]
    i_opt_after = intervention_capture["optimizer_after"]
    pre_param = difference_norm(i_before, c_before)
    post_param = difference_norm(i_after, c_after)
    pre_m = difference_norm(moment_map(i_opt_before, "exp_avg"), moment_map(c_opt_before, "exp_avg"))
    post_m = difference_norm(moment_map(i_opt_after, "exp_avg"), moment_map(c_opt_after, "exp_avg"))
    pre_v = difference_norm(moment_map(i_opt_before, "exp_avg_sq"), moment_map(c_opt_before, "exp_avg_sq"))
    post_v = difference_norm(moment_map(i_opt_after, "exp_avg_sq"), moment_map(c_opt_after, "exp_avg_sq"))
    c_delta = control_capture["parameter_delta"]
    i_delta = intervention_capture["parameter_delta"]
    update_increment = norm(i_delta[name] - c_delta[name] for name in ALL_NAMES)
    values = {
        "pre_update_parameter_distance_norm": pre_param,
        "branch_vs_control_parameter_distance_norm": post_param,
        "pre_update_exp_avg_difference_norm": pre_m,
        "branch_vs_control_exp_avg_difference_norm": post_m,
        "pre_update_exp_avg_sq_difference_norm": pre_v,
        "branch_vs_control_exp_avg_sq_difference_norm": post_v,
        "update_increment_difference_norm": update_increment,
        "parameter_distance_increment": post_param - pre_param,
        "parameter_distance_by_block": {
            "GRU": norm((i_after[name] - c_after[name]) for name in GRU_NAMES),
            "HEAD": norm((i_after[name] - c_after[name]) for name in HEAD_NAMES),
            "FULL_TRAINABLE": post_param,
        },
        "exp_avg_difference_by_block": {
            "GRU": norm(moment_map(i_opt_after, "exp_avg")[name] - moment_map(c_opt_after, "exp_avg")[name] for name in GRU_NAMES),
            "HEAD": norm(moment_map(i_opt_after, "exp_avg")[name] - moment_map(c_opt_after, "exp_avg")[name] for name in HEAD_NAMES),
            "FULL_TRAINABLE": post_m,
        },
        "exp_avg_sq_difference_by_block": {
            "GRU": norm(moment_map(i_opt_after, "exp_avg_sq")[name] - moment_map(c_opt_after, "exp_avg_sq")[name] for name in GRU_NAMES),
            "HEAD": norm(moment_map(i_opt_after, "exp_avg_sq")[name] - moment_map(c_opt_after, "exp_avg_sq")[name] for name in HEAD_NAMES),
            "FULL_TRAINABLE": post_v,
        },
    }
    for record in (control, intervention):
        if record["branch"] == "CONTROL":
            record.update({
                "pre_update_parameter_distance_norm": 0.0,
                "branch_vs_control_parameter_distance_norm": 0.0,
                "pre_update_exp_avg_difference_norm": 0.0,
                "branch_vs_control_exp_avg_difference_norm": 0.0,
                "pre_update_exp_avg_sq_difference_norm": 0.0,
                "branch_vs_control_exp_avg_sq_difference_norm": 0.0,
                "update_increment_difference_norm": 0.0,
                "parameter_distance_increment": 0.0,
                "parameter_distance_by_block": {"GRU": 0.0, "HEAD": 0.0, "FULL_TRAINABLE": 0.0},
                "exp_avg_difference_by_block": {"GRU": 0.0, "HEAD": 0.0, "FULL_TRAINABLE": 0.0},
                "exp_avg_sq_difference_by_block": {"GRU": 0.0, "HEAD": 0.0, "FULL_TRAINABLE": 0.0},
            })
        else:
            record.update(values)


def compare_control_to_d15(control_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    d15_path = D15_DIR / "per_update_measurements.json"
    if not d15_path.is_file():
        raise StageBlocker("D15_PER_UPDATE_REFERENCE_MISSING")
    expected = json.loads(d15_path.read_text(encoding="utf-8"))
    exact_fields = ("update", "zero_based_schedule_index", "batch_window_sha256", "canonical_state_hash_before", "canonical_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest")
    scalar_fields = ("total_loss", "rollout_loss", "hidden_loss_unweighted", "h1_loss", "raw_total_gradient_norm", "returned_pre_clipping_total_norm", "clipped_total_gradient_norm", "parameter_update_norm", "exp_avg_norm", "exp_avg_sq_norm", "eval_h1", "eval_h32")
    comparisons: list[dict[str, Any]] = []
    for row in control_rows:
        update = str(row["update"])
        reference = expected.get(update, {}).get("row")
        if reference is None:
            raise StageBlocker(f"D15_CONTROL_REFERENCE_UPDATE_MISSING:{update}")
        failures: list[str] = []
        for field in exact_fields:
            if field == "canonical_state_hash_before": observed = row["model_state_hash_before"]
            elif field == "canonical_state_hash_after": observed = row["model_state_hash_after"]
            elif field == "optimizer_state_hash_before": observed = row["optimizer_state_hash_before"]
            elif field == "optimizer_state_hash_after": observed = row["optimizer_state_hash_after"]
            elif field == "clipped_gradient_digest": observed = row["post_clipping_gradient_digest"]
            elif field == "zero_based_schedule_index": observed = row["schedule_index"]
            else: observed = row[field]
            if observed != reference[field]:
                failures.append(field)
        for field in scalar_fields:
            if field == "total_loss": observed = row["total_loss"]
            elif field == "rollout_loss": observed = row["rollout_loss_full"]
            elif field == "hidden_loss_unweighted": observed = row["hidden_loss_unweighted"]
            elif field == "h1_loss": observed = row["explicit_h1_loss"]
            elif field == "raw_total_gradient_norm": observed = row["raw_total_gradient_norm"]
            elif field == "returned_pre_clipping_total_norm": observed = row["returned_pre_clipping_total_norm"]
            elif field == "clipped_total_gradient_norm": observed = row["post_clipping_gradient_norm"]
            elif field == "parameter_update_norm": observed = row["parameter_update_norm"]
            elif field == "exp_avg_norm": observed = row["exp_avg_norm"]
            elif field == "exp_avg_sq_norm": observed = row["exp_avg_sq_norm"]
            elif field == "eval_h1": observed = row["per_update_H1"]
            else: observed = row["per_update_H32"]
            if not math.isclose(float(observed), float(reference[field]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
                failures.append(field)
        comparisons.append({"update": int(update), "status": "PASS" if not failures else "FAIL", "failed_fields": failures})
    failed = [item for item in comparisons if item["status"] != "PASS"]
    return {"status": "PASS" if not failed else "FAIL", "reference_bundle": str(D15_DIR.resolve()), "reference_measurements": str(d15_path.resolve()), "comparisons": comparisons, "failed_updates": failed}


def classify_causal_result(control_h1: float, intervention_h1: float, intervention_h32: float, integrity_pass: bool) -> tuple[str, bool, bool, float]:
    if not integrity_pass:
        return "EXECUTION_INVALID_BLOCKED", False, intervention_h32 <= H32_PASS_THRESHOLD, control_h1 - intervention_h1
    effect = control_h1 - intervention_h1
    material = effect >= H1_MATERIAL_EFFECT_THRESHOLD
    preserved = intervention_h32 <= H32_PASS_THRESHOLD
    if effect <= -H1_MATERIAL_EFFECT_THRESHOLD:
        label = "ADAMW_FULL_PIPELINE_H32_CAUSALLY_PROTECTIVE_FOR_H1"
    elif material and preserved:
        label = "ADAMW_FULL_PIPELINE_H32_CAUSAL_AND_COMPATIBLE"
    elif material and not preserved:
        label = "ADAMW_FULL_PIPELINE_H32_CAUSAL_BUT_H32_TRADEOFF"
    else:
        label = "ADAMW_FULL_PIPELINE_H32_NOT_CAUSAL_FOR_H1_BLOCKER"
    return label, material, preserved, effect


def write_manifest(output: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    total_bytes = 0
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256_MANIFEST.json"):
        rel = str(path.relative_to(output)).replace("\\", "/")
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        total_bytes += path.stat().st_size
    manifest = {"schema_version": "stage3_h13_post_d15_d16_execution_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "file_count": len(files), "total_bytes": total_bytes, "files": files}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return manifest


def verify_output_manifest(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    total_bytes = 0
    for name, entry in manifest["files"].items():
        path = output / name
        if not path.is_file():
            failures.append({"path": name, "reason": "missing"})
            continue
        total_bytes += path.stat().st_size
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            failures.append({"path": name, "reason": "hash_or_size_mismatch"})
    return {"status": "VERIFIED" if not failures else "BLOCKED", "checked": len(manifest["files"]), "total_bytes": total_bytes, "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json"), "failures": failures}


def render_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| update | branch | raw/preclip norm | clip coeff | postclip norm | AdamW displacement | parameter distance | exp_avg difference | exp_avg_sq difference |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {update} | {branch} | {raw:.17g} | {coef:.17g} | {post:.17g} | {disp:.17g} | {param:.17g} | {m:.17g} | {v:.17g} |".format(update=row["update"], branch=row["branch"], raw=row["raw_total_gradient_norm"], coef=row["exact_clip_coefficient"], post=row["post_clipping_gradient_norm"], disp=row["parameter_update_norm"], param=row["branch_vs_control_parameter_distance_norm"], m=row["branch_vs_control_exp_avg_difference_norm"], v=row["branch_vs_control_exp_avg_sq_difference_norm"]))
    return "\n".join(lines)


def render_report(result: Mapping[str, Any], output: Path, manifest_check: Mapping[str, Any]) -> str:
    control = result["endpoint"]["control"]
    intervention = result["endpoint"]["intervention"]
    classification = result["classification"]
    control_anchor = result["control_reproduction"]
    rows = result["per_update_measurements"]
    c385 = next(item for item in rows if item["branch"] == "CONTROL" and item["update"] == 385)
    i385 = next(item for item in rows if item["branch"] == "INTERVENTION" and item["update"] == 385)
    i_rows = [item for item in rows if item["branch"] == "INTERVENTION"]
    first = i_rows[0]
    last = i_rows[-1]
    if last["branch_vs_control_parameter_distance_norm"] > first["branch_vs_control_parameter_distance_norm"] * 1.000001:
        accumulation_pattern = "accumulated"
    elif last["branch_vs_control_parameter_distance_norm"] < first["branch_vs_control_parameter_distance_norm"] * 0.999999:
        accumulation_pattern = "partially reconverged/non-monotonic"
    else:
        accumulation_pattern = "approximately stable"
    if first["branch_vs_control_parameter_distance_norm"] == 0.0:
        immediate_statement = "No post-update divergence was recorded, so immediate-vs-accumulated separation is not applicable."
    else:
        immediate_statement = f"At update 385 the post-update parameter distance was {first['branch_vs_control_parameter_distance_norm']:.17g} and the update-increment difference was {first['update_increment_difference_norm']:.17g}; subsequent moment and parameter distances were tracked independently."
    q = result["q_answers"]
    lines = [
        f"{EXPERIMENT_ID}:",
        result["status"],
        "",
        "FIRST_BLOCKER:", result["first_blocker"],
        "",
        "CAUSAL_CLASSIFICATION:", classification["label"],
        "",
        "ONE_SENTENCE_VERDICT:", result["one_sentence_verdict"],
        "",
        "## 1. PROTOCOL_INTEGRITY", "",
        "PROJECT_OWNER_D16_EXECUTION_AUTHORIZATION: YES",
        "D16_DESIGN_REVIEW_PASSED: YES",
        "D16_DESIGN_MANIFEST_VERIFIED: YES",
        "CERTIFIED_UPDATE_384_IDENTITY: MATCH",
        "SOURCE_REVISION_MATCH: YES",
        "RUNTIME_IDENTITY_MATCH: YES",
        "CANONICAL_BATCH_SCHEDULE_MATCH: YES",
        "RNG_MATCHING_CONTRACT: PASS",
        "CONTROL_BRANCH_EXECUTED: YES",
        "INTERVENTION_BRANCH_EXECUTED: YES",
        "UPDATES_385_392_EXECUTED: YES",
        "UPDATE_393_EXECUTED: NO",
        "FROZEN20_OPENED: NO",
        "FROZEN20_USED: NO",
        "R9_EXECUTED: NO",
        "R8E_A1_MODIFIED: NO",
        "HYPERPARAMETER_TUNING_EXECUTED: NO",
        "HISTORICAL_ARTIFACTS_MODIFIED: NO",
        "",
        "## 2. AUTHORITATIVE_INPUT_IDENTITY", "",
        f"D16_DESIGN_BUNDLE: {DESIGN_DIR}",
        f"D16_DESIGN_MANIFEST: {DESIGN_DIR / 'SHA256_MANIFEST.json'}",
        f"D16_DESIGN_MANIFEST_SHA256: {result['design_manifest_sha256']}",
        f"CERTIFIED_POST_UPDATE_384_STATE: {BUNDLE_PATH}",
        f"CERTIFIED_POST_UPDATE_384_STATE_SHA256: {EXPECTED_BUNDLE_SHA256}",
        f"SOURCE_REVISION: {EXPECTED_SOURCE_REVISION}",
        f"OPTIMIZER_STATE_IDENTITY: {EXPECTED_OPTIMIZER_HASH}",
        f"SCHEDULE_IDENTITY: {SCHEDULE_IDENTITY} (sha256={EXPECTED_SCHEDULE_IDENTITY_SHA256})",
        f"SCHEDULE_STORAGE: {SCHEDULE_STORAGE} (sha256={EXPECTED_SCHEDULE_STORAGE_SHA256})",
        f"RNG_IDENTITY: {EXPECTED_RNG_HASH}",
        f"AUTHORIZATION_SOURCE: {AUTHORIZATION_PATH}",
        "",
        "## 3. EXACT_INTERVENTION_EXECUTED", "",
        "DELETED_TERM: w_32 * SmoothL1(free_prediction_32, target_32; beta=0.001)",
        "RENORMALIZATION: NO",
        "COMPENSATION: NO",
        "REPLACEMENT_GRADIENT: NO",
        "GRADIENT_PROJECTION: NO",
        "OPTIMIZER_CHANGE: NO",
        "CLIPPING_CHANGE: NO",
        "H32_FORWARD_PATH_RETAINED: YES",
        "H32_ENDPOINT_EVALUATION_RETAINED: YES",
        "",
        "## 4. CONTROL_REPRODUCTION", "",
        f"EXPECTED_CONTROL_H1: {control['expected_h1']:.17g}", f"OBSERVED_CONTROL_H1: {control['observed_h1']:.17g}", f"CONTROL_H1_DEVIATION: {control['h1_deviation']:.17g}",
        f"EXPECTED_CONTROL_H32: {control['expected_h32']:.17g}", f"OBSERVED_CONTROL_H32: {control['observed_h32']:.17g}", f"CONTROL_H32_DEVIATION: {control['h32_deviation']:.17g}",
        f"CONTROL_TRAJECTORY_INTEGRITY: {control_anchor['status']}", f"CONTROL_ENDPOINT_TOLERANCE: rtol={REPLAY_RTOL}, atol={REPLAY_ATOL}",
        "",
        "## 5. PRIMARY_CAUSAL_ESTIMAND", "",
        f"CONTROL_H1: {control['observed_h1']:.17g}", f"INTERVENTION_H1: {intervention['observed_h1']:.17g}", f"E_H1: {result['estimand']['E_H1']:.17g}", f"H1_MATERIAL_EFFECT_THRESHOLD: {H1_MATERIAL_EFFECT_THRESHOLD:.17g}", f"H1_MATERIAL_EFFECT: {str(result['estimand']['H1_MATERIAL_EFFECT']).upper()}",
        "",
        "## 6. H32_PRESERVATION", "",
        f"CONTROL_H32: {control['observed_h32']:.17g}", f"INTERVENTION_H32: {intervention['observed_h32']:.17g}", f"H32_PASS_THRESHOLD: {H32_PASS_THRESHOLD:.17g}", f"H32_PRESERVED: {str(result['estimand']['H32_PRESERVED']).upper()}",
        "",
        "## 7. PREDECLARED_CAUSAL_CLASSIFICATION", "",
        "1. Invalid integrity/fidelity gate -> EXECUTION_INVALID_BLOCKED.",
        "2. Valid and E_H1 <= -4.099006815405639e-6 -> ADAMW_FULL_PIPELINE_H32_CAUSALLY_PROTECTIVE_FOR_H1.",
        "3. Valid and E_H1 >= 4.099006815405639e-6 and INTERVENTION_H32 <= 0.01856902565856056 -> ADAMW_FULL_PIPELINE_H32_CAUSAL_AND_COMPATIBLE.",
        "4. Valid and E_H1 >= 4.099006815405639e-6 and INTERVENTION_H32 > 0.01856902565856056 -> ADAMW_FULL_PIPELINE_H32_CAUSAL_BUT_H32_TRADEOFF.",
        "5. Otherwise, within the material-effect interval -> ADAMW_FULL_PIPELINE_H32_NOT_CAUSAL_FOR_H1_BLOCKER.",
        f"CONDITION_FIRED: {classification['condition_fired']}", f"FORMAL_LABEL: {classification['label']}",
        "",
        "## 8. PER_UPDATE_PIPELINE_TABLE", "",
        render_table(rows),
        "",
        f"Complete measurements: {output / 'per_update_measurements.json'}",
        "",
        "## 9. UPDATE_385_IMMEDIATE_EFFECT", "",
        immediate_statement,
        f"CONTROL clipping coefficient={c385['exact_clip_coefficient']:.17g}; INTERVENTION clipping coefficient={i385['exact_clip_coefficient']:.17g}; coefficient difference={abs(c385['exact_clip_coefficient'] - i385['exact_clip_coefficient']):.17g}.",
        f"CONTROL AdamW displacement={c385['parameter_update_norm']:.17g}; INTERVENTION AdamW displacement={i385['parameter_update_norm']:.17g}.",
        f"First-block parameter-distance descriptors: GRU={i385['parameter_distance_by_block']['GRU']:.17g}, HEAD={i385['parameter_distance_by_block']['HEAD']:.17g}.",
        "These are descriptive diagnostics, not secondary causal estimands.",
        "",
        "## 10. UPDATES_386_392_ACCUMULATION", "",
        f"Across updates 386-392, parameter distance was {accumulation_pattern}; update-392 parameter distance={last['branch_vs_control_parameter_distance_norm']:.17g}, exp_avg distance={last['branch_vs_control_exp_avg_difference_norm']:.17g}, exp_avg_sq distance={last['branch_vs_control_exp_avg_sq_difference_norm']:.17g}, and update-increment difference={last['update_increment_difference_norm']:.17g}.",
        "The complete quantitative sequence is retained in the machine-readable per-update artifact.",
        "",
        "## 11. IMMEDIATE_VS_ACCUMULATED_EFFECT", "",
        f"{result['immediate_vs_accumulated']}",
        "",
        "## 12. D15_TO_D16_INTERPRETATION", "",
        "D15 was observational/mechanistic evidence about realized update-space geometry. D16 is the causal replay: it deletes the weighted H32 objective source before backward and lets branch-local clipping, AdamW updates, parameters, moments, and later forward passes evolve for all eight updates.",
        f"Under this exact recipe, the D16 endpoint classification is {classification['label']}; this does not generalize beyond the sealed objective, model, optimizer, schedule, and updates 385-392.",
        "",
        "## Q1-Q12 REQUIRED ANSWERS", "",
    ]
    for key in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12"):
        lines.extend([f"{key}: {q[key]}", ""])
    lines.extend([
        "## FINAL_ARTIFACT_REPORTING", "",
        f"OUTPUT_BUNDLE_PATH: {output}", f"FINAL_REPORT_PATH: {output / 'FINAL_REPORT.md'}", f"RESULT_JSON_PATH: {output / 'result.json'}", f"SHA256_MANIFEST_PATH: {output / 'SHA256_MANIFEST.json'}", f"PER_UPDATE_MEASUREMENTS_PATH: {output / 'per_update_measurements.json'}", f"ENDPOINT_EVIDENCE_PATH: {output / 'endpoint_evidence.json'}", f"CAUSAL_CLASSIFICATION_PATH: {output / 'causal_classification.json'}", f"SHA256_VERIFICATION: {manifest_check['status']}",
        "",
        "STOP CONDITION: D16 only was executed; no D17, D16A, D16B, Frozen-20, R9, update 393+, or other experiment was initiated.",
    ])
    return "\n".join(lines) + "\n"


def fail_closed(output: Path, blocker: str, design: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol_integrity.json", {"status": "BLOCKED", "first_blocker": blocker, "design_bundle_verified": "YES" if design else "NO", "updates_executed": [], "update_393_executed": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "hyperparameter_tuning_executed": "NO", "historical_artifacts_modified": "NO"})
    write_json(output / "terminal_certificate.json", {"experiment_identifier": EXPERIMENT_ID, "status": "BLOCKED", "first_blocker": blocker, "causal_classification": "EXECUTION_INVALID_BLOCKED", "update_393_executed": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "historical_artifacts_modified": "NO"})
    (output / "FINAL_REPORT.md").write_text(f"{EXPERIMENT_ID}:\nBLOCKED\n\nFIRST_BLOCKER:\n{blocker}\n\nCAUSAL_CLASSIFICATION:\nEXECUTION_INVALID_BLOCKED\n", encoding="utf-8", newline="\n")
    manifest = write_manifest(output)
    check = verify_output_manifest(output, manifest)
    write_json(output / "manifest_verification.json", check)
    return 2


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"OUTPUT_DIRECTORY_ALREADY_EXISTS_NO_RESUME:{output}")
    output.mkdir(parents=True, exist_ok=False)
    design = None
    try:
        design = verify_design()
        runtime = verify_runtime_and_load(design)
        bundle = runtime["bundle"]
        state = {**runtime, "table": d7.parameter_table(runtime["model"])[0]}
        control_model, control_optimizer = new_branch(bundle)
        intervention_model, intervention_optimizer = new_branch(bundle)
        control_start_hash = canonical_model_state_sha256(control_model.state_dict())
        intervention_start_hash = canonical_model_state_sha256(intervention_model.state_dict())
        control_start_optimizer_hash = base.optimizer_semantic_hash(control_optimizer)
        intervention_start_optimizer_hash = base.optimizer_semantic_hash(intervention_optimizer)
        if control_start_hash != EXPECTED_MODEL_HASH or intervention_start_hash != EXPECTED_MODEL_HASH or control_start_optimizer_hash != EXPECTED_OPTIMIZER_HASH or intervention_start_optimizer_hash != EXPECTED_OPTIMIZER_HASH:
            raise StageBlocker("BRANCH_START_STATE_IDENTITY_MISMATCH")
        start_rng = copy.deepcopy(bundle["rng_state"])
        control_rng = copy.deepcopy(start_rng)
        intervention_rng = copy.deepcopy(start_rng)
        write_json(output / "design_bundle_identity.json", {"design_bundle": str(DESIGN_DIR.resolve()), "design_manifest_sha256": design["design_manifest"]["sha256"], "design_manifest_verified": "YES", "design_files": design["design_manifest"]["files"], "d15_final_manifest": design["d15_manifest"], "classification_contract": design["classification_contract"]})
        write_json(output / "authoritative_input_identity.json", {
            "design_bundle": str(DESIGN_DIR.resolve()), "design_manifest": str((DESIGN_DIR / "SHA256_MANIFEST.json").resolve()), "design_manifest_sha256": design["design_manifest"]["sha256"],
            "certified_start": {"path": str(BUNDLE_PATH.resolve()), "sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_step": 384, "model_semantic_hash": EXPECTED_MODEL_HASH, "optimizer_semantic_hash": EXPECTED_OPTIMIZER_HASH, "exp_avg_semantic_hash": EXPECTED_EXP_AVG_HASH, "exp_avg_sq_semantic_hash": EXPECTED_EXP_AVG_SQ_HASH, "rng_sha256": EXPECTED_RNG_HASH},
            "source_revision": design["source_revision_identity"], "schedule": {"identity_path": str(SCHEDULE_IDENTITY.resolve()), "identity_sha256": EXPECTED_SCHEDULE_IDENTITY_SHA256, "storage_path": str(SCHEDULE_STORAGE.resolve()), "storage_sha256": EXPECTED_SCHEDULE_STORAGE_SHA256, "logical_sha256": EXPECTED_SCHEDULE_LOGICAL_SHA256, "slice": "indices 384..391 => updates 385..392"},
            "classification_contract": {"path": str(CLASSIFICATION_CONTRACT.resolve()), "sha256": EXPECTED_CLASSIFICATION_CONTRACT_SHA256}, "authorization": design["authorization"], "d15_predecessor": design["d15_manifest"],
        })
        write_json(output / "runtime_source_identity.json", {"runner_path": str(Path(__file__).resolve()), "runner_sha256": runtime["runtime_source_hash"], "git_head": runtime["git_head"], "authenticated_source_files": design["source_revision_identity"]["inspected_source_files"]})
        write_json(output / "execution_configuration.json", {"schema_version": "stage3_h13_post_d15_d16_execution_configuration_v1", "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(), "updates": list(UPDATES), "start_update": START_UPDATE, "branches": ["CONTROL", "INTERVENTION"], "treatment": "delete w_32 * SmoothL1(free_prediction_32,target_32; beta=0.001) from INTERVENTION scalar objective before backward", "renormalization": "NONE", "lambda_h1": LAMBDA_H1, "lambda_hidden": LAMBDA_HIDDEN, "position_beta": POSITION_BETA, "active_horizon": ACTIVE_HORIZON, "clip": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False, "foreach": None}, "optimizer": base.optimizer_contract(control_optimizer), "runtime": runtime["environment"], "prohibited": {"update_393": "NO", "Frozen20": "NO", "R9": "NO", "R8E_A1_modification": "NO", "hyperparameter_tuning": "NO"}})
        write_json(output / "repair_log.json", {"schema_version": "stage3_h13_post_d15_d16_repair_log_v1", "repairs": [], "statement": "No engineering repair was required."})
        write_json(output / "protocol_integrity.json", {"status": "RUNNING", "project_owner_authorization": "YES", "design_bundle_verified": "YES", "certified_start_match": "YES", "source_revision_match": "YES", "runtime_identity_match": "YES", "canonical_schedule_match": "YES", "rng_matching_contract": "PENDING", "control_branch_executed": "RUNNING", "intervention_branch_executed": "RUNNING", "updates": list(UPDATES), "update_393_executed": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifacts_modified": "NO"})
        schedule_rows = [batch_identity(runtime["authority"]["schedule_manifest"], index) for index in range(384, 392)]
        write_json(output / "canonical_batch_schedule.json", {"identity_sha256": EXPECTED_SCHEDULE_IDENTITY_SHA256, "storage_sha256": EXPECTED_SCHEDULE_STORAGE_SHA256, "logical_sha256": EXPECTED_SCHEDULE_LOGICAL_SHA256, "required_slice": schedule_rows})
        all_records: list[dict[str, Any]] = []
        captures_root = output / "branch_state_and_optimizer_captures"
        logs = [f"{utc_now()} D16_PREFLIGHT_PASS"]
        for zero_index in range(384, 392):
            update = zero_index + 1
            logs.append(f"{utc_now()} BEGIN update={update}")
            control_record, control_capture, control_rng = run_branch_update("CONTROL", control_model, control_optimizer, state, zero_index, control_rng, output)
            intervention_record, intervention_capture, intervention_rng = run_branch_update("INTERVENTION", intervention_model, intervention_optimizer, state, zero_index, intervention_rng, output)
            if control_record["rng_before_digest"] != intervention_record["rng_before_digest"] or control_record["rng_after_training_digest"] != intervention_record["rng_after_training_digest"] or control_record["rng_after_validation_digest"] != intervention_record["rng_after_validation_digest"]:
                raise StageBlocker(f"RNG_MATCHING_CONTRACT_FAILED:{update}")
            control_record["branch_rng_contract_status"] = "PASS"
            intervention_record["branch_rng_contract_status"] = "PASS"
            add_longitudinal_divergence(control_record, intervention_record, control_capture, intervention_capture)
            all_records.extend((control_record, intervention_record))
            write_tensor(captures_root / f"update_{update:05d}" / "CONTROL.pt", control_capture)
            write_tensor(captures_root / f"update_{update:05d}" / "INTERVENTION.pt", intervention_capture)
            write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": list(range(385, update + 1)), "control_optimizer_step": update, "intervention_optimizer_step": update, "update_393_executed": "NO", "rng_matching": "PASS"})
            logs.append(f"{utc_now()} END update={update} control={control_record['model_state_hash_after']} intervention={intervention_record['model_state_hash_after']}")
        all_records.sort(key=lambda item: (int(item["update"]), item["branch"]))
        control_rows = [item for item in all_records if item["branch"] == "CONTROL"]
        intervention_rows = [item for item in all_records if item["branch"] == "INTERVENTION"]
        control_reproduction = compare_control_to_d15(control_rows)
        endpoint_control = control_rows[-1]
        endpoint_intervention = intervention_rows[-1]
        endpoint = {
            "update": 392,
            "control": {"H1": endpoint_control["per_update_H1"], "H32": endpoint_control["per_update_H32"], "observed_h1": endpoint_control["per_update_H1"], "observed_h32": endpoint_control["per_update_H32"], "expected_h1": 0.00009096969417553673, "expected_h32": 0.018159905521264792, "h1_deviation": endpoint_control["per_update_H1"] - 0.00009096969417553673, "h32_deviation": endpoint_control["per_update_H32"] - 0.018159905521264792, "h1_pass": math.isclose(endpoint_control["per_update_H1"], 0.00009096969417553673, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL), "h32_pass": math.isclose(endpoint_control["per_update_H32"], 0.018159905521264792, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)},
            "intervention": {"H1": endpoint_intervention["per_update_H1"], "H32": endpoint_intervention["per_update_H32"], "observed_h1": endpoint_intervention["per_update_H1"], "observed_h32": endpoint_intervention["per_update_H32"], "h1_guardrail_diagnostic": endpoint_intervention["per_update_H1"] <= H1_GUARDRAIL, "h32_preservation": endpoint_intervention["per_update_H32"] <= H32_PASS_THRESHOLD},
        }
        integrity_pass = control_reproduction["status"] == "PASS" and endpoint["control"]["h1_pass"] and endpoint["control"]["h32_pass"] and all(item["finite_status"] == "FINITE" and item["adamw_decomposition_status"] == "PASS" for item in all_records)
        classification_label, material, preserved, effect = classify_causal_result(endpoint["control"]["observed_h1"], endpoint["intervention"]["observed_h1"], endpoint["intervention"]["observed_h32"], integrity_pass)
        if not integrity_pass:
            raise StageBlocker("CONTROL_OR_FIDELITY_GATE_FAILED")
        if effect <= -H1_MATERIAL_EFFECT_THRESHOLD:
            condition_fired = "E_H1 <= -H1_MATERIAL_EFFECT_THRESHOLD"
        elif material and preserved:
            condition_fired = "E_H1 >= H1_MATERIAL_EFFECT_THRESHOLD AND INTERVENTION_H32 <= H32_PASS_THRESHOLD"
        elif material and not preserved:
            condition_fired = "E_H1 >= H1_MATERIAL_EFFECT_THRESHOLD AND INTERVENTION_H32 > H32_PASS_THRESHOLD"
        else:
            condition_fired = "-H1_MATERIAL_EFFECT_THRESHOLD < E_H1 < H1_MATERIAL_EFFECT_THRESHOLD"
        classification = {"label": classification_label, "condition_fired": condition_fired, "integrity_gate": "PASS", "H1_MATERIAL_EFFECT": material, "H32_PRESERVED": preserved}
        estimand = {"CONTROL_H1": endpoint["control"]["observed_h1"], "INTERVENTION_H1": endpoint["intervention"]["observed_h1"], "E_H1": effect, "H1_MATERIAL_EFFECT_THRESHOLD": H1_MATERIAL_EFFECT_THRESHOLD, "H1_MATERIAL_EFFECT": material, "CONTROL_H32": endpoint["control"]["observed_h32"], "INTERVENTION_H32": endpoint["intervention"]["observed_h32"], "H32_PASS_THRESHOLD": H32_PASS_THRESHOLD, "H32_PRESERVED": preserved, "H32_CHANGE_DIAGNOSTIC": endpoint["intervention"]["observed_h32"] - endpoint["control"]["observed_h32"]}
        immediate = next(item for item in intervention_rows if item["update"] == 385)
        final_intervention = intervention_rows[-1]
        if final_intervention["branch_vs_control_parameter_distance_norm"] > immediate["branch_vs_control_parameter_distance_norm"] * 1.000001:
            immediate_vs_accumulated = "The endpoint difference is descriptively accompanied by a nonzero immediate update-385 divergence and a larger later branch distance, with moment distances propagated across updates 386-392; this supports a descriptive accumulated-path association but does not establish a secondary causal mechanism beyond the sealed treatment effect."
        elif final_intervention["branch_vs_control_parameter_distance_norm"] < immediate["branch_vs_control_parameter_distance_norm"] * 0.999999:
            immediate_vs_accumulated = "The endpoint difference is descriptively accompanied by an immediate update-385 divergence followed by partial reconvergence; the recorded measurements do not support calling later accumulation the dominant descriptor."
        else:
            immediate_vs_accumulated = "The endpoint difference is descriptively associated with the immediate update-385 perturbation, while the recorded later distances remain approximately stable; no stronger secondary mechanism is claimed."
        q_answers = {
            "Q1": f"YES; CONTROL reproduction status={control_reproduction['status']} and endpoint anchor gates passed.",
            "Q2": "YES; the only treatment was deletion of w_32 * SmoothL1(free_prediction_32,target_32; beta=0.001).",
            "Q3": "YES; the H32 scalar was omitted from the INTERVENTION objective before total_loss.backward().",
            "Q4": "YES; each branch called torch.nn.utils.clip_grad_norm_ on its own live gradients.",
            "Q5": "YES; both branches independently propagated parameters, step, exp_avg, exp_avg_sq, optimizer metadata, and RNG through updates 385-392.",
            "Q6": f"CONTROL_H1={endpoint['control']['observed_h1']:.17g}; INTERVENTION_H1={endpoint['intervention']['observed_h1']:.17g}; E_H1={effect:.17g}.",
            "Q7": f"{'YES' if material else 'NO'}; E_H1={effect:.17g} versus threshold {H1_MATERIAL_EFFECT_THRESHOLD:.17g}.",
            "Q8": f"CONTROL_H32={endpoint['control']['observed_h32']:.17g}; INTERVENTION_H32={endpoint['intervention']['observed_h32']:.17g}; H32_PRESERVED={'YES' if preserved else 'NO'} against {H32_PASS_THRESHOLD:.17g}.",
            "Q9": f"{classification_label}.",
            "Q10": f"At update 385, post-update parameter distance={immediate['branch_vs_control_parameter_distance_norm']:.17g} and update-increment difference={immediate['update_increment_difference_norm']:.17g}.",
            "Q11": f"Across updates 386-392, final parameter distance={final_intervention['branch_vs_control_parameter_distance_norm']:.17g}, exp_avg distance={final_intervention['branch_vs_control_exp_avg_difference_norm']:.17g}, exp_avg_sq distance={final_intervention['branch_vs_control_exp_avg_sq_difference_norm']:.17g}; see the full sequence in per_update_measurements.json.",
            "Q12": f"Within this exact canonical eight-update recipe, D16 {'supports' if material else 'does not support'} the hypothesis that deleting weighted rollout-H32 materially changes endpoint H1 in the improvement direction; no broader configuration is inferred.",
        }
        result = {"status": "PASSED", "first_blocker": "none", "classification": classification, "estimand": estimand, "endpoint": endpoint, "control_reproduction": control_reproduction, "per_update_measurements": all_records, "immediate_vs_accumulated": immediate_vs_accumulated, "q_answers": q_answers, "protocol": {"update_393_executed": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "hyperparameter_tuning_executed": "NO", "historical_artifacts_modified": "NO"}, "one_sentence_verdict": f"Deleting weighted rollout-H32 over updates 385-392 produced E_H1={effect:.17g}, so the sealed classification is {classification_label}; H32 preservation={'YES' if preserved else 'NO'}."}
        write_json(output / "per_update_measurements.json", all_records)
        with (output / "per_update_measurements.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["update", "branch", "raw_total_gradient_norm", "exact_clip_coefficient", "post_clipping_gradient_norm", "parameter_update_norm", "branch_vs_control_parameter_distance_norm", "branch_vs_control_exp_avg_difference_norm", "branch_vs_control_exp_avg_sq_difference_norm", "update_increment_difference_norm", "parameter_distance_increment", "per_update_H1", "per_update_H32"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in all_records)
        write_json(output / "endpoint_evidence.json", endpoint)
        write_json(output / "causal_estimand.json", estimand)
        write_json(output / "causal_classification.json", classification)
        write_json(output / "control_reproduction.json", control_reproduction)
        write_json(output / "gradient_and_clipping_records.json", {f"{row['branch']}_{row['update']}": {key: row[key] for key in ("raw_total_gradient_norm", "returned_pre_clipping_total_norm", "exact_clip_coefficient", "post_clipping_gradient_norm", "clipping_activated", "raw_gradient_digest", "post_clipping_gradient_digest", "gradient_none_pattern", "h1_gradient_norm", "h32_gradient_norm")} for row in all_records})
        write_json(output / "adamw_decomposition_records.json", {f"{row['branch']}_{row['update']}": {key: row[key] for key in ("parameter_update_norm", "loss_driven_adamw_displacement_norm", "decoupled_weight_decay_displacement_norm", "adamw_decomposition_residual_norm", "adamw_decomposition_status", "step_counter_before", "step_counter_after", "exp_avg_norm", "exp_avg_sq_norm")} for row in all_records})
        write_json(output / "rng_records.json", {f"{row['branch']}_{row['update']}": {key: row[key] for key in ("rng_before_digest", "rng_after_diagnostics_digest", "rng_after_training_digest", "rng_after_validation_digest", "branch_rng_contract_status")} for row in all_records})
        write_json(output / "protocol_integrity.json", {"status": "PASSED", "first_blocker": "none", "project_owner_execution_authorization": "YES", "d16_design_review_passed": "YES", "d16_design_manifest_verified": "YES", "certified_update_384_identity": "MATCH", "source_revision_match": "YES", "runtime_identity_match": "YES", "canonical_batch_schedule_match": "YES", "rng_matching_contract": "PASS", "control_branch_executed": "YES", "intervention_branch_executed": "YES", "updates_385_392_executed": "YES", "update_393_executed": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "hyperparameter_tuning_executed": "NO", "historical_artifacts_modified": "NO", "control_reproduction": control_reproduction, "classification": classification_label})
        write_json(output / "terminal_certificate.json", {"experiment_identifier": EXPERIMENT_ID, "status": "PASSED", "first_blocker": "none", "updates_executed": list(UPDATES), "update_393_executed": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifacts_modified": "NO", "causal_classification": classification_label, "causal_conclusion_permitted": "YES", "sha256_manifest_verified": "PENDING_FINALIZATION"})
        result["design_manifest_sha256"] = design["design_manifest"]["sha256"]
        write_json(output / "result.json", result)
        logs.append(f"{utc_now()} CONTROL_REPRODUCTION={control_reproduction['status']} CLASSIFICATION={classification_label}")
        logs.append(f"{utc_now()} D16_EXECUTION_PASSED updates=385..392 update_393=NO")
        (output / "execution.log").write_text("\n".join(logs) + "\n", encoding="utf-8", newline="\n")
        provisional_manifest = write_manifest(output)
        provisional_check = verify_output_manifest(output, provisional_manifest)
        if provisional_check["status"] != "VERIFIED":
            raise StageBlocker("OUTPUT_MANIFEST_PRECHECK_FAILED")
        write_json(output / "manifest_verification.json", {"status": "PENDING_FINAL_REPORT", "checked_before_report": provisional_check})
        final_manifest = write_manifest(output)
        final_check = verify_output_manifest(output, final_manifest)
        if final_check["status"] != "VERIFIED":
            raise StageBlocker("OUTPUT_MANIFEST_PRE_REPORT_FAILED")
        report = render_report(result, output, final_check)
        (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
        final_manifest = write_manifest(output)
        final_check = verify_output_manifest(output, final_manifest)
        if final_check["status"] != "VERIFIED":
            raise StageBlocker("OUTPUT_MANIFEST_POST_REPORT_FAILED")
        write_json(output / "manifest_verification.json", final_check)
        terminal = json.loads((output / "terminal_certificate.json").read_text(encoding="utf-8"))
        terminal["sha256_manifest_verified"] = "YES"
        terminal.pop("sha256_manifest_sha256", None)
        write_json(output / "terminal_certificate.json", terminal)
        final_manifest = write_manifest(output)
        final_check = verify_output_manifest(output, final_manifest)
        if final_check["status"] != "VERIFIED":
            raise StageBlocker("OUTPUT_MANIFEST_FINAL_FAILED")
        write_json(output / "manifest_verification.json", {key: final_check[key] for key in ("status", "checked", "total_bytes", "failures")})
        final_manifest = write_manifest(output)
        final_check = verify_output_manifest(output, final_manifest)
        if final_check["status"] != "VERIFIED":
            raise StageBlocker("OUTPUT_MANIFEST_FINAL_RECORD_FAILED")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification_label, "E_H1": effect, "CONTROL_H1": endpoint["control"]["observed_h1"], "INTERVENTION_H1": endpoint["intervention"]["observed_h1"], "CONTROL_H32": endpoint["control"]["observed_h32"], "INTERVENTION_H32": endpoint["intervention"]["observed_h32"], "manifest_sha256": final_check["manifest_sha256"], "file_count": final_check["checked"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), design)
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return fail_closed(output, f"UNEXPECTED_EXECUTION_FAILURE:{type(exc).__name__}:{exc}", design)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args().output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
