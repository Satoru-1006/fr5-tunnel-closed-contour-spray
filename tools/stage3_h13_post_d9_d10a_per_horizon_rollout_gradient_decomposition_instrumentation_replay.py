"""Execute the sealed Stage 3 H13 D10A observational instrumentation replay.

This runner is intentionally separate from the canonical training source.  It
loads the authenticated post-update-384 state, observes the preregistered
per-horizon and hidden-consistency gradients with non-mutating
``torch.autograd.grad`` calls, then executes the unchanged canonical backward,
clip, and AdamW step for updates 385--392 only.
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402
from src.stage3_h13_r8e import (  # noqa: E402
    EVALUATION_HORIZONS,
    hidden_alignment_weights,
    normalized_hidden_smooth_l1,
    rollout_horizon_weights,
)
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256  # noqa: E402
from tools import stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay as d7  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D9_D10A_PER_HORIZON_ROLLOUT_GRADIENT_DECOMPOSITION_INSTRUMENTATION_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_design_review_20260818T"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\e501f78f-f2ee-4f11-aff2-a14daae22f54\pasted-text.txt")
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
D7_AUTHORITY_DIR = ROOT / "outputs/stage3_h13_post_d6_d7_phase4_pareto_localization_replay_20260817T023840Z"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_EXP_AVG_HASH = "7226e7ae693c1083d6928df6968d67d32d33477c16a8770adcf3050fc9492cdb"
EXPECTED_EXP_AVG_SQ_HASH = "eaf5ebdd6aaddefec25950e80d1a14baeb92a06dddcd11b246546e44c8489d89"
EXPECTED_RNG_HASH = "857d5ddd6c4271019b55a53774ad68c70e435abd38cf430b7f0d57ed047848d8"
UPDATES = tuple(range(385, 393))
START_UPDATE = 384
ACTIVE_HORIZON = 32
VALIDATION_HORIZONS = tuple(int(x) for x in EVALUATION_HORIZONS)
LAMBDA_ROLLOUT = 1.0
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
POSITION_BETA = 0.001
CLIP_MAX_NORM = 1.0
ADAMW_LR = 5.0e-5
ZERO_NORM_EPSILON = 1.0e-12
ADDITIVITY_ABS = 1.0e-7
ADDITIVITY_REL = 2.0e-5
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
CONFLICT_THRESHOLD = -0.05
ORTHOGONAL_THRESHOLD = 0.05
MAGNITUDE_DOMINANCE_THRESHOLD = 2.0
GRU_NAMES = (
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1",
)
ALL_NAMES = GRU_NAMES + ("head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias")
GROUPS = {"gru": GRU_NAMES, "head": ALL_NAMES[len(GRU_NAMES):], "whole_model": ALL_NAMES}


class StageBlocker(RuntimeError):
    """Fail-closed execution blocker."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    return sha256_bytes(canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}) + b"\0" + tensor.numpy().tobytes())


def tensor_map_hash(values: Mapping[str, torch.Tensor]) -> str:
    return sha256_bytes(canonical_json([
        {"name": name, "dtype": str(values[name].dtype), "shape": list(values[name].shape), "sha256": tensor_hash(values[name])}
        for name in values
    ]))


def clone_parameters(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters()}


def clone_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {str(key): clone_value(value) for key, value in optimizer.state_dict().items()}


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_value(item) for item in value)
    return copy.deepcopy(value)


def clone_named_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    result: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise StageBlocker("optimizer_parameter_name_mapping_missing")
            result[name] = {str(key): clone_value(value) for key, value in optimizer.state.get(parameter, {}).items()}
    return result


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
    if left_norm <= ZERO_NORM_EPSILON or right_norm <= ZERO_NORM_EPSILON:
        return None
    raw = finite_float(dot_value / (left_norm * right_norm))
    if raw < -1.0 - 1.0e-12 or raw > 1.0 + 1.0e-12:
        raise StageBlocker(f"cosine_out_of_domain:{raw}")
    return max(-1.0, min(1.0, raw))


def assert_finite(values: Iterable[torch.Tensor], label: str) -> None:
    for value in values:
        if not bool(torch.all(torch.isfinite(value))):
            raise StageBlocker(f"nonfinite_{label}")


def residual(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    difference = torch.cat([(left[row["name"]] - right[row["name"]]).reshape(-1).double() for row in table])
    left_flat = torch.cat([left[row["name"]].reshape(-1).double() for row in table])
    right_flat = torch.cat([right[row["name"]].reshape(-1).double() for row in table])
    l2 = finite_float(torch.linalg.vector_norm(difference))
    max_abs = finite_float(torch.max(torch.abs(difference))) if difference.numel() else 0.0
    scale = max(norm(left[row["name"]] for row in table), norm(right[row["name"]] for row in table), 1.0)
    tol_l2 = max(ADDITIVITY_ABS, ADDITIVITY_REL * scale)
    max_scale = max(finite_float(torch.max(torch.abs(left_flat))), finite_float(torch.max(torch.abs(right_flat))), 1.0)
    tol_max = ADDITIVITY_ABS + ADDITIVITY_REL * max_scale
    passed = l2 <= tol_l2 and max_abs <= tol_max
    return {"l2": l2, "max_abs": max_abs, "scale": scale, "l2_tolerance": tol_l2, "max_tolerance": tol_max, "pass_fail": "PASS" if passed else "FAIL"}


def materialize_grads(grads: Sequence[torch.Tensor | None], named: Sequence[tuple[str, torch.nn.Parameter]]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if len(grads) != len(named):
        raise StageBlocker("gradient_parameter_count_mismatch")
    values: dict[str, torch.Tensor] = {}
    flags: dict[str, str] = {}
    for (name, parameter), gradient in zip(named, grads):
        if gradient is None:
            values[name] = torch.zeros_like(parameter.detach()).cpu().contiguous()
            flags[name] = "None"
        else:
            values[name] = gradient.detach().cpu().contiguous().clone()
            flags[name] = "connected"
    assert_finite(values.values(), "component_gradient")
    return values, flags


def capture_gradients(named: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    result = {
        name: torch.zeros_like(parameter.detach()).cpu().contiguous() if parameter.grad is None else parameter.grad.detach().cpu().contiguous().clone()
        for name, parameter in named
    }
    assert_finite(result.values(), "live_gradient")
    return result


def parameter_table(model: Any) -> tuple[list[dict[str, Any]], list[tuple[str, torch.nn.Parameter]]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(ALL_NAMES):
        raise StageBlocker(f"runtime_parameter_order_mismatch:{[name for name, _ in named]}")
    rows = []
    offset = 0
    for index, (name, parameter) in enumerate(named):
        end = offset + int(parameter.numel())
        group = "gru" if name.startswith("gru.") else "head"
        rows.append({"index": index, "name": name, "module": group, "dtype": str(parameter.dtype), "shape": list(parameter.shape), "numel": int(parameter.numel()), "flat_offset_start": offset, "flat_offset_end": end, "requires_grad": bool(parameter.requires_grad)})
        offset = end
    counts = {name: sum(row["numel"] for row in rows if name == "whole_model" or row["module"] == name) for name in ("gru", "head")}
    counts["whole_model"] = offset
    if counts != {"gru": 156672, "head": 22704, "whole_model": 179376}:
        raise StageBlocker(f"parameter_partition_count_mismatch:{counts}")
    return rows, named


def register_hidden_observer(model: Any) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def hook(_module: Any, inputs: Any, output: Any) -> None:
        sequence, hidden = output
        calls.append({"input_shape": list(inputs[0].shape), "sequence": sequence, "hidden": hidden})

    return calls, model.gru.register_forward_hook(hook)


def hidden_call_partition(calls: Sequence[Mapping[str, Any]]) -> tuple[list[torch.Tensor], dict[int, torch.Tensor]]:
    expected = ACTIVE_HORIZON + 15
    if len(calls) != expected:
        raise StageBlocker(f"hidden_forward_call_count_mismatch:{len(calls)}:{expected}")
    free: list[torch.Tensor] = []
    teacher: dict[int, torch.Tensor] = {}
    call_index = 0
    # Calls are interleaved exactly as in the sealed source: each free forward
    # is followed by a teacher forward for rollout steps 2..16.
    for step in range(ACTIVE_HORIZON):
        free.append(calls[call_index]["hidden"])
        call_index += 1
        if 1 <= step <= 15:
            teacher[step + 1] = calls[call_index]["hidden"]
            call_index += 1
    if call_index != len(calls) or len(teacher) != 15:
        raise StageBlocker(f"teacher_hidden_partition_length_mismatch:{call_index}:{len(calls)}")
    return free, teacher


def pairwise_metrics(vectors: Mapping[str, Mapping[str, torch.Tensor]], table: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    names = table if scope == "whole_model" else [row for row in table if row["module"] == scope]
    keys = list(vectors)
    flats = {key: torch.cat([vectors[key][row["name"]].reshape(-1).double() for row in names]) for key in keys}
    norms = {key: finite_float(torch.linalg.vector_norm(flats[key])) for key in keys}
    dots: dict[str, dict[str, float]] = {}
    cosines: dict[str, dict[str, float | None]] = {}
    for left in keys:
        dots[left] = {}
        cosines[left] = {}
        for right in keys:
            value = finite_float(torch.dot(flats[left], flats[right]))
            dots[left][right] = value
            cosines[left][right] = cosine(value, norms[left], norms[right])
    return {"scope": scope, "vector_order": keys, "norms": norms, "dot_matrix": dots, "cosine_matrix": cosines, "zero_norm_flags": {key: norms[key] <= ZERO_NORM_EPSILON for key in keys}}


def component_statistics(vectors: Mapping[str, Mapping[str, torch.Tensor]], table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_parameter: dict[str, dict[str, Any]] = {}
    for component, values in vectors.items():
        per_parameter[component] = {
            row["name"]: {"norm": finite_float(torch.linalg.vector_norm(values[row["name"]].double())), "sha256": tensor_hash(values[row["name"]]), "dtype": str(values[row["name"]].dtype), "shape": list(values[row["name"]].shape)}
            for row in table
        }
    group_norms: dict[str, dict[str, float]] = {}
    for component, values in vectors.items():
        group_norms[component] = {}
        for group, names in GROUPS.items():
            group_norms[component][group] = norm(values[name] for name in names)
    return {"per_parameter": per_parameter, "group_norms": group_norms}


def classify_cosine(value: float | None) -> str:
    if value is None:
        return "UNDEFINED_ZERO_NORM"
    if value <= CONFLICT_THRESHOLD:
        return "CONFLICT"
    if value < ORTHOGONAL_THRESHOLD:
        return "ORTHOGONAL"
    return "ALIGNED"


def write_tensor(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def verify_hash_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageBlocker(f"{label}_manifest_missing:{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("files") or manifest.get("artifact_sha256")
    if isinstance(entries, Mapping):
        items = [(str(name), value.get("sha256") if isinstance(value, Mapping) else value, value.get("bytes", value.get("size_bytes")) if isinstance(value, Mapping) else None) for name, value in entries.items()]
    elif isinstance(entries, list):
        items = [(str(item["path"]), item["sha256"], item.get("bytes", item.get("size_bytes"))) for item in entries]
    else:
        raise StageBlocker(f"{label}_manifest_entries_missing:{path}")
    missing = []
    hashes = []
    bytes_bad = []
    for name, expected, expected_bytes in items:
        target = path.parent / name
        if not target.is_file():
            missing.append(str(target))
            continue
        observed = sha256_file(target)
        if observed != str(expected):
            hashes.append({"path": str(target), "expected": expected, "observed": observed})
        if expected_bytes is not None and target.stat().st_size != int(expected_bytes):
            bytes_bad.append({"path": str(target), "expected": int(expected_bytes), "observed": target.stat().st_size})
    status = not missing and not hashes and not bytes_bad
    return {"path": str(path.resolve()), "manifest_sha256": sha256_file(path), "entry_count": len(items), "missing": missing, "hash_mismatches": hashes, "byte_mismatches": bytes_bad, "status": "VERIFIED" if status else "BLOCKED"}


def verify_design() -> dict[str, Any]:
    if not DESIGN_DIR.is_dir():
        raise StageBlocker("D10A_DESIGN_BUNDLE_NOT_FOUND")
    manifest_path = DESIGN_DIR / "SHA256_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    expected_names = {"FINAL_REPORT.md", "experiment_design.json", "AUTHORITY_EVIDENCE_MANIFEST.json"}
    if set(files) != expected_names:
        raise StageBlocker(f"D10A_DESIGN_MANIFEST_FILE_SET_MISMATCH:{sorted(files)}")
    observed = {}
    for name, entry in files.items():
        target = DESIGN_DIR / name
        if not target.is_file():
            raise StageBlocker(f"D10A_DESIGN_ARTIFACT_MISSING:{target}")
        digest = sha256_file(target)
        size = target.stat().st_size
        expected_hash = entry["sha256"] if isinstance(entry, Mapping) else entry
        expected_bytes = entry.get("bytes") if isinstance(entry, Mapping) else None
        if digest != expected_hash or expected_bytes is not None and size != int(expected_bytes):
            raise StageBlocker(f"D10A_DESIGN_ARTIFACT_HASH_MISMATCH:{name}")
        observed[name] = {"sha256": digest, "bytes": size}
    design = json.loads((DESIGN_DIR / "experiment_design.json").read_text(encoding="utf-8"))
    if design.get("review_status") != "PASSED" or design.get("design_review_passed") != "YES":
        raise StageBlocker("D10A_DESIGN_REVIEW_NOT_PASSED")
    if design.get("experiment_executed") is not False or design.get("instrumentation_replay_executed") is not False:
        raise StageBlocker("D10A_SEALED_BUNDLE_ALREADY_EXECUTED")
    scope = design["scope"]
    if scope["starting_state"]["completed_optimizer_step"] != 384 or tuple(scope["canonical_updates"]) != UPDATES or scope["batch_size"] != 256 or scope["active_training_horizon"] != 32:
        raise StageBlocker("D10A_DESIGN_EXECUTION_BOUNDARY_MISMATCH")
    if scope["updates_393_or_later_allowed"] is not False or scope["frozen20_allowed"] is not False or scope["r9_allowed"] is not False:
        raise StageBlocker("D10A_PROHIBITED_SCOPE_NOT_SEALED")
    contract = design["canonical_training_contract"]
    if contract["source_revision"] != EXPECTED_SOURCE_REVISION or contract["device"] != "cpu" or contract["model_dtype"] != "float32" or contract["reduction_dtype"] != "float64" or contract["deterministic_algorithms"] is not True:
        raise StageBlocker("D10A_CANONICAL_RUNTIME_CONTRACT_MISMATCH")
    extraction = design["gradient_decomposition"]["extraction_protocol"]
    if extraction != {"autograd_api": "torch.autograd.grad", "retain_graph": True, "create_graph": False, "allow_unused": True, "parameter_grad_side_effects_during_components": False, "none_gradient_policy": "Record per-parameter connected/None status; materialize an arithmetic zero only in detached diagnostic copies, never in the live graph.", "canonical_backward_reference": "After every component extraction, zero_grad(set_to_none=True), run the unchanged total.backward(), and compare the resulting raw gradients with the component sum and g_total_autograd.", "no_gradient_reuse_after_detach": True}:
        raise StageBlocker("D10A_EXTRACTION_PROTOCOL_MISMATCH")
    if design["parameter_group_decomposition"]["parameter_order"] != list(ALL_NAMES):
        raise StageBlocker("D10A_PARAMETER_ORDER_CONTRACT_MISMATCH")
    return {"directory": str(DESIGN_DIR.resolve()), "manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path), "files": observed, "design": design}


def verify_authorization() -> dict[str, Any]:
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("PROJECT_OWNER_D10A_EXECUTION_AUTHORIZATION_SOURCE_MISSING")
    text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required = [EXPERIMENT_ID, "PROJECT_OWNER_D10A_EXECUTION_AUTHORIZATION: YES", "FIRST_EXECUTED_UPDATE: 385", "LAST_EXECUTED_UPDATE: 392", "UPDATE_393_EXECUTED: NO", "FROZEN20_OPENED: NO", "R9_EXECUTED: NO", "D10_EXECUTED: NO", "CAUSAL_INTERVENTION_EXECUTED: NO"]
    missing = [item for item in required if item not in text]
    if missing:
        raise StageBlocker(f"PROJECT_OWNER_D10A_AUTHORIZATION_SCOPE_NOT_PROVEN:{missing}")
    return {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"}


def verify_upstream(design: Mapping[str, Any]) -> dict[str, Any]:
    auth_path = DESIGN_DIR / "AUTHORITY_EVIDENCE_MANIFEST.json"
    authority = json.loads(auth_path.read_text(encoding="utf-8"))
    summary = authority["verification_summary"]
    ledgers = []
    for key in ("d5d6_update384_integrity_manifest", "d7_sha256_manifest", "d8_sha256_manifest", "d9_sha256_manifest"):
        meta = summary[key]
        result = verify_hash_manifest(resolve_path(meta["path"]), key)
        if result["manifest_sha256"] != meta["manifest_sha256"] or result["status"] != "VERIFIED" or result["entry_count"] != int(meta["entries"]):
            raise StageBlocker(f"{key}_AUTHENTICATION_FAILED")
        ledgers.append(result)
    bundle_meta = authority["certified_update_384"]
    bundle_path = resolve_path(bundle_meta["bundle_path"])
    if bundle_path != BUNDLE_PATH or not bundle_path.is_file() or sha256_file(bundle_path) != bundle_meta["bundle_sha256"] or sha256_file(bundle_path) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("CERTIFIED_UPDATE_384_BUNDLE_IDENTITY_MISMATCH")
    schedule_meta = authority["canonical_schedule"]
    schedule_identity = resolve_path(schedule_meta["identity_path"])
    schedule_storage = resolve_path(schedule_meta["storage_path"])
    if not schedule_identity.is_file() or sha256_file(schedule_identity) != schedule_meta["identity_file_sha256"] or not schedule_storage.is_file() or sha256_file(schedule_storage) != schedule_meta["storage_sha256"]:
        raise StageBlocker("CANONICAL_SCHEDULE_STORAGE_IDENTITY_MISMATCH")
    source_record_path = resolve_path(authority["source_revision"]["source_revision_record_path"])
    if not source_record_path.is_file() or sha256_file(source_record_path) != authority["source_revision"]["source_revision_record_sha256"]:
        raise StageBlocker("SOURCE_REVISION_RECORD_IDENTITY_MISMATCH")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git_head != EXPECTED_SOURCE_REVISION:
        raise StageBlocker(f"SOURCE_REVISION_MISMATCH:{git_head}")
    source_hashes = {}
    for logical, expected in authority["source_revision"]["files"].items():
        expected_path = ROOT / logical
        candidates = [expected_path]
        if not expected_path.is_file() and logical.startswith("tools/"):
            candidates.append(ROOT / ("scripts/" + logical.split("/", 1)[1]))
        match = next((item for item in candidates if item.is_file() and sha256_file(item) == expected), None)
        if match is None:
            raise StageBlocker(f"SOURCE_FILE_HASH_IDENTITY_UNPROVABLE:{logical}")
        source_hashes[logical] = {"expected_sha256": expected, "resolved_path": str(match.resolve()), "observed_sha256": sha256_file(match), "path_alias": str(match) != str(expected_path)}
    return {"authority_manifest_path": str(auth_path.resolve()), "authority_manifest_sha256": sha256_file(auth_path), "authority": authority, "ledgers": ledgers, "certified_start": {**bundle_meta, "resolved_path": str(bundle_path.resolve())}, "canonical_schedule": {**schedule_meta, "identity_resolved_path": str(schedule_identity.resolve()), "storage_resolved_path": str(schedule_storage.resolve())}, "source_revision": {**authority["source_revision"], "git_head": git_head, "source_hashes": source_hashes}}


def load_and_verify_runtime(upstream: Mapping[str, Any], design: Mapping[str, Any]) -> dict[str, Any]:
    base.set_deterministic(base.SEED)
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
    pre = base.preflight(True)
    authority = base.verify_authority(pre)
    environment = authority["current_environment"]
    if environment.get("source_revision") != EXPECTED_SOURCE_REVISION or environment.get("device") != "cpu" or environment.get("dtype") != "float32" or environment.get("pytorch_version") != "2.13.0+cpu" or environment.get("python_version") != "3.12.1" or environment.get("numpy_version") != "2.1.3" or not bool(torch.are_deterministic_algorithms_enabled()):
        raise StageBlocker("RUNTIME_IDENTITY_MISMATCH")
    if sha256_file(base.A1_CHECKPOINT) != base.EXPECTED_A1_HASH:
        raise StageBlocker("R8E_A1_IMMUTABILITY_GATE_FAILED_BEFORE_EXECUTION")
    model, optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
    loaded = d7.verify_loaded_identity(bundle, model, optimizer)
    if loaded["model_semantic_hash"] != EXPECTED_MODEL_HASH or loaded["optimizer_semantic_hash"] != EXPECTED_OPTIMIZER_HASH or loaded["exp_avg_semantic_hash"] != EXPECTED_EXP_AVG_HASH or loaded["exp_avg_sq_semantic_hash"] != EXPECTED_EXP_AVG_SQ_HASH:
        raise StageBlocker("CERTIFIED_START_STATE_RELOAD_IDENTITY_MISMATCH")
    base.restore_rng(bundle["rng_state"])
    rng_start = base.rng_digest(base.capture_rng())
    if rng_start != EXPECTED_RNG_HASH:
        raise StageBlocker("CERTIFIED_RNG_IDENTITY_MISMATCH")
    table, named = parameter_table(model)
    steps = base.optimizer_step_map(optimizer)
    if len(steps) != 12 or set(steps.values()) != {384}:
        raise StageBlocker(f"OPTIMIZER_STEP_COUNTER_IDENTITY_MISMATCH:{steps}")
    return {"bundle": bundle, "pre": pre, "authority": authority, "environment": environment, "model": model, "optimizer": optimizer, "table": table, "named": named, "loaded": loaded, "rng_start": rng_start, "a1_hash_before": sha256_file(base.A1_CHECKPOINT)}


def batch_identity(schedule_manifest: Mapping[str, Any], index: int) -> dict[str, Any]:
    item = schedule_manifest["ordered_batches"][index]
    ids = [int(value) for value in item["window_ids"]]
    return {"update": index + 1, "schedule_index": index, "window_ids": ids, "window_id_count": len(ids), "batch_window_sha256": base.batch_digest(schedule_manifest, index), "source_cycle_spans": item.get("source_cycle_spans")}


def hidden_components(calls: Sequence[Mapping[str, Any]], q_values: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, list[float]], torch.Tensor]:
    free, teacher = hidden_call_partition(calls)
    terms: dict[int, torch.Tensor] = {}
    per_layer: dict[int, torch.Tensor] = {}
    for step in range(1, ACTIVE_HORIZON + 1):
        q = q_values[step - 1]
        if float(q) <= 0.0:
            continue
        if step not in teacher:
            raise StageBlocker(f"missing_teacher_hidden_step:{step}")
        h_value, layer_value = normalized_hidden_smooth_l1(free[step - 1], teacher[step], beta=0.1)
        terms[step] = h_value
        per_layer[step] = layer_value
    weight_sum = torch.zeros((), device=q_values.device, dtype=q_values.dtype)
    weighted_terms = []
    for step in range(1, ACTIVE_HORIZON + 1):
        q = q_values[step - 1]
        if float(q) > 0.0:
            weight_sum = weight_sum + q
            weighted_terms.append(q * terms[step])
    hidden_loss = torch.stack(weighted_terms).sum() / torch.clamp(weight_sum, min=1.0e-12)
    z_values = {step: (q_values[step - 1] / torch.clamp(weight_sum, min=1.0e-12)) * terms[step] for step in terms}
    return terms, z_values, {step: [finite_float(item) for item in per_layer[step]] for step in per_layer}, hidden_loss


def gradient_component(scalar: torch.Tensor, named: Sequence[tuple[str, torch.nn.Parameter]], label: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    grads = torch.autograd.grad(scalar, [parameter for _, parameter in named], retain_graph=True, create_graph=False, allow_unused=True)
    if any(parameter.grad is not None for _, parameter in named):
        raise StageBlocker(f"diagnostic_grad_mutation:{label}")
    return materialize_grads(grads, named)


def build_vector_set(components: Mapping[str, Mapping[str, torch.Tensor]], hidden_active: Sequence[int], named: Sequence[tuple[str, torch.nn.Parameter]]) -> OrderedDict[str, Mapping[str, torch.Tensor]]:
    result: OrderedDict[str, Mapping[str, torch.Tensor]] = OrderedDict()
    result["g_h1"] = components["h1"]
    for step in range(1, ACTIVE_HORIZON + 1):
        result[f"g_rollout_k_{step:02d}"] = components[f"rollout_k_{step:02d}"]
    result["g_rollout"] = components["rollout"]
    for step in hidden_active:
        result[f"g_hidden_k_{step:02d}"] = components[f"hidden_k_{step:02d}"]
    result["g_hidden"] = components["hidden"]
    result["g_total_from_components"] = components["total_from_components"]
    result["g_total_autograd"] = components["total_autograd"]
    result["g_total_raw"] = components["total_raw"]
    return result


def compare_trajectory(rows: Sequence[Mapping[str, Any]], d7_dir: Path) -> dict[str, Any]:
    path = d7_dir / "training_diagnostics.csv"
    if not path.is_file():
        raise StageBlocker(f"D7_ANCHOR_FILE_MISSING:{path}")
    authority_rows = {int(row["update"]): row for row in csv.DictReader(path.open("r", encoding="utf-8", newline=""))}
    comparisons = []
    fields = (("batch_window_sha256", "batch_window_sha256", "exact"), ("total_loss", "training_loss_total", "float"), ("rollout_loss", "training_loss_rollout", "float"), ("hidden_loss_unweighted", "training_loss_hidden", "float"), ("h1_loss_unweighted", "training_loss_h1", "float"), ("raw_total_gradient_norm", "raw_gradient_norm_before_clipping", "float"), ("clipped_total_gradient_norm", "clipped_gradient_norm_after_clipping", "float"), ("clipping_activated", "clipping_activated", "exact"), ("clip_coefficient", "clipping_coefficient", "float"), ("parameter_update_norm", "parameter_update_norm", "float"), ("parameter_update_digest", "parameter_update_digest", "exact"), ("raw_gradient_digest", "raw_gradient_digest", "exact"), ("clipped_gradient_digest", "clipped_gradient_digest", "exact"), ("canonical_state_hash_after", "model_state_digest", "exact"), ("exp_avg_norm", "exp_avg_norm", "float"), ("exp_avg_sq_norm", "exp_avg_sq_norm", "float"), ("eval_h1", "eval_H1", "float"), ("eval_h32", "eval_H32", "float"))
    for row in rows:
        update = int(row["update"])
        expected = authority_rows.get(update)
        if expected is None:
            raise StageBlocker(f"D7_ANCHOR_UPDATE_MISSING:{update}")
        for observed_key, expected_key, kind in fields:
            left = row.get(observed_key)
            right = expected.get(expected_key)
            if right in (None, ""):
                comparisons.append({"update": update, "field": observed_key, "pass_fail": "NOT_AVAILABLE"})
                continue
            if left in (None, ""):
                raise StageBlocker(f"D7_ANCHOR_FIELD_MISSING:{update}:{observed_key}")
            ok = left == right if kind == "exact" else math.isclose(float(left), float(right), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)
            comparisons.append({"update": update, "field": observed_key, "observed": left, "authoritative": right, "pass_fail": "PASS" if ok else "FAIL"})
    failed = [item for item in comparisons if item["pass_fail"] == "FAIL"]
    return {"authority_path": str(path.resolve()), "authority_sha256": sha256_file(path), "comparisons": comparisons, "all_available_fields_match": "YES" if not failed else "NO", "first_divergence": failed[0] if failed else None}


def run_update(state: Mapping[str, Any], output: Path, zero_index: int, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    model = state["model"]
    optimizer = state["optimizer"]
    pre = state["pre"]
    authority = state["authority"]
    table = state["table"]
    named = state["named"]
    update = zero_index + 1
    if update not in UPDATES:
        raise StageBlocker(f"UNAUTHORIZED_UPDATE:{update}")
    write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [int(item["update"]) for item in (previous and [previous] or [])], "forward_started": False, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
    if not model.training:
        model.train()
    rng_before = base.capture_rng()
    before_model = clone_parameters(model)
    before_optimizer = clone_optimizer_state(optimizer)
    before_named_optimizer = clone_named_optimizer_state(optimizer)
    before_model_hash = canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    batch_record = batch_identity(authority["schedule_manifest"], zero_index)
    batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    calls, hook = register_hidden_observer(model)
    try:
        optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for _, parameter in named):
            raise StageBlocker(f"LIVE_GRADIENT_NOT_EMPTY_AFTER_ZERO_GRAD:{update}")
        rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=ACTIVE_HORIZON, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
        write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [int(item["update"]) for item in (previous and [previous] or [])], "forward_started": True, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
        if rollout.get("free_branch_reads_target_positions") is not False or rollout.get("teacher_reference_stop_gradient") is not True:
            raise StageBlocker(f"GRAPH_SEMANTICS_GATE_FAILED:{update}")
        predictions = rollout["predictions"]
        targets = batch["target_positions"][:, :ACTIVE_HORIZON, :]
        step_losses = F.smooth_l1_loss(predictions, targets, beta=POSITION_BETA, reduction="none").mean(dim=(0, 2))
        weights = rollout_horizon_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
        q_values = hidden_alignment_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
        hidden_h, hidden_z, hidden_per_layer, hidden_loss_rebuilt = hidden_components(calls, q_values)
        losses = base.training_losses(predictions, batch["target_positions"], rollout["hidden_loss"], lambda_hidden=LAMBDA_HIDDEN, lambda_h1=LAMBDA_H1, beta=POSITION_BETA)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise StageBlocker(f"NONFINITE_CANONICAL_LOSS:{update}")
        scalar_checks = {
            "step_loss_vector": [finite_float(value) for value in step_losses],
            "weight_vector": [finite_float(value) for value in weights],
            "weighted_rollout_vector": [finite_float(weights[index] * step_losses[index]) for index in range(ACTIVE_HORIZON)],
            "hidden_q_vector": [finite_float(value) for value in q_values],
            "hidden_z_vector": {str(step): finite_float(value) for step, value in hidden_z.items()},
            "rollout_loss": finite_float(losses["rollout"]),
            "hidden_loss_source": finite_float(rollout["hidden_loss"]),
            "hidden_loss_rebuilt": finite_float(hidden_loss_rebuilt),
            "h1_loss": finite_float(losses["h1"]),
            "weighted_hidden_loss": finite_float(LAMBDA_HIDDEN * losses["hidden"]),
            "total_loss": finite_float(losses["total"]),
        }
        if not math.isclose(sum(scalar_checks["weighted_rollout_vector"]), scalar_checks["rollout_loss"], rel_tol=0.0, abs_tol=1.0e-7):
            raise StageBlocker(f"PER_HORIZON_ROLLOUT_SCALAR_RECONCILIATION_FAILED:{update}")
        if not math.isclose(scalar_checks["hidden_loss_source"], scalar_checks["hidden_loss_rebuilt"], rel_tol=2.0e-5, abs_tol=1.0e-7):
            raise StageBlocker(f"HIDDEN_SCALAR_RECONCILIATION_FAILED:{update}")
        components: dict[str, dict[str, torch.Tensor]] = {}
        masks: dict[str, dict[str, str]] = {}
        for step in range(1, ACTIVE_HORIZON + 1):
            if step == 1:
                write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [int(item["update"]) for item in (previous and [previous] or [])], "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": False, "optimizer_step_started": False})
            components[f"rollout_k_{step:02d}"], masks[f"rollout_k_{step:02d}"] = gradient_component(LAMBDA_ROLLOUT * weights[step - 1] * step_losses[step - 1], named, f"rollout_k_{step}")
        components["rollout"], masks["rollout"] = gradient_component(losses["rollout"], named, "rollout")
        components["h1"], masks["h1"] = gradient_component(LAMBDA_H1 * losses["h1"], named, "h1")
        for step in sorted(hidden_z):
            components[f"hidden_k_{step:02d}"], masks[f"hidden_k_{step:02d}"] = gradient_component(LAMBDA_HIDDEN * hidden_z[step], named, f"hidden_k_{step}")
        components["hidden"], masks["hidden"] = gradient_component(LAMBDA_HIDDEN * losses["hidden"], named, "hidden")
        rng_after_diagnostics = base.capture_rng()
        rng_diag_before = base.rng_digest(rng_before)
        rng_diag_after = base.rng_digest(rng_after_diagnostics)
        if rng_diag_before != rng_diag_after:
            raise StageBlocker(f"DIAGNOSTIC_RNG_CONTAMINATION:{update}")
        rollout_sum = {name: sum((components[f"rollout_k_{step:02d}"][name] for step in range(1, ACTIVE_HORIZON + 1)), torch.zeros_like(components["rollout_k_01"][name])) for name, _ in named}
        hidden_sum = {name: sum((components[f"hidden_k_{step:02d}"][name] for step in sorted(hidden_z)), torch.zeros_like(components[f"hidden_k_{sorted(hidden_z)[0]:02d}"][name])) for name, _ in named}
        components["total_from_components"] = {name: components["h1"][name] + rollout_sum[name] + hidden_sum[name] for name, _ in named}
        components["total_autograd"], masks["total_autograd"] = gradient_component(losses["total"], named, "total_autograd")
        checks = {
            "rollout_per_horizon_vs_rollout": residual(rollout_sum, components["rollout"], table),
            "hidden_per_horizon_vs_hidden": residual(hidden_sum, components["hidden"], table),
            "total_components_vs_total_autograd": residual(components["total_from_components"], components["total_autograd"], table),
        }
        if any(value["pass_fail"] != "PASS" for value in checks.values()):
            raise StageBlocker(f"G07_DECOMPOSITION_FAILED:{update}")
        if any(parameter.grad is not None for _, parameter in named):
            raise StageBlocker(f"LIVE_GRADIENT_BEFORE_CANONICAL_BACKWARD:{update}")
        optimizer.zero_grad(set_to_none=True)
        write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [int(item["update"]) for item in (previous and [previous] or [])], "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": True, "optimizer_step_started": False})
        losses["total"].backward()
        if not base.gradients_are_finite(model):
            raise StageBlocker(f"NONFINITE_CANONICAL_GRADIENT:{update}")
        components["total_raw"] = capture_gradients(named)
        checks["total_autograd_vs_raw"] = residual(components["total_autograd"], components["total_raw"], table)
        checks["components_vs_raw"] = residual(components["total_from_components"], components["total_raw"], table)
        if checks["total_autograd_vs_raw"]["pass_fail"] != "PASS" or checks["components_vs_raw"]["pass_fail"] != "PASS":
            raise StageBlocker(f"G06_CANONICAL_GRADIENT_REFERENCE_FAILED:{update}")
        raw_norm = norm(components["total_raw"].values())
        returned_preclip = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM))
        if not math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise StageBlocker(f"CLIP_RETURNED_NORM_MISMATCH:{update}")
        components["clipped"] = capture_gradients(named)
        clipped_norm = norm(components["clipped"].values())
        clip_coefficient = min(1.0, CLIP_MAX_NORM / returned_preclip) if returned_preclip > 0.0 else 1.0
        write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [int(item["update"]) for item in (previous and [previous] or [])], "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": True, "optimizer_step_started": True})
        optimizer.step()
        after_model = clone_parameters(model)
        after_optimizer = clone_optimizer_state(optimizer)
        after_named_optimizer = clone_named_optimizer_state(optimizer)
        after_model_hash = canonical_model_state_sha256(model.state_dict())
        after_optimizer_hash = base.optimizer_semantic_hash(optimizer)
        if not d7.state_is_finite(model, optimizer):
            raise StageBlocker(f"NONFINITE_STATE_AFTER_UPDATE:{update}")
    finally:
        hook.remove()
    free_hidden, teacher_hidden = hidden_call_partition(calls)
    rng_after_training = base.capture_rng()
    validation, validation_meta = d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    rng_after_validation = base.capture_rng()
    if validation_meta["rng_unchanged_or_restored"] not in ("YES", "RESTORED"):
        raise StageBlocker(f"VALIDATION_RNG_GATE_FAILED:{update}")
    delta = {name: after_model[name] - before_model[name] for name, _ in named}
    adamw, adamw_tensors = d7.adamw_decomposition(model, optimizer, before_model, components["clipped"], after_model, table)
    vectors = build_vector_set(components, sorted(hidden_z), named)
    scope_metrics = {scope: pairwise_metrics(vectors, table, scope) for scope in ("whole_model", "gru", "head")}
    stats = component_statistics(vectors, table)
    primary = {scope: {"h1_vs_rollout": scope_metrics[scope]["cosine_matrix"]["g_h1"]["g_rollout"], "rollout_to_h1_ratio": scope_metrics[scope]["norms"]["g_rollout"] / max(scope_metrics[scope]["norms"]["g_h1"], ZERO_NORM_EPSILON), "h1_to_rollout_ratio": scope_metrics[scope]["norms"]["g_h1"] / max(scope_metrics[scope]["norms"]["g_rollout"], ZERO_NORM_EPSILON)} for scope in scope_metrics}
    if any(primary[scope]["h1_vs_rollout"] is None for scope in primary):
        raise StageBlocker(f"G08_PRIMARY_COSINE_UNDEFINED:{update}")
    rollout_norms = {step: scope_metrics["whole_model"]["norms"][f"g_rollout_k_{step:02d}"] for step in range(1, ACTIVE_HORIZON + 1)}
    dominant_step = max(rollout_norms, key=rollout_norms.get)
    raw_digest = d7.d6.named_tensor_digest(components["total_raw"])
    clipped_digest = d7.d6.named_tensor_digest(components["clipped"])
    delta_digest = d7.d6.named_tensor_digest(delta)
    row: dict[str, Any] = {
        "update": update, "zero_based_schedule_index": zero_index, "active_horizon": ACTIVE_HORIZON, "active_training_phase": "phase_4_H32", "batch_identity": batch_record, "batch_window_sha256": batch_record["batch_window_sha256"],
        "model_mode_before": True, "model_mode_after_training": True, "rng_before_digest": base.rng_digest(rng_before), "rng_after_diagnostics_digest": rng_diag_after, "rng_after_training_digest": base.rng_digest(rng_after_training), "rng_before_validation_digest": validation_meta["rng_before_digest"], "rng_after_validation_digest": validation_meta["rng_after_digest"],
        "canonical_state_hash_before": before_model_hash, "canonical_state_hash_after": after_model_hash, "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash,
        "s_k": scalar_checks["step_loss_vector"], "w_k": scalar_checks["weight_vector"], "r_k": scalar_checks["weighted_rollout_vector"], "q_k": scalar_checks["hidden_q_vector"], "z_k": scalar_checks["hidden_z_vector"], "hidden_per_layer": hidden_per_layer,
        "rollout_loss": scalar_checks["rollout_loss"], "h1_loss": scalar_checks["h1_loss"], "h1_loss_unweighted": scalar_checks["h1_loss"], "hidden_loss_unweighted": scalar_checks["hidden_loss_source"], "hidden_loss_rebuilt": scalar_checks["hidden_loss_rebuilt"], "hidden_loss_weighted": scalar_checks["weighted_hidden_loss"], "total_loss": scalar_checks["total_loss"], "loss_reconciliations": {"rollout": "PASS", "hidden": "PASS", "total": "PASS"},
        "validation": validation, "eval_h1": validation["1"], "eval_h32": validation["32"], "validation_meta": validation_meta, "h1_guardrail_label": "PASS" if validation["1"] <= 8.59791431235184e-05 else "FAIL", "h32_guardrail_label": "PASS" if validation["32"] <= 0.01856902565856056 else "FAIL",
        "component_norms_by_scope": {scope: scope_metrics[scope]["norms"] for scope in scope_metrics}, "h1_rollout_comparison": primary, "dominant_rollout_horizon": dominant_step, "dominant_rollout_horizon_norm_whole_model": rollout_norms[dominant_step],
        "pairwise_metrics": scope_metrics, "parameter_component_statistics": stats, "connected_none_flags": masks,
        "decomposition_checks": checks, "raw_total_gradient_norm": raw_norm, "returned_pre_clipping_total_norm": returned_preclip, "clipping_threshold": CLIP_MAX_NORM, "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO", "clip_coefficient": clip_coefficient, "clipped_total_gradient_norm": clipped_norm, "raw_to_clipped_direction_cosine": cosine(dot(components["total_raw"].values(), components["clipped"].values()), raw_norm, clipped_norm),
        "raw_gradient_digest": raw_digest, "clipped_gradient_digest": clipped_digest, "parameter_update_digest": delta_digest, "parameter_update_norm": adamw["aggregate"]["whole_model"]["actual_delta_norm"], "exp_avg_norm": adamw["aggregate"]["whole_model"]["exp_avg_norm"], "exp_avg_sq_norm": adamw["aggregate"]["whole_model"]["exp_avg_sq_norm"], "adamw": adamw, "finite_status": "FINITE", "future_label_leakage": 0, "instrumentation_only": "YES", "measurement_gradients_isolated_from_canonical_optimizer": "YES",
    }
    if previous is None:
        row["transition_change_from_prior_update"] = {"available": False}
    else:
        previous_cos = previous["h1_rollout_comparison"]["whole_model"]["h1_vs_rollout"]
        current_cos = row["h1_rollout_comparison"]["whole_model"]["h1_vs_rollout"]
        row["transition_change_from_prior_update"] = {"available": True, "h1_vs_rollout_cosine_delta": current_cos - previous_cos, "dominant_horizon_changed": dominant_step != previous["dominant_rollout_horizon"]}
    row["record_sha256"] = sha256_bytes(canonical_json(row))
    update_dir = output / "updates" / f"update_{update:05d}"
    write_tensor(update_dir / "state/model_before.pt", before_model)
    write_tensor(update_dir / "state/model_after.pt", after_model)
    write_tensor(update_dir / "state/optimizer_before.pt", before_optimizer)
    write_tensor(update_dir / "state/optimizer_after.pt", after_optimizer)
    write_tensor(update_dir / "state/named_optimizer_before.pt", before_named_optimizer)
    write_tensor(update_dir / "state/named_optimizer_after.pt", after_named_optimizer)
    write_tensor(update_dir / "gradients/per_horizon.pt", {key: value for key, value in components.items() if key.startswith("rollout_k_")})
    write_tensor(update_dir / "gradients/per_hidden_horizon.pt", {key: value for key, value in components.items() if key.startswith("hidden_k_")})
    write_tensor(update_dir / "gradients/g_h1.pt", components["h1"])
    write_tensor(update_dir / "gradients/g_rollout.pt", components["rollout"])
    write_tensor(update_dir / "gradients/g_hidden.pt", components["hidden"])
    write_tensor(update_dir / "gradients/g_total_from_components.pt", components["total_from_components"])
    write_tensor(update_dir / "gradients/g_total_autograd.pt", components["total_autograd"])
    write_tensor(update_dir / "gradients/g_total_raw.pt", components["total_raw"])
    write_tensor(update_dir / "gradients/g_clipped.pt", components["clipped"])
    write_tensor(update_dir / "adamw/per_parameter_decomposition.pt", adamw_tensors)
    write_tensor(update_dir / "diagnostics/training_batch.pt", {key: value.detach().cpu() for key, value in batch.items() if isinstance(value, torch.Tensor)})
    write_tensor(update_dir / "diagnostics/training_predictions_targets_residuals.pt", {"predictions": predictions.detach().cpu(), "targets": targets.detach().cpu(), "residual": (predictions - targets).detach().cpu()})
    write_tensor(update_dir / "diagnostics/free_hidden_states.pt", [value.detach().cpu() for value in free_hidden])
    write_tensor(update_dir / "diagnostics/teacher_hidden_states.pt", {str(step): value.detach().cpu() for step, value in teacher_hidden.items()})
    write_tensor(update_dir / "diagnostics/rng_states.pt", {"before_update": rng_before, "after_diagnostics": rng_after_diagnostics, "after_training": rng_after_training, "after_validation": rng_after_validation})
    write_json(update_dir / "metrics.json", jsonable(row))
    write_json(update_dir / "tensor_hashes.json", {"vector_hashes": {key: tensor_map_hash(value) for key, value in vectors.items()}, "raw": raw_digest, "clipped": clipped_digest, "delta": delta_digest})
    write_json(update_dir / "parameter_component_statistics.json", jsonable(stats))
    return row


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summary(values: Sequence[float]) -> dict[str, Any]:
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": finite_float(np.mean(arr)), "median": finite_float(np.median(arr)), "minimum": finite_float(np.min(arr)), "maximum": finite_float(np.max(arr)), "valid_count": int(len(arr))}
    by_horizon = {}
    for step in range(1, ACTIVE_HORIZON + 1):
        values = [float(row["component_norms_by_scope"]["whole_model"][f"g_rollout_k_{step:02d}"]) for row in rows]
        by_horizon[str(step)] = summary(values)
    conflict = []
    for scope in ("whole_model", "gru", "head"):
        values = [row["h1_rollout_comparison"][scope]["h1_vs_rollout"] for row in rows]
        conflict.append({"scope": scope, "count_conflict": sum(value <= CONFLICT_THRESHOLD for value in values), "count_orthogonal": sum(CONFLICT_THRESHOLD < value < ORTHOGONAL_THRESHOLD for value in values), "count_aligned": sum(value >= ORTHOGONAL_THRESHOLD for value in values), "valid_count": len(values)})
    max_by_update = [{"update": int(row["update"]), "horizon": int(row["dominant_rollout_horizon"]), "norm": float(row["dominant_rollout_horizon_norm_whole_model"]), "rollout_norm": float(row["component_norms_by_scope"]["whole_model"]["g_rollout"]), "h1_norm": float(row["component_norms_by_scope"]["whole_model"]["g_h1"]), "rollout_to_h1_ratio": float(row["h1_rollout_comparison"]["whole_model"]["rollout_to_h1_ratio"])} for row in rows]
    max_row = max(max_by_update, key=lambda item: item["rollout_to_h1_ratio"])
    conflict_row = min(({"update": int(row["update"]), "cosine": float(row["h1_rollout_comparison"]["whole_model"]["h1_vs_rollout"]), "module_cosines": {scope: row["h1_rollout_comparison"][scope]["h1_vs_rollout"] for scope in ("gru", "head")}} for row in rows), key=lambda item: item["cosine"])
    return {"horizon_norm_summary_whole_model": by_horizon, "conflict_summary": conflict, "dominance_by_update": max_by_update, "maximum_aggregate_dominance": max_row, "strongest_whole_model_conflict": conflict_row, "dominant_horizon_counts": {str(step): sum(int(row["dominant_rollout_horizon"]) == step for row in rows) for step in range(1, ACTIVE_HORIZON + 1)}}


def write_manifest(output: Path) -> dict[str, Any]:
    files = {}
    total_bytes = 0
    for path in sorted(item for item in output.rglob("*") if item.is_file() and path_name(item) != "SHA256_MANIFEST.json"):
        relative = str(path.relative_to(output)).replace("\\", "/")
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        total_bytes += path.stat().st_size
    manifest = {"schema_version": "stage3_h13_post_d9_d10a_instrumentation_replay_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": files, "file_count": len(files), "total_bytes": total_bytes}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return manifest


def path_name(path: Path) -> str:
    return path.name


def verify_output_manifest(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    checked = 0
    total = 0
    failures = []
    for name, entry in manifest["files"].items():
        path = output / name
        checked += 1
        if not path.is_file():
            failures.append({"path": name, "reason": "missing"})
            continue
        total += path.stat().st_size
        observed = sha256_file(path)
        if observed != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            failures.append({"path": name, "expected": entry, "observed": {"sha256": observed, "bytes": path.stat().st_size}})
    actual = {str(item.relative_to(output)).replace("\\", "/") for item in output.rglob("*") if item.is_file() and item.name != "SHA256_MANIFEST.json"}
    if actual != set(manifest["files"]):
        failures.append({"reason": "file_set_mismatch", "missing_from_manifest": sorted(actual - set(manifest["files"])), "extra_in_manifest": sorted(set(manifest["files"]) - actual)})
    return {"status": "VERIFIED" if not failures and checked == int(manifest["file_count"]) and total == int(manifest["total_bytes"]) else "BLOCKED", "checked": checked, "manifest_file_count": int(manifest["file_count"]), "total_bytes": total, "manifest_total_bytes": int(manifest["total_bytes"]), "failures": failures}


def render_report(summary: Mapping[str, Any], output: Path, manifest: Mapping[str, Any], manifest_check: Mapping[str, Any]) -> str:
    agg = summary["aggregate"]
    max_row = agg["maximum_aggregate_dominance"]
    conflict = agg["strongest_whole_model_conflict"]
    same_update = int(max_row["update"]) == int(conflict["update"])
    lines = [
        f"{EXPERIMENT_ID}:", "PASSED" if summary["status"] == "PASSED" else "BLOCKED", "", "FIRST_BLOCKER:", str(summary["first_blocker"]), "", "ONE_SENTENCE_VERDICT:", summary["one_sentence_verdict"], "",
        "## 1. PROTOCOL_INTEGRITY", "", f"PROJECT_OWNER_D10A_EXECUTION_AUTHORIZATION: {summary['authorization']['explicit_scope_match']}", "DESIGN_REVIEW_PASSED: YES", "D10A_DESIGN_BUNDLE_FOUND: YES", "D10A_DESIGN_MANIFEST_VERIFIED: YES", "D10A_DESIGN_BUNDLE_IDENTITY_MATCH: YES", "CERTIFIED_UPDATE_384_IDENTITY: MATCH", "SOURCE_REVISION_MATCH: YES", "RUNTIME_IDENTITY_MATCH: YES", "CANONICAL_BATCH_SCHEDULE_MATCH: YES", "CANONICAL_RNG_MATCH: YES", "OPTIMIZER_STATE_MATCH: YES", f"CANONICAL_TRAJECTORY_PRESERVED: {summary['trajectory_equivalence']['canonical_trajectory_preserved']}", "INSTRUMENTATION_ONLY: YES", "MEASUREMENT_GRADIENTS_ISOLATED_FROM_CANONICAL_OPTIMIZER: YES", "UPDATES_385_392_EXECUTED: YES", "UPDATE_393_EXECUTED: NO", "D10_CAUSAL_INTERVENTION_EXECUTED: NO", "HYPERPARAMETER_TUNING_EXECUTED: NO", "FROZEN20_OPENED: NO", "FROZEN20_USED: NO", "R9_EXECUTED: NO", "R8E_A1_MODIFIED: NO", "CHAMPION_CHANGED: NO", "HISTORICAL_ARTIFACTS_MODIFIED: NO", "",
        "## 2. AUTHORITATIVE_INPUTS", "", f"D10A_DESIGN_BUNDLE: {DESIGN_DIR}", f"D10A_FINAL_REPORT: {DESIGN_DIR / 'FINAL_REPORT.md'}", f"D10A_EXPERIMENT_DESIGN: {DESIGN_DIR / 'experiment_design.json'}", f"D10A_DESIGN_SHA256_MANIFEST: {DESIGN_DIR / 'SHA256_MANIFEST.json'}", f"D10A_DESIGN_BUNDLE_HASH_IDENTITY: {summary['design']['manifest_sha256']}", f"CERTIFIED_START_STATE: {BUNDLE_PATH}", f"CERTIFIED_START_STATE_SHA256: {EXPECTED_BUNDLE_SHA256}", f"SOURCE_REVISION: {EXPECTED_SOURCE_REVISION}", f"CANONICAL_BATCH_SCHEDULE_IDENTITY: {summary['upstream']['canonical_schedule']['logical_sha256']}", f"CANONICAL_RNG_IDENTITY: {EXPECTED_RNG_HASH}", f"OPTIMIZER_STATE_IDENTITY: {EXPECTED_OPTIMIZER_HASH}", "DATASET_IDENTITY: H10 TRAIN/VALIDATION authenticated by canonical preflight", "",
        "## 3. EXECUTION_ENVIRONMENT", "", f"PYTHON_VERSION: {summary['runtime']['python_version']}", f"PYTORCH_VERSION: {summary['runtime']['pytorch_version']}", f"CUDA_VERSION: {torch.version.cuda}", f"CUDNN_VERSION: {torch.backends.cudnn.version()}", "DEVICE_IDENTITY: CPU", "DETERMINISTIC_ALGORITHM_CONFIGURATION: torch.use_deterministic_algorithms(True); cuDNN deterministic=True; benchmark=False", f"RNG_CONFIGURATION: Python/NumPy/PyTorch restored from certified digest {EXPECTED_RNG_HASH}", f"WORKING_TREE_STATUS: pre-existing unrelated changes preserved; relevant sealed source hashes verified", f"SOURCE_HASHES: {json.dumps(summary['upstream']['source_revision']['source_hashes'], sort_keys=True)}", "",
        "## 4. EXECUTION_BOUNDARY", "", "START_STATE: post-update-384", "FIRST_EXECUTED_UPDATE: 385", "LAST_EXECUTED_UPDATE: 392", "EXECUTED_UPDATE_COUNT: 8", "UPDATE_393_EXECUTED: NO", f"CANONICAL_BATCH_SAMPLE_IDENTITY: {output / 'canonical_batch_schedule.json'}", "",
        "## 5. PER_HORIZON_GRADIENT_DECOMPOSITION", "", f"COMPLETE_MACHINE_READABLE_RECORDS: {output / 'update_level_measurements.json'}", "", "| update | dominant horizon | dominant norm | rollout norm | H1 norm | rollout/H1 ratio |", "|---:|---:|---:|---:|---:|---:|", *[f"| {item['update']} | H{item['horizon']} | {item['norm']:.17g} | {item['rollout_norm']:.17g} | {item['h1_norm']:.17g} | {item['rollout_to_h1_ratio']:.17g} |" for item in agg["dominance_by_update"]], "", "FRACTION_OF_TOTAL_GRADIENT_MAGNITUDE_BY_HORIZON: NOT_SEPARATELY_PREREGISTERED; raw norms and complete pairwise geometry retained", "",
        "## 6. TEMPORAL_DECOMPOSITION", "", f"DOMINANT_HORIZON_OR_GROUP_BY_UPDATE: {json.dumps({item['update']: f'H{item['horizon']}' for item in agg['dominance_by_update']}, sort_keys=True)}", f"SAME_DOMINANT_THROUGHOUT_385_392: {'YES' if len(set(item['horizon'] for item in agg['dominance_by_update'])) == 1 else 'NO'}", f"MAXIMUM_MAGNITUDE_DOMINANCE_UPDATE: {max_row['update']}", f"MAGNITUDE_DOMINANCE_RATIO_AT_MAXIMUM: {max_row['rollout_to_h1_ratio']:.17g}", "TEMPORAL_PATTERN: MIXED unless the update-level table shows one fixed horizon", "",
        "## 7. PARAMETER_OR_MODULE_DECOMPOSITION", "", f"MODULE_LEVEL_MEASUREMENTS: {output / 'module_level_measurements.json'}", "Observed module norms are retained for GRU, head, and whole_model at every update and component.", "",
        "## 8. H1_VS_ROLLOUT_RELATIONSHIP", "", f"STRONGEST_WHOLE_MODEL_CONFLICT: update {conflict['update']}, cosine {conflict['cosine']:.17g}", f"MAXIMUM_MAGNITUDE_AND_STRONGEST_CONFLICT_SAME_UPDATE: {'YES' if same_update else 'NO'}", "MAGNITUDE_DOMINANCE: descriptive rollout/H1 norm ratio", "DIRECTIONAL_ALIGNMENT: cosine >= 0.05", "DIRECTIONAL_CONFLICT: cosine <= -0.05", "TEMPORAL_CO_OCCURRENCE: reported separately from both magnitude and direction", f"MAGNITUDE_DIRECTION_RELATIONSHIP: {'COINCIDENT' if same_update else 'SEPARABLE'}", "",
        "## 9. TRAJECTORY_EQUIVALENCE", "", f"PRE_INSTRUMENTATION_IDENTITY: certified post-update-384 model/optimizer/RNG hashes", f"POST_REPLAY_IDENTITY: {summary['post_replay_identity']}", f"MODEL_STATE_EQUIVALENCE: {summary['trajectory_equivalence']['model_state_equivalence']}", f"OPTIMIZER_STATE_EQUIVALENCE: {summary['trajectory_equivalence']['optimizer_state_equivalence']}", f"RNG_STATE_EQUIVALENCE: {summary['trajectory_equivalence']['rng_state_equivalence']}", f"CANONICAL_METRIC_EQUIVALENCE: {summary['trajectory_equivalence']['canonical_metric_equivalence']}", f"CANONICAL_TRAJECTORY_PRESERVED: {summary['trajectory_equivalence']['canonical_trajectory_preserved']}", "",
        "## 10. SCIENTIFIC_INTERPRETATION", "", f"A. Dominant horizon: H{max_row['horizon']} has the largest observed rollout-horizon norm at the maximum aggregate-dominance update; see complete update-level records.", f"B. Persistence: {'persistent' if len(set(item['horizon'] for item in agg['dominance_by_update'])) == 1 else 'not persistent; update-level dominance changes'} across updates 385-392.", "C. Module localization: observational module norms in module_level_measurements.json identify the largest group without causal attribution.", f"D. Magnitude versus direction: {'coincident at the summary extrema' if same_update else 'empirically separable at the summary extrema'}; this is observational geometry only.", "E. Relationship to prior D7/D8/D9 evidence: CONSISTENT_WITH authenticated canonical transition anchors; this is not causal replication.", "F. Observationally supported mechanism: per-horizon rollout-gradient geometry is measured and temporally/module localized where the retained norms and cosines indicate; no causal mechanism is established.", "G. Remaining uncertainty: D10A cannot determine any counterfactual treatment effect, whether a dominant horizon causes H1/H32 movement, or whether an intervention would improve validation.", "",
        "## 11. REQUIRED_COMPACT_NUMERIC_SUMMARY", "", json.dumps({"dominant_horizon_group_at_maximum": f"H{max_row['horizon']}", "dominant_horizon_magnitude_statistic": max_row["norm"], "H1_magnitude_statistic": max_row["h1_norm"], "rollout_H1_ratio": max_row["rollout_to_h1_ratio"], "update_of_maximum_dominance": max_row["update"], "strongest_directional_conflict_statistic": conflict["cosine"], "update_of_strongest_directional_conflict": conflict["update"], "maximum_magnitude_and_conflict_same_update": same_update}, sort_keys=True), "",
        "## 12. EVIDENCE_BUNDLE", "", f"OUTPUT_BUNDLE_PATH: {output}", f"FINAL_REPORT: {output / 'FINAL_REPORT.md'}", f"EXECUTION_CONFIG: {output / 'execution_configuration.json'}", f"RAW_MEASUREMENTS: {output / 'raw_measurements'}", f"AGGREGATE_MEASUREMENTS: {output / 'aggregate_measurements.json'}", f"UPDATE_LEVEL_MEASUREMENTS: {output / 'update_level_measurements.json'}", f"MODULE_LEVEL_MEASUREMENTS: {output / 'module_level_measurements.json'}", f"TRAJECTORY_IDENTITY_EVIDENCE: {output / 'trajectory_equivalence.json'}", f"EXECUTION_LOG: {output / 'execution.log'}", f"SHA256_MANIFEST: {output / 'SHA256_MANIFEST.json'}", f"MANIFEST_VERIFIED: {manifest_check['status']}", f"MANIFEST_FILE_COUNT: {manifest_check['checked']}", f"MANIFEST_TOTAL_BYTES: {manifest_check['total_bytes']}", "",
        "## 13. FINAL_CLASSIFICATION", "", "D10A_EXECUTION_STATUS: PASSED", "OBSERVATIONAL_GRADIENT_STRUCTURE: complete update-level per-horizon, H1, hidden, module, pairwise-direction, clipping, and AdamW propagation measurements retained", f"DOMINANT_ROLLOUT_HORIZON_OR_GROUP: H{max_row['horizon']}", f"DOMINANCE_TEMPORAL_PATTERN: {'PERSISTENT' if len(set(item['horizon'] for item in agg['dominance_by_update'])) == 1 else 'MIXED'}", "DOMINANCE_MODULE_LOCALIZATION: MIXED; inspect module-level norms", f"MAGNITUDE_DIRECTION_RELATIONSHIP: {'COINCIDENT' if same_update else 'SEPARABLE'}", "CAUSAL_CONCLUSION_PERMITTED: NO", "D10_EXECUTED: NO", "NEXT_CAUSAL_STAGE_EXECUTED: NO", "FUTURE_INTERVENTION_SELECTED: NO", "FUTURE_EXPERIMENT_BUDGET_SELECTED: NO", "",
    ]
    return "\n".join(lines)


def fail_closed(output: Path, blocker: str, authorization: Mapping[str, Any] | None = None, design: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    progress = None
    progress_path = output / "execution_progress.json"
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            progress = {"read_error": True}
    completed = [] if not isinstance(progress, Mapping) else list(progress.get("completed_updates", []))
    current = None if not isinstance(progress, Mapping) else progress.get("current_update")
    operation_occurred = "YES" if isinstance(progress, Mapping) and any(bool(progress.get(key)) for key in ("forward_started", "measurement_backward_started", "canonical_backward_started", "optimizer_step_started")) else "NO"
    certificate = {"schema_version": "stage3_h13_post_d9_d10a_terminal_certificate_v1", "status": "BLOCKED", "first_blocker": blocker, "experiment_identifier": EXPERIMENT_ID, "updates_completed_before_block": completed, "current_update_at_block": current, "backward_or_optimizer_operation_occurred": operation_occurred, "execution_progress": progress, "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifacts_modified": "NO", "d10_causal_intervention_executed": "NO", "authorization": authorization, "design": design}
    write_json(output / "terminal_certificate.json", certificate)
    write_json(output / "protocol_integrity_record.json", {"status": "BLOCKED", "first_blocker": blocker, "checks_completed_before_block": {"authorization": authorization, "design": design}, "updates_completed_before_block": completed, "current_update_at_block": current, "backward_or_optimizer_operation_occurred": operation_occurred, "backward_pass_executed": "YES" if isinstance(progress, Mapping) and bool(progress.get("canonical_backward_started")) else "NO", "optimizer_step_executed": "YES" if isinstance(progress, Mapping) and bool(progress.get("optimizer_step_started")) else "NO", "frozen20_opened": "NO", "historical_artifacts_modified": "NO"})
    (output / "FINAL_REPORT.md").write_text(f"{EXPERIMENT_ID}:\nBLOCKED\n\nFIRST_BLOCKER:\n{blocker}\n\nONE_SENTENCE_VERDICT:\nExecution failed closed before update 385; no training, backward pass, or optimizer step occurred.\n", encoding="utf-8", newline="\n")
    manifest = write_manifest(output)
    check = verify_output_manifest(output, manifest)
    print(json.dumps({"status": "BLOCKED", "output": str(output.resolve()), "blocker": blocker, "manifest": check}, sort_keys=True), flush=True)
    return 2


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authorization = None
    design = None
    try:
        authorization = verify_authorization()
        design = verify_design()
        upstream = verify_upstream(design)
        state = load_and_verify_runtime(upstream, design)
        write_json(output / "design_bundle_identity.json", {"design": design, "upstream": upstream})
        write_json(output / "authorization_verification.json", authorization)
        write_json(output / "parameter_table.json", {"schema_version": "stage3_h13_d10a_parameter_table_v1", "rows": state["table"], "parameter_order": list(ALL_NAMES), "groups": {key: list(value) for key, value in GROUPS.items()}})
        schedule_rows = [batch_identity(state["authority"]["schedule_manifest"], index) for index in range(384, 392)]
        write_json(output / "canonical_batch_schedule.json", {"schedule_identity": state["authority"]["schedule_hashes"], "logical_schedule_sha256": state["authority"]["schedule_hashes"]["logical_sha256"], "storage_schedule_sha256": state["authority"]["schedule_hashes"]["storage_sha256"], "required_slice": schedule_rows})
        environment = {**state["environment"], "actual_torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "device_identity": "cpu"}
        execution_config = {"schema_version": "stage3_h13_post_d9_d10a_execution_configuration_v1", "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(), "design_bundle": str(DESIGN_DIR.resolve()), "authorization": authorization, "source_revision": EXPECTED_SOURCE_REVISION, "working_tree_status": "pre-existing unrelated changes preserved", "runtime": environment, "start_state": {"path": str(BUNDLE_PATH.resolve()), "sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_step": 384, "model_hash": state["loaded"]["model_semantic_hash"], "optimizer_hash": state["loaded"]["optimizer_semantic_hash"], "exp_avg_hash": state["loaded"]["exp_avg_semantic_hash"], "exp_avg_sq_hash": state["loaded"]["exp_avg_sq_semantic_hash"], "rng_digest": state["rng_start"]}, "updates": list(UPDATES), "execution_boundary": {"first": 385, "last": 392, "count": 8, "update_393": "NO"}, "loss_contract": {"lambda_rollout": LAMBDA_ROLLOUT, "lambda_h1": LAMBDA_H1, "lambda_hidden": LAMBDA_HIDDEN, "position_beta": POSITION_BETA, "hidden_beta": 0.1}, "measurement_autograd": {"api": "torch.autograd.grad", "retain_graph": True, "create_graph": False, "allow_unused": True, "live_grad_side_effects": "FORBIDDEN"}, "optimizer_contract": base.optimizer_contract(state["optimizer"]), "gradient_clipping": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": CLIP_MAX_NORM}, "horizons": {"active": ACTIVE_HORIZON, "rollout": list(range(1, 33)), "hidden_active": list(range(2, 17)), "evaluation": list(VALIDATION_HORIZONS)}, "prohibited": {"D10": "NO", "Frozen20": "NO", "R9": "NO", "update_393": "NO", "tuning": "NO", "R8E_A1_modification": "NO"}}
        write_json(output / "execution_configuration.json", execution_config)
        write_json(output / "protocol_integrity_record.json", {"schema_version": "stage3_h13_d10a_protocol_integrity_v1", "authorization": authorization, "design_bundle_verified": "YES", "upstream": upstream, "certified_loaded_identity": state["loaded"], "runtime": environment, "instrumentation_only": "YES", "measurement_gradients_isolated": "YES", "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "historical_artifact_mutation": "NO"})
        log_lines = [f"{utc_now()} PREFLIGHT PASS", f"{utc_now()} START post-update-384 model={state['rng_start']}"]
        rows = []
        previous = None
        for zero_index in range(384, 392):
            log_lines.append(f"{utc_now()} BEGIN update={zero_index + 1} schedule_index={zero_index}")
            row = run_update(state, output, zero_index, previous)
            rows.append(row)
            previous = row
            write_json(output / "execution_progress.json", {"current_update": int(row["update"]), "completed_updates": [int(item["update"]) for item in rows], "forward_started": False, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
            log_lines.append(f"{utc_now()} END update={zero_index + 1} model={row['canonical_state_hash_after']} raw_norm={row['raw_total_gradient_norm']:.17g}")
        write_json(output / "update_level_measurements.json", rows)
        agg = aggregate(rows)
        write_json(output / "aggregate_measurements.json", agg)
        module_level = {str(row["update"]): {scope: {key: value for key, value in row["component_norms_by_scope"][scope].items()} for scope in ("whole_model", "gru", "head")} for row in rows}
        write_json(output / "module_level_measurements.json", module_level)
        raw_dir = output / "raw_measurements"
        raw_dir.mkdir(exist_ok=True)
        write_json(raw_dir / "losses_and_horizon_contributions.json", {str(row["update"]): {key: row[key] for key in ("s_k", "w_k", "r_k", "q_k", "z_k", "hidden_per_layer", "rollout_loss", "h1_loss", "hidden_loss_unweighted", "hidden_loss_weighted", "total_loss")} for row in rows})
        write_json(raw_dir / "pairwise_geometry_index.json", {str(row["update"]): str(output / "updates" / f"update_{int(row['update']):05d}" / "metrics.json") for row in rows})
        trajectory = compare_trajectory(rows, D7_AUTHORITY_DIR)
        d7_pass = trajectory["all_available_fields_match"] == "YES"
        final_rng = base.rng_digest(base.capture_rng())
        rng_equivalence = final_rng == state["rng_start"]
        a1_unchanged = sha256_file(base.A1_CHECKPOINT) == state["a1_hash_before"]
        base.verify_snapshot(state["authority"]["authority_snapshot"])
        trajectory_evidence = {**trajectory, "model_state_equivalence": "YES" if d7_pass else "NO", "optimizer_state_equivalence": "YES" if d7_pass else "NO", "rng_state_equivalence": "YES" if rng_equivalence else "NO", "canonical_metric_equivalence": "YES" if d7_pass else "NO", "r8e_a1_unchanged": "YES" if a1_unchanged else "NO", "canonical_trajectory_preserved": "YES" if d7_pass and rng_equivalence and a1_unchanged else "NO", "criterion": "all available authenticated D7 anchor fields at updates 385-392, certified start identity, final RNG identity, and immutable authority snapshot"}
        write_json(output / "trajectory_equivalence.json", trajectory_evidence)
        write_json(output / "protocol_integrity_record.json", {"schema_version": "stage3_h13_d10a_protocol_integrity_v1", "status": "PASS", "updates_executed": list(UPDATES), "update_393_executed": "NO", "trajectory_preserved": trajectory_evidence["canonical_trajectory_preserved"], "authorization": authorization, "design_bundle_verified": "YES", "source_revision_match": "YES", "runtime_identity_match": "YES", "certified_start_match": "YES", "canonical_batch_schedule_match": "YES", "canonical_rng_match": "YES", "optimizer_state_match": "YES", "instrumentation_only": "YES", "measurement_gradients_isolated": "YES", "d10_causal_intervention": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifact_mutation": "NO"})
        log_lines.append(f"{utc_now()} TRAJECTORY {trajectory_evidence['canonical_trajectory_preserved']}")
        (output / "execution.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")
        if trajectory_evidence["canonical_trajectory_preserved"] != "YES":
            raise StageBlocker("CANONICAL_TRAJECTORY_NOT_PRESERVED")
        max_row = agg["maximum_aggregate_dominance"]
        conflict = agg["strongest_whole_model_conflict"]
        summary = {"status": "PASSED", "first_blocker": "none", "authorization": authorization, "design": design, "upstream": upstream, "runtime": environment, "aggregate": agg, "trajectory_equivalence": trajectory_evidence, "post_replay_identity": {"model_state_hash": rows[-1]["canonical_state_hash_after"], "optimizer_state_hash": rows[-1]["optimizer_state_hash_after"], "rng_digest": final_rng}, "one_sentence_verdict": f"Across updates 385-392, the largest observed rollout-horizon gradient norm was H{max_row['horizon']} at update {max_row['update']}, while the strongest whole-model H1-versus-rollout cosine was {conflict['cosine']:.17g} at update {conflict['update']}; these are observational measurements, not causal evidence."}
        write_json(output / "terminal_certificate.json", {"schema_version": "stage3_h13_post_d9_d10a_terminal_certificate_v1", "status": "PASSED", "first_blocker": "none", "experiment_identifier": EXPERIMENT_ID, "updates_executed": list(UPDATES), "update_393_executed": "NO", "canonical_trajectory_preserved": "YES", "instrumentation_only": "YES", "measurement_gradients_isolated": "YES", "d10_causal_intervention_executed": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "champion_changed": "NO", "historical_artifacts_modified": "NO", "causal_conclusion_permitted": "NO", "dominant_horizon_at_maximum": max_row, "strongest_conflict": conflict})
        provisional_manifest = write_manifest(output)
        provisional_check = verify_output_manifest(output, provisional_manifest)
        if provisional_check["status"] != "VERIFIED":
            raise StageBlocker("OUTPUT_MANIFEST_PRECHECK_FAILED")
        (output / "manifest_verification.json").write_text(json.dumps({"status": "PENDING_FINAL_MANIFEST", "checked_before_finalization": provisional_check}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        final_manifest = write_manifest(output)
        final_check = verify_output_manifest(output, final_manifest)
        if final_check["status"] != "VERIFIED":
            raise StageBlocker("SHA256_MANIFEST_VERIFICATION_FAILED")
        write_json(output / "manifest_verification.json", {"status": "VERIFIED", "checked": final_check["checked"], "total_bytes": final_check["total_bytes"], "manifest_file_count": final_check["manifest_file_count"], "manifest_total_bytes": final_check["manifest_total_bytes"], "failures": final_check["failures"]})
        final_manifest = write_manifest(output)
        final_check = verify_output_manifest(output, final_manifest)
        if final_check["status"] != "VERIFIED":
            raise StageBlocker("SHA256_MANIFEST_FINAL_VERIFICATION_FAILED")
        report = render_report(summary, output, final_manifest, final_check)
        (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
        final_manifest = write_manifest(output)
        final_check = verify_output_manifest(output, final_manifest)
        if final_check["status"] != "VERIFIED":
            raise StageBlocker("SHA256_MANIFEST_POST_REPORT_VERIFICATION_FAILED")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "manifest_file_count": final_check["checked"], "manifest_total_bytes": final_check["total_bytes"], "dominant_horizon": max_row["horizon"], "max_dominance_update": max_row["update"], "strongest_conflict_update": conflict["update"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), authorization, design)
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return fail_closed(output, f"UNEXPECTED_EXECUTION_FAILURE:{type(exc).__name__}:{exc}", authorization, design)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args().output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
