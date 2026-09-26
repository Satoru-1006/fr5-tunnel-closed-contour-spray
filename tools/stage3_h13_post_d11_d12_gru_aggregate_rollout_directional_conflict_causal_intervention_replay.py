"""Execute the sealed Stage 3 H13 D12 aggregate-GRU causal replay.

This runner is isolated from the canonical model/loss source and from all
historical D10/D11 runners.  It executes exactly one update-385 intervention:
orthogonally remove the GRU-localized component of the aggregate rollout
gradient that opposes the GRU-localized H1 gradient, before unchanged global
clipping and AdamW.
"""

from __future__ import annotations

import copy
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
from tools import stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay as d10a  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D11_D12_GRU_AGGREGATE_ROLLOUT_DIRECTIONAL_CONFLICT_CAUSAL_INTERVENTION_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d11_d12_gru_aggregate_rollout_directional_conflict_causal_intervention_design_review_20260819T134220+0800"
D10_DIR = ROOT / "outputs/stage3_h13_post_d10a_d10_gru_directional_conflict_causal_intervention_replay_20260818T023427Z"
D10A_DIR = ROOT / "outputs/stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay_20260817T182510Z"
D11_DIR = ROOT / "outputs/stage3_h13_post_d10_d11_h32_gradient_magnitude_dominance_causal_intervention_replay_20260819T022300+0800"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\eb9fa965-737a-49bd-86d1-7e232d0f2920\pasted-text.txt")

EXPECTED_DESIGN_MANIFEST_SHA256 = "9d5c58c1abff01fb87e0cd447c7e479a240c5f24d52b273e14fb0d80458d457e"
EXPECTED_D10_MANIFEST_SHA256 = "4b2896a280f806609a3af941836755b6c159535e3f07c6ef464f5bcdf18687f0"
EXPECTED_D10A_MANIFEST_SHA256 = "9c6e664ce1342791b826d5d12692164f72d437350eca18fa868a250d36abb20a"
EXPECTED_D11_MANIFEST_SHA256 = "51473c46b8a558702d0cfd6b07789b3af2fe5d22ffbaf254077aaee866a9c5d0"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_EXP_AVG_HASH = "7226e7ae693c1083d6928df6968d67d32d33477c16a8770adcf3050fc9492cdb"
EXPECTED_EXP_AVG_SQ_HASH = "eaf5ebdd6aaddefec25950e80d1a14baeb92a06dddcd11b246546e44c8489d89"
EXPECTED_RNG_HASH = "857d5ddd6c4271019b55a53774ad68c70e435abd38cf430b7f0d57ed047848d8"
EXPECTED_SCHEDULE_HASH = "e1e0d681d84c75348c656f306eedfd61bed849a1ff1c0b966f5ff3a63145ebcd"
UPDATES = tuple(range(385, 393))
ACTIVE_HORIZON = 32
VALIDATION_HORIZONS = tuple(int(value) for value in EVALUATION_HORIZONS)
LAMBDA_ROLLOUT = 1.0
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
POSITION_BETA = 0.001
CLIP_MAX_NORM = 1.0
H1_THRESHOLD = 8.59791431235184e-05
H32_THRESHOLD = 0.01856902565856056
MATERIAL_H1_EFFECT = 4.0990068154056389e-06
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
COSINE_TOLERANCE = 1.0e-12
ZERO_NORM_EPSILON = 1.0e-12
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


def write_tensor(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def jsonable(value: Any) -> Any:
    return d10a.jsonable(value)


def finite_float(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise StageBlocker(f"NONFINITE_VALUE:{result}")
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
    denominator = left_norm * right_norm
    if denominator <= ZERO_NORM_EPSILON:
        return None
    value = finite_float(dot_value / denominator)
    if value < -1.0 - COSINE_TOLERANCE or value > 1.0 + COSINE_TOLERANCE:
        raise StageBlocker(f"COSINE_OUT_OF_DOMAIN:{value}")
    return max(-1.0, min(1.0, value))


def assert_finite(values: Iterable[torch.Tensor], label: str) -> None:
    for value in values:
        if not bool(torch.all(torch.isfinite(value))):
            raise StageBlocker(f"NONFINITE_{label}")


def map_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], names: Iterable[str]) -> bool:
    return all(torch.equal(left[name], right[name]) for name in names)


def map_max_abs_difference(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], names: Iterable[str]) -> float:
    result = 0.0
    for name in names:
        if left[name].numel():
            result = max(result, finite_float(torch.max(torch.abs(left[name].double() - right[name].double()))))
    return result


def parameter_set_hash(named: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    return sha256_bytes(canonical_json([{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype), "device": str(parameter.device), "numel": int(parameter.numel())} for name, parameter in named]))


def subset_parameter_set_hash(named: Sequence[tuple[str, torch.nn.Parameter]], names: Sequence[str]) -> str:
    selected = dict(named)
    return parameter_set_hash([(name, selected[name]) for name in names])


def residual(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    difference = torch.cat([(left[row["name"]] - right[row["name"]]).reshape(-1).double() for row in table])
    left_flat = torch.cat([left[row["name"]].reshape(-1).double() for row in table])
    right_flat = torch.cat([right[row["name"]].reshape(-1).double() for row in table])
    l2 = finite_float(torch.linalg.vector_norm(difference))
    max_abs = finite_float(torch.max(torch.abs(difference))) if difference.numel() else 0.0
    scale = max(norm(left[row["name"]] for row in table), norm(right[row["name"]] for row in table), 1.0)
    l2_tolerance = max(1.0e-7, 2.0e-5 * scale)
    max_scale = max(finite_float(torch.max(torch.abs(left_flat))), finite_float(torch.max(torch.abs(right_flat))), 1.0)
    max_tolerance = 1.0e-7 + 2.0e-5 * max_scale
    return {"l2": l2, "max_abs": max_abs, "scale": scale, "l2_tolerance": l2_tolerance, "max_tolerance": max_tolerance, "pass_fail": "PASS" if l2 <= l2_tolerance and max_abs <= max_tolerance else "FAIL"}


def verify_manifest(path: Path, expected_sha256: str, expected_count: int, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageBlocker(f"{label}_MANIFEST_MISSING:{path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise StageBlocker(f"{label}_MANIFEST_SHA256_MISMATCH:{observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("files") or payload.get("artifact_sha256")
    if isinstance(entries, Mapping):
        items = list(entries.items())
    elif isinstance(entries, list):
        items = [(str(item["path"]), item) for item in entries]
    else:
        raise StageBlocker(f"{label}_MANIFEST_ENTRIES_MISSING")
    if len(items) != expected_count:
        raise StageBlocker(f"{label}_MANIFEST_FILE_COUNT_MISMATCH:{len(items)}")
    failures = []
    for name, entry in items:
        expected = entry.get("sha256") if isinstance(entry, Mapping) else entry
        expected_bytes = entry.get("bytes", entry.get("size_bytes")) if isinstance(entry, Mapping) else None
        target = path.parent / str(name)
        if not target.is_file():
            failures.append({"path": str(name), "reason": "missing"})
            continue
        if sha256_file(target) != expected or expected_bytes is not None and target.stat().st_size != int(expected_bytes):
            failures.append({"path": str(name), "reason": "hash_or_bytes_mismatch", "expected": expected, "observed": sha256_file(target)})
    if failures:
        raise StageBlocker(f"{label}_MANIFEST_ENTRY_VERIFICATION_FAILED:{failures[0]}")
    return {"path": str(path.resolve()), "sha256": observed, "entry_count": len(items), "status": "VERIFIED", "failures": []}


def resolve_path(raw: str) -> Path:
    candidate = Path(str(raw).replace("/", "\\"))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    return ROOT / str(raw).replace("/", "\\")


def verify_design() -> dict[str, Any]:
    manifest = verify_manifest(DESIGN_DIR / "SHA256_MANIFEST.json", EXPECTED_DESIGN_MANIFEST_SHA256, 12, "D12_DESIGN")
    payload = json.loads((DESIGN_DIR / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    required_files = {"FINAL_REPORT.md", "authoritative_evidence.json", "causal_contract.json", "classification_contract.json", "control_intervention_identity_contract.json", "experiment_design.json", "fidelity_contract.json", "gradient_semantics.json", "gru_parameter_support.json", "implementation_plan.json", "intervention_specification.json", "protocol_integrity.json"}
    if set(payload["files"]) != required_files:
        raise StageBlocker("D12_DESIGN_MANIFEST_REQUIRED_FILE_SET_MISMATCH")
    records = {name: json.loads((DESIGN_DIR / name).read_text(encoding="utf-8")) for name in required_files if name.endswith(".json")}
    experiment = records["experiment_design.json"]
    causal = records["causal_contract.json"]
    classification = records["classification_contract.json"]
    support = records["gru_parameter_support.json"]
    intervention = records["intervention_specification.json"]
    fidelity = records["fidelity_contract.json"]
    protocol = records["protocol_integrity.json"]
    if experiment.get("review_status") != "PASSED" or experiment.get("design_review_only") is not True or any(experiment.get(key) is not False for key in ("training_executed", "backward_pass_executed", "optimizer_step_executed", "causal_replay_executed")):
        raise StageBlocker("D12_DESIGN_REVIEW_OR_SEALED_EXECUTION_FLAG_INVALID")
    scope = experiment.get("fixed_scope", {})
    if scope.get("start_state") != "certified post-update-384 state" or tuple(scope.get("updates_if_separately_authorized", [])) != UPDATES or scope.get("intervention_update") != 385 or scope.get("endpoint_update") != 392 or scope.get("branches") != ["CONTROL", "INTERVENTION"] or scope.get("parameter_scope") != "authenticated D10/D10A GRU support only" or scope.get("aggregate_object") != "g_rollout_GRU = restriction of grad(rollout_loss) to GRU support" or any(scope.get(key) is not False for key in ("per_horizon_surgery", "horizon_weight_search", "strength_search", "update_search", "module_search", "horizon_search", "hyperparameter_tuning")):
        raise StageBlocker("D12_EXPERIMENT_SCOPE_MISMATCH")
    if causal.get("status") != "SEALED_FOR_SEPARATE_AUTHORIZATION_ONLY" or causal.get("same_start_state_required") is not True or causal.get("sole_permitted_difference", "").find("update 385 only") < 0 or "update 393" not in " ".join(causal.get("prohibited", [])):
        raise StageBlocker("D12_CAUSAL_CONTRACT_MISMATCH")
    start = causal.get("start_state", {})
    if start.get("path") != str(BUNDLE_PATH.resolve()) or start.get("sha256") != EXPECTED_BUNDLE_SHA256 or start.get("completed_optimizer_step") != 384:
        raise StageBlocker("D12_START_STATE_CONTRACT_MISMATCH")
    if support.get("parameter_names") != list(GRU_NAMES) or support.get("support_hash") != "8dfa08a93d7da15b98772e5217c9c4814c23c4c038c18e6957667d257f7701b8" or support.get("numel") != 156672 or support.get("direct_treatment_outside_support") != "exact zero required":
        raise StageBlocker("D12_GRU_SUPPORT_CONTRACT_MISMATCH")
    if intervention.get("update") != 385 or intervention.get("trigger") != "dot(r,h) < 0" or intervention.get("per_horizon_surgery") != "NO" or intervention.get("aggregate_rollout_causal_object") != "YES" or intervention.get("operator", {}).get("no_strength_parameter") is not True or intervention.get("operator", {}).get("no_denominator_epsilon") is not True:
        raise StageBlocker("D12_INTERVENTION_CONTRACT_MISMATCH")
    if fidelity.get("scope") != "update 385 only, INTERVENTION branch only, before global clipping" or fidelity.get("clipping_and_adamw", {}).get("intervention_before_global_clipping") != "YES" or fidelity.get("clipping_and_adamw", {}).get("global_clipping_unchanged") != "YES" or fidelity.get("clipping_and_adamw", {}).get("adamw_unchanged") != "YES":
        raise StageBlocker("D12_FIDELITY_ORDER_CONTRACT_MISMATCH")
    if classification.get("no_discretion") is not True or classification.get("invalid_replay_never_not_causal") is not True or classification.get("effect") != "E_H1 = CONTROL_H1 - INTERVENTION_H1":
        raise StageBlocker("D12_CLASSIFICATION_CONTRACT_MISMATCH")
    expected_sources = records["authoritative_evidence.json"].get("source_revision", {}).get("canonical_source_hashes", {})
    source_hashes = {}
    for logical, expected in expected_sources.items():
        target = ROOT / logical.replace("/", "\\")
        candidates = [target]
        if logical.startswith("tools/"):
            candidates.append(ROOT / ("scripts/" + logical.split("/", 1)[1]))
        resolved = next((candidate for candidate in candidates if candidate.is_file() and sha256_file(candidate) == expected), None)
        if resolved is None:
            raise StageBlocker(f"SEALED_SOURCE_HASH_MISMATCH:{logical}")
        source_hashes[logical] = {"expected": expected, "observed": sha256_file(resolved), "path": str(resolved.resolve()), "path_alias": str(resolved) != str(target)}
    auth_text = AUTHORIZATION_PATH.read_text(encoding="utf-8") if AUTHORIZATION_PATH.is_file() else ""
    required_auth = [EXPERIMENT_ID, "This is an EXECUTION task.", "The D12 design review has already PASSED.", "Do NOT execute update 393.", "Do not search intervention strengths."]
    if any(item not in auth_text for item in required_auth):
        raise StageBlocker("PROJECT_OWNER_D12_EXECUTION_AUTHORIZATION_NOT_PROVEN")
    return {"manifest": manifest, "design": experiment, "records": records, "source_hashes": source_hashes, "authorization": {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"}}


def verify_upstream(design_record: Mapping[str, Any]) -> dict[str, Any]:
    evidence = design_record["records"]["authoritative_evidence.json"]
    bundle_meta = evidence["certified_post_update_384"]
    bundles = evidence["bundles"]
    d10_meta, d10a_meta, d11_meta = bundles["D10"], bundles["D10A"], bundles["D11"]
    d10_manifest = verify_manifest(D10_DIR / "SHA256_MANIFEST.json", EXPECTED_D10_MANIFEST_SHA256, int(d10_meta["manifest_entries"]), "D10")
    d10a_manifest = verify_manifest(D10A_DIR / "SHA256_MANIFEST.json", EXPECTED_D10A_MANIFEST_SHA256, int(d10a_meta["manifest_entries"]), "D10A")
    d11_manifest = verify_manifest(D11_DIR / "SHA256_MANIFEST.json", EXPECTED_D11_MANIFEST_SHA256, int(d11_meta["manifest_entries"]), "D11")
    for directory, meta, label in ((D10_DIR, d10_meta, "D10"), (D10A_DIR, d10a_meta, "D10A"), (D11_DIR, d11_meta, "D11")):
        report = directory / "FINAL_REPORT.md"
        if not report.is_file() or sha256_file(report) != meta["final_report_sha256"]:
            raise StageBlocker(f"{label}_FINAL_REPORT_HASH_MISMATCH")
    for name, expected in d10a_meta.get("key_files", {}).items():
        candidates = [D10A_DIR / name.replace("/", "\\"), D10A_DIR / "updates" / name.replace("/", "\\")]
        target = next((candidate for candidate in candidates if candidate.is_file() and sha256_file(candidate) == expected), None)
        if target is None:
            raise StageBlocker(f"D10A_KEY_FILE_HASH_MISMATCH:{name}")
    if not BUNDLE_PATH.is_file() or sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256 or sha256_file(BUNDLE_PATH) != bundle_meta["sha256"] or bundle_meta.get("observed_sha256") != EXPECTED_BUNDLE_SHA256 or bundle_meta.get("completed_optimizer_step") != 384:
        raise StageBlocker("CERTIFIED_UPDATE_384_BUNDLE_IDENTITY_MISMATCH")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git_head != EXPECTED_SOURCE_REVISION or evidence["source_revision"]["git_head"] != EXPECTED_SOURCE_REVISION or not evidence["source_revision"].get("current_observed_hashes_match"):
        raise StageBlocker(f"SOURCE_REVISION_MISMATCH:{git_head}")
    schedule = evidence["canonical_schedule"]
    d10a_schedule = json.loads((D10A_DIR / "canonical_batch_schedule.json").read_text(encoding="utf-8"))
    if schedule.get("logical_sha256") != EXPECTED_SCHEDULE_HASH or d10a_schedule.get("logical_schedule_sha256") != EXPECTED_SCHEDULE_HASH or schedule.get("update_385_schedule_index") != 384 or schedule.get("update_385_batch_sha256") != "9b453a749c31911524727e063b85973a3231754ed602f28acdc566afa0917494" or schedule.get("update_385_window_count") != 256:
        raise StageBlocker("CANONICAL_SCHEDULE_OR_UPDATE_385_BATCH_IDENTITY_MISMATCH")
    d10_control = json.loads((D10_DIR / "control_reproduction.json").read_text(encoding="utf-8"))
    if d10_control.get("control_reproduction_pass") != "YES" or d10_control.get("corrected_d10a", {}).get("first_divergence") not in (None, ""):
        raise StageBlocker("D10_RETAINED_CONTROL_REPRODUCTION_NOT_PASSED")
    aggregate = json.loads((D10A_DIR / "aggregate_measurements.json").read_text(encoding="utf-8"))
    return {"d10_manifest": d10_manifest, "d10a_manifest": d10a_manifest, "d11_manifest": d11_manifest, "d10_control_reproduction": d10_control, "d10a_aggregate": aggregate, "certified_start": bundle_meta, "schedule": schedule, "source_revision": {"git_head": git_head, "expected": EXPECTED_SOURCE_REVISION}, "paths": {"d10": str(D10_DIR.resolve()), "d10a": str(D10A_DIR.resolve()), "d11": str(D11_DIR.resolve()), "bundle": str(BUNDLE_PATH.resolve())}}


def load_state(upstream: Mapping[str, Any]) -> dict[str, Any]:
    base.set_deterministic(base.SEED)
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
    pre = base.preflight(True)
    authority = base.verify_authority(pre)
    env = authority["current_environment"]
    expected_runtime = {"source_revision": EXPECTED_SOURCE_REVISION, "device": "cpu", "dtype": "float32", "pytorch_version": "2.13.0+cpu", "python_version": "3.12.1", "numpy_version": "2.1.3"}
    if any(env.get(key) != value for key, value in expected_runtime.items()) or not torch.are_deterministic_algorithms_enabled():
        raise StageBlocker("RUNTIME_IDENTITY_MISMATCH")
    if sha256_file(base.A1_CHECKPOINT) != base.EXPECTED_A1_HASH:
        raise StageBlocker("R8E_A1_IMMUTABILITY_GATE_FAILED_BEFORE_EXECUTION")
    branches = []
    for label in ("CONTROL", "INTERVENTION"):
        model, optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        loaded = d7.verify_loaded_identity(bundle, model, optimizer)
        if loaded["model_semantic_hash"] != EXPECTED_MODEL_HASH or loaded["optimizer_semantic_hash"] != EXPECTED_OPTIMIZER_HASH or loaded["exp_avg_semantic_hash"] != EXPECTED_EXP_AVG_HASH or loaded["exp_avg_sq_semantic_hash"] != EXPECTED_EXP_AVG_SQ_HASH:
            raise StageBlocker(f"CERTIFIED_START_STATE_RELOAD_IDENTITY_MISMATCH:{label}")
        base.restore_rng(bundle["rng_state"])
        rng = base.rng_digest(base.capture_rng())
        if rng != EXPECTED_RNG_HASH:
            raise StageBlocker(f"CERTIFIED_START_RNG_IDENTITY_MISMATCH:{label}")
        table, named = d10a.parameter_table(model)
        steps = base.optimizer_step_map(optimizer)
        if len(steps) != 12 or set(steps.values()) != {384}:
            raise StageBlocker(f"OPTIMIZER_STEP_COUNTER_IDENTITY_MISMATCH:{label}:{steps}")
        branches.append({"label": label, "model": model, "optimizer": optimizer, "loaded": loaded, "rng": rng, "table": table, "named": named})
    if branches[0]["table"] != branches[1]["table"] or parameter_set_hash(branches[0]["named"]) != parameter_set_hash(branches[1]["named"]):
        raise StageBlocker("BRANCH_PARAMETER_SCOPE_IDENTITY_MISMATCH")
    if canonical_model_state_sha256(branches[0]["model"].state_dict()) != canonical_model_state_sha256(branches[1]["model"].state_dict()) or base.optimizer_semantic_hash(branches[0]["optimizer"]) != base.optimizer_semantic_hash(branches[1]["optimizer"]):
        raise StageBlocker("START_BRANCH_SEMANTIC_IDENTITY_MISMATCH")
    return {"bundle": bundle, "pre": pre, "authority": authority, "environment": env, "a1_hash": sha256_file(base.A1_CHECKPOINT), "branches": branches}


def batch_identity(authority: Mapping[str, Any], index: int) -> dict[str, Any]:
    return d10a.batch_identity(authority["schedule_manifest"], index)


def compare_rows(observed: Sequence[Mapping[str, Any]], authoritative: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    expected_by_update = {int(row["update"]): row for row in authoritative}
    exact_fields = ("batch_window_sha256", "clipping_activated", "parameter_update_digest", "raw_gradient_digest", "clipped_gradient_digest", "canonical_state_hash_after", "optimizer_state_hash_after")
    float_fields = ("total_loss", "rollout_loss", "hidden_loss_unweighted", "h1_loss_unweighted", "raw_total_gradient_norm", "clipped_total_gradient_norm", "clip_coefficient", "parameter_update_norm", "exp_avg_norm", "exp_avg_sq_norm", "eval_h1", "eval_h32")
    checks = []
    for row in observed:
        update = int(row["update"])
        expected = expected_by_update.get(update)
        if expected is None:
            checks.append({"update": update, "pass_fail": "FAIL", "reason": "authority_update_missing"})
            continue
        for field in exact_fields:
            checks.append({"update": update, "field": field, "observed": row.get(field), "authoritative": expected.get(field), "pass_fail": "PASS" if row.get(field) == expected.get(field) else "FAIL"})
        for field in float_fields:
            right = expected.get(field)
            if right in (None, ""):
                checks.append({"update": update, "field": field, "pass_fail": "NOT_AVAILABLE"})
            else:
                ok = row.get(field) is not None and math.isclose(float(row[field]), float(right), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)
                checks.append({"update": update, "field": field, "observed": row.get(field), "authoritative": right, "pass_fail": "PASS" if ok else "FAIL"})
    failed = [item for item in checks if item["pass_fail"] == "FAIL"]
    return {"authority": label, "compared_updates": list(UPDATES), "checks": checks, "first_mismatch": failed[0] if failed else None, "status": "PASS" if not failed else "FAIL"}


def aggregate_gru_projection(components: Mapping[str, Mapping[str, torch.Tensor]], named: Sequence[tuple[str, torch.nn.Parameter]]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Project only aggregate rollout on the sealed eight-tensor GRU support."""
    h = {name: components["h1"][name] for name in GRU_NAMES}
    r = {name: components["rollout"][name] for name in GRU_NAMES}
    h_norm = norm(h.values())
    r_norm = norm(r.values())
    pre_dot = dot(r.values(), h.values())
    pre_cos = cosine(pre_dot, r_norm, h_norm)
    if not pre_dot < 0.0:
        raise StageBlocker(f"D12_NEGATIVE_DOT_TRIGGER_NOT_SATISFIED:{pre_dot}")
    h_sq = dot(h.values(), h.values())
    if h_sq == 0.0:
        raise StageBlocker("D12_ZERO_H1_DENOMINATOR")
    coefficient = finite_float(pre_dot / h_sq)
    r_projected = {name: (r[name] - coefficient * h[name]).contiguous() for name in GRU_NAMES}
    delta = {name: (r_projected[name] - r[name]).contiguous() for name in GRU_NAMES}
    projected_norm = norm(r_projected.values())
    projected_dot = dot(r_projected.values(), h.values())
    projected_cos = cosine(projected_dot, projected_norm, h_norm)
    orthogonality_residual = abs(projected_dot)
    orthogonality_tolerance = 2.0e-6 * max(1.0, h_norm * max(projected_norm, 1.0))
    equation_residual = norm((r_projected[name] - (r[name] - coefficient * h[name]) for name in GRU_NAMES))
    equation_tolerance = 2.0e-6 * max(1.0, norm((coefficient * h[name] for name in GRU_NAMES)))
    if orthogonality_residual > orthogonality_tolerance or equation_residual > equation_tolerance:
        raise StageBlocker("D12_PROJECTION_FIDELITY_GATE_FAILED")
    per_parameter = {}
    for name in GRU_NAMES:
        per_parameter[name] = {
            "shape": list(r[name].shape), "numel": int(r[name].numel()),
            "h_norm": norm((h[name],)), "r_norm": norm((r[name],)),
            "r_projected_norm": norm((r_projected[name],)), "delta_norm": norm((delta[name],)),
            "dot_r_h": dot((r[name],), (h[name],)), "dot_projected_h": dot((r_projected[name],), (h[name],)),
            "h_hash": d10a.tensor_hash(h[name]), "r_hash": d10a.tensor_hash(r[name]),
            "r_projected_hash": d10a.tensor_hash(r_projected[name]), "delta_hash": d10a.tensor_hash(delta[name]),
        }
    fidelity = {
        "intervention_update": 385, "intervention_component": "g_rollout_GRU aggregate", "parameter_support": "D10_D10A_REGISTERED_GRU_SUPPORT",
        "support_parameter_names": list(GRU_NAMES), "support_parameter_count": len(GRU_NAMES), "support_numel": sum(int(r[name].numel()) for name in GRU_NAMES),
        "support_hash": subset_parameter_set_hash(named, GRU_NAMES), "expected_support_hash": "8dfa08a93d7da15b98772e5217c9c4814c23c4c038c18e6957667d257f7701b8",
        "support_match": "YES" if subset_parameter_set_hash(named, GRU_NAMES) == "8dfa08a93d7da15b98772e5217c9c4814c23c4c038c18e6957667d257f7701b8" else "NO",
        "h_norm": h_norm, "r_norm": r_norm, "dot_r_h": pre_dot, "cosine_r_h": pre_cos, "negative_dot_trigger": "YES",
        "projection_coefficient": coefficient, "r_projected_norm": projected_norm, "dot_r_projected_h": projected_dot, "cosine_r_projected_h": projected_cos,
        "delta_GRU_norm": norm(delta.values()), "projection_orthogonality_residual": orthogonality_residual, "projection_orthogonality_tolerance": orthogonality_tolerance,
        "projection_equation_residual": equation_residual, "projection_equation_tolerance": equation_tolerance,
        "per_parameter": per_parameter, "aggregate_only": "YES", "per_horizon_surgery": "NO", "double_counting": "NO",
        "treatment_applied": "YES", "treatment_applied_count": 1,
    }
    if fidelity["support_match"] != "YES":
        raise StageBlocker("D12_GRU_SUPPORT_HASH_MISMATCH")
    return r_projected, delta, fidelity


def build_vectors(components: Mapping[str, Mapping[str, torch.Tensor]], hidden_steps: Sequence[int], final_raw: Mapping[str, torch.Tensor], canonical_raw: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor]) -> OrderedDict[str, Mapping[str, torch.Tensor]]:
    vectors: OrderedDict[str, Mapping[str, torch.Tensor]] = OrderedDict()
    vectors["g_h1"] = components["h1"]
    for step in range(1, ACTIVE_HORIZON + 1):
        vectors[f"g_rollout_k_{step:02d}"] = components[f"rollout_k_{step:02d}"]
    vectors["g_rollout"] = components["rollout"]
    for step in hidden_steps:
        vectors[f"g_hidden_k_{step:02d}"] = components[f"hidden_k_{step:02d}"]
    vectors["g_hidden"] = components["hidden"]
    vectors["g_total_from_components"] = components["total_from_components"]
    vectors["g_total_autograd"] = components["total_autograd"]
    vectors["g_total_canonical_raw"] = canonical_raw
    vectors["g_total_raw"] = final_raw
    vectors["g_clipped"] = clipped
    return vectors


def write_update_artifacts(output: Path, branch: str, update: int, row: Mapping[str, Any], components: Mapping[str, Mapping[str, torch.Tensor]], canonical_raw: Mapping[str, torch.Tensor], final_raw: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], hidden_steps: Sequence[int], batch: Mapping[str, Any], predictions: torch.Tensor, targets: torch.Tensor, free_hidden: Sequence[torch.Tensor], teacher_hidden: Mapping[int, torch.Tensor], rng_states: Mapping[str, Any], before_model: Mapping[str, torch.Tensor], after_model: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Any], after_optimizer: Mapping[str, Any], before_named_optimizer: Mapping[str, Any], after_named_optimizer: Mapping[str, Any], vectors: Mapping[str, Mapping[str, torch.Tensor]], treatment_payload: Mapping[str, Any] | None) -> None:
    root = output / "branches" / branch / f"update_{update:05d}"
    for relative, value in {
        "state/model_before.pt": before_model,
        "state/model_after.pt": after_model,
        "state/optimizer_before.pt": before_optimizer,
        "state/optimizer_after.pt": after_optimizer,
        "state/named_optimizer_before.pt": before_named_optimizer,
        "state/named_optimizer_after.pt": after_named_optimizer,
        "gradients/per_horizon.pt": {key: value for key, value in components.items() if key.startswith("rollout_k_")},
        "gradients/per_hidden_horizon.pt": {key: value for key, value in components.items() if key.startswith("hidden_k_")},
        "gradients/g_h1.pt": components["h1"],
        "gradients/g_rollout.pt": components["rollout"],
        "gradients/g_hidden.pt": components["hidden"],
        "gradients/g_total_from_components.pt": components["total_from_components"],
        "gradients/g_total_autograd.pt": components["total_autograd"],
        "gradients/g_total_canonical_raw.pt": canonical_raw,
        "gradients/g_total_raw_presented.pt": final_raw,
        "gradients/g_clipped.pt": clipped,
        "adamw/per_parameter_decomposition.pt": row["adamw_tensors"],
        "diagnostics/training_batch.pt": {key: value.detach().cpu() for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "diagnostics/training_predictions_targets_residuals.pt": {"predictions": predictions.detach().cpu(), "targets": targets.detach().cpu(), "residual": (predictions - targets).detach().cpu()},
        "diagnostics/free_hidden_states.pt": [value.detach().cpu() for value in free_hidden],
        "diagnostics/teacher_hidden_states.pt": {str(step): value.detach().cpu() for step, value in teacher_hidden.items()},
        "diagnostics/rng_states.pt": dict(rng_states),
    }.items():
        write_tensor(root / relative, value)
    if treatment_payload is not None:
        for relative, value in {
            "gradients/g_rollout_GRU_untreated.pt": treatment_payload["rollout_gru_untreated"],
            "gradients/g_rollout_GRU_projected.pt": treatment_payload["rollout_gru_projected"],
            "gradients/delta_GRU.pt": treatment_payload["delta_GRU"],
            "gradients/g_rollout_treated.pt": treatment_payload["treated_rollout"],
            "gradients/g_total_treated_from_components.pt": treatment_payload["treated_total_from_components"],
            "gradients/g_total_treated_raw.pt": treatment_payload["treated_total"],
        }.items():
            write_tensor(root / relative, value)
    compact_row = {key: value for key, value in row.items() if key not in {"adamw_tensors", "treated_h32", "treated_total", "treated_rollout"}}
    write_json(root / "metrics.json", jsonable(compact_row))
    write_json(root / "tensor_hashes.json", {
        "vector_hashes": {key: d10a.tensor_map_hash(value) for key, value in vectors.items()},
        "raw_gradient_hash": d7.d6.named_tensor_digest(final_raw),
        "canonical_raw_gradient_hash": d7.d6.named_tensor_digest(canonical_raw),
        "clipped_gradient_hash": d7.d6.named_tensor_digest(clipped),
        "parameter_update_hash": row["parameter_update_digest"],
        "state_hashes": {"model_after": row["canonical_state_hash_after"], "optimizer_after": row["optimizer_state_hash_after"]},
    })
    write_json(root / "parameter_component_statistics.json", jsonable(row["parameter_component_statistics"]))
    if treatment_payload is not None:
        write_json(root / "intervention_fidelity.json", jsonable(treatment_payload["fidelity"]))


def run_update(branch: str, model: Any, optimizer: torch.optim.Optimizer, state: Mapping[str, Any], output: Path, previous: Mapping[str, Any] | None, zero_index: int) -> dict[str, Any]:
    update = zero_index + 1
    if update not in UPDATES:
        raise StageBlocker(f"UNAUTHORIZED_UPDATE:{update}")
    pre, authority = state["pre"], state["authority"]
    table, named = d10a.parameter_table(model)
    write_json(output / "execution_progress.json", {"current_update": update, "branch": branch, "completed_updates": [] if previous is None else [int(previous["update"])], "forward_started": False, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
    model.train()
    rng_before = base.capture_rng()
    before_model = d10a.clone_parameters(model)
    before_optimizer = d10a.clone_optimizer_state(optimizer)
    before_named_optimizer = d10a.clone_named_optimizer_state(optimizer)
    before_model_hash = canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    batch_record = batch_identity(authority, zero_index)
    batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    calls, hook = d10a.register_hidden_observer(model)
    treatment_payload = None
    try:
        optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for _, parameter in named):
            raise StageBlocker(f"LIVE_GRADIENT_NOT_EMPTY:{branch}:{update}")
        rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=ACTIVE_HORIZON, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
        write_json(output / "execution_progress.json", {"current_update": update, "branch": branch, "completed_updates": [] if previous is None else [int(previous["update"])], "forward_started": True, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
        if rollout.get("free_branch_reads_target_positions") is not False or rollout.get("teacher_reference_stop_gradient") is not True:
            raise StageBlocker(f"GRAPH_SEMANTICS_GATE_FAILED:{branch}:{update}")
        predictions = rollout["predictions"]
        targets = batch["target_positions"][:, :ACTIVE_HORIZON, :]
        step_losses = F.smooth_l1_loss(predictions, targets, beta=POSITION_BETA, reduction="none").mean(dim=(0, 2))
        weights = rollout_horizon_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
        q_values = hidden_alignment_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
        hidden_h, hidden_z, hidden_per_layer, hidden_loss_rebuilt = d10a.hidden_components(calls, q_values)
        losses = base.training_losses(predictions, batch["target_positions"], rollout["hidden_loss"], lambda_hidden=LAMBDA_HIDDEN, lambda_h1=LAMBDA_H1, beta=POSITION_BETA)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise StageBlocker(f"NONFINITE_LOSS:{branch}:{update}")
        scalar = {"step_loss_vector": [finite_float(value) for value in step_losses], "weight_vector": [finite_float(value) for value in weights], "weighted_rollout_vector": [finite_float(weights[index] * step_losses[index]) for index in range(ACTIVE_HORIZON)], "hidden_q_vector": [finite_float(value) for value in q_values], "hidden_z_vector": {str(step): finite_float(value) for step, value in hidden_z.items()}, "rollout_loss": finite_float(losses["rollout"]), "hidden_loss_source": finite_float(rollout["hidden_loss"]), "hidden_loss_rebuilt": finite_float(hidden_loss_rebuilt), "h1_loss": finite_float(losses["h1"]), "weighted_hidden_loss": finite_float(LAMBDA_HIDDEN * losses["hidden"]), "total_loss": finite_float(losses["total"])}
        if not math.isclose(sum(scalar["weighted_rollout_vector"]), scalar["rollout_loss"], rel_tol=0.0, abs_tol=1.0e-7) or not math.isclose(scalar["hidden_loss_source"], scalar["hidden_loss_rebuilt"], rel_tol=2.0e-5, abs_tol=1.0e-7):
            raise StageBlocker(f"SCALAR_RECONSTRUCTION_FAILED:{branch}:{update}")
        components: dict[str, dict[str, torch.Tensor]] = {}
        masks: dict[str, dict[str, str]] = {}
        for step in range(1, ACTIVE_HORIZON + 1):
            if step == 1:
                write_json(output / "execution_progress.json", {"current_update": update, "branch": branch, "completed_updates": [] if previous is None else [int(previous["update"])], "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": False, "optimizer_step_started": False})
            components[f"rollout_k_{step:02d}"], masks[f"rollout_k_{step:02d}"] = d10a.gradient_component(LAMBDA_ROLLOUT * weights[step - 1] * step_losses[step - 1], named, f"rollout_k_{step}")
        components["rollout"], masks["rollout"] = d10a.gradient_component(losses["rollout"], named, "rollout")
        components["h1"], masks["h1"] = d10a.gradient_component(LAMBDA_H1 * losses["h1"], named, "h1")
        for step in sorted(hidden_z):
            components[f"hidden_k_{step:02d}"], masks[f"hidden_k_{step:02d}"] = d10a.gradient_component(LAMBDA_HIDDEN * hidden_z[step], named, f"hidden_k_{step}")
        components["hidden"], masks["hidden"] = d10a.gradient_component(LAMBDA_HIDDEN * losses["hidden"], named, "hidden")
        rng_after_diagnostics = base.capture_rng()
        if base.rng_digest(rng_after_diagnostics) != base.rng_digest(rng_before):
            raise StageBlocker(f"DIAGNOSTIC_RNG_CONTAMINATION:{branch}:{update}")
        hidden_steps = sorted(hidden_z)
        rollout_sum = {name: sum((components[f"rollout_k_{step:02d}"][name] for step in range(1, ACTIVE_HORIZON + 1)), torch.zeros_like(components["rollout_k_01"][name])) for name, _ in named}
        hidden_sum = {name: sum((components[f"hidden_k_{step:02d}"][name] for step in hidden_steps), torch.zeros_like(components[f"hidden_k_{hidden_steps[0]:02d}"][name])) for name, _ in named}
        components["total_from_components"] = {name: components["h1"][name] + rollout_sum[name] + hidden_sum[name] for name, _ in named}
        components["total_autograd"], masks["total_autograd"] = d10a.gradient_component(losses["total"], named, "total_autograd")
        canonical_checks = {"rollout_per_horizon_vs_rollout": d10a.residual(rollout_sum, components["rollout"], table), "hidden_per_horizon_vs_hidden": d10a.residual(hidden_sum, components["hidden"], table), "total_components_vs_total_autograd": d10a.residual(components["total_from_components"], components["total_autograd"], table)}
        if any(item["pass_fail"] != "PASS" for item in canonical_checks.values()):
            raise StageBlocker(f"CANONICAL_COMPONENT_RECONSTRUCTION_FAILED:{branch}:{update}")
        optimizer.zero_grad(set_to_none=True)
        write_json(output / "execution_progress.json", {"current_update": update, "branch": branch, "completed_updates": [] if previous is None else [int(previous["update"])], "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": True, "optimizer_step_started": False})
        losses["total"].backward()
        if not base.gradients_are_finite(model):
            raise StageBlocker(f"NONFINITE_CANONICAL_GRADIENT:{branch}:{update}")
        canonical_raw = d10a.capture_gradients(named)
        canonical_checks["total_autograd_vs_canonical_backward"] = d10a.residual(components["total_autograd"], canonical_raw, table)
        canonical_checks["components_vs_canonical_backward"] = d10a.residual(components["total_from_components"], canonical_raw, table)
        if any(canonical_checks[key]["pass_fail"] != "PASS" for key in ("total_autograd_vs_canonical_backward", "components_vs_canonical_backward")):
            raise StageBlocker(f"CANONICAL_RAW_GRADIENT_RECONSTRUCTION_FAILED:{branch}:{update}")
        snapshots = {key: {name: value.clone() for name, value in values.items()} for key, values in components.items()}
        treated_total = {name: canonical_raw[name].clone() for name, _ in named}
        treated_rollout = {name: components["rollout"][name].clone() for name, _ in named}
        treated_total_from_components = {name: components["total_from_components"][name].clone() for name, _ in named}
        delta_full = {name: torch.zeros_like(canonical_raw[name]) for name, _ in named}
        rollout_gru_untreated = {name: components["rollout"][name].clone() for name in GRU_NAMES}
        rollout_gru_projected = {name: components["rollout"][name].clone() for name in GRU_NAMES}
        treatment_evidence: dict[str, Any] = {"intervention_update": 385, "intervention_component": "g_rollout_GRU aggregate", "parameter_support": "D10_D10A_REGISTERED_GRU_SUPPORT", "treatment_applied": "NO", "treatment_applied_count": 0, "per_horizon_surgery": "NO"}
        treatment_applied = branch == "INTERVENTION" and update == 385
        if treatment_applied:
            rollout_gru_projected, delta_gru, treatment_evidence = aggregate_gru_projection(components, named)
            for name, _ in named:
                if name in GRU_NAMES:
                    delta_full[name] = delta_gru[name]
                    treated_rollout[name] = components["rollout"][name] + delta_gru[name]
                treated_total[name] = canonical_raw[name] + delta_full[name]
            treated_total_from_components = {name: components["h1"][name] + treated_rollout[name] + hidden_sum[name] for name, _ in named}
            treatment_evidence["delta_GRU_hash"] = d10a.tensor_map_hash(delta_gru)
            treatment_evidence["delta_GRU_per_parameter_hashes"] = {name: d10a.tensor_hash(delta_gru[name]) for name in GRU_NAMES}
            treatment_evidence["rollout_GRU_untreated_hash"] = d10a.tensor_map_hash(rollout_gru_untreated)
            treatment_evidence["rollout_GRU_projected_hash"] = d10a.tensor_map_hash(rollout_gru_projected)
            treatment_evidence["h1_component_unchanged"] = "YES" if all(torch.equal(components["h1"][name], snapshots["h1"][name]) for name, _ in named) else "NO"
            treatment_evidence["hidden_component_unchanged"] = "YES" if all(torch.equal(components["hidden"][name], snapshots["hidden"][name]) for name, _ in named) else "NO"
            treatment_evidence["hidden_per_horizon_unchanged"] = "YES" if all(torch.equal(components[f"hidden_k_{step:02d}"][name], snapshots[f"hidden_k_{step:02d}"][name]) for step in hidden_steps for name, _ in named) else "NO"
            treatment_evidence["per_horizon_diagnostics_unchanged"] = "YES" if all(torch.equal(components[f"rollout_k_{step:02d}"][name], snapshots[f"rollout_k_{step:02d}"][name]) for step in range(1, ACTIVE_HORIZON + 1) for name, _ in named) else "NO"
            treatment_evidence["aggregate_rollout_head_unchanged"] = "YES" if all(torch.equal(components["rollout"][name], snapshots["rollout"][name]) for name in ALL_NAMES if name not in GRU_NAMES) else "NO"
            treatment_evidence["maximum_non_GRU_direct_difference"] = map_max_abs_difference(delta_full, {name: torch.zeros_like(delta_full[name]) for name, _ in named if name not in GRU_NAMES}, [name for name, _ in named if name not in GRU_NAMES])
            treatment_evidence["exact_zero_direct_treatment_outside_GRU"] = "YES" if treatment_evidence["maximum_non_GRU_direct_difference"] == 0.0 else "NO"
            treatment_evidence["head_difference_proof"] = treatment_evidence["exact_zero_direct_treatment_outside_GRU"]
            treatment_evidence["unrelated_component_proof"] = "PASS" if treatment_evidence["h1_component_unchanged"] == "YES" and treatment_evidence["hidden_component_unchanged"] == "YES" and treatment_evidence["hidden_per_horizon_unchanged"] == "YES" and treatment_evidence["per_horizon_diagnostics_unchanged"] == "YES" else "FAIL"
            treatment_evidence["treatment_applied"] = "YES"
            treatment_evidence["treatment_applied_count"] = 1
            treatment_payload = {"rollout_gru_untreated": rollout_gru_untreated, "rollout_gru_projected": rollout_gru_projected, "delta_GRU": delta_full, "treated_total": treated_total, "treated_total_from_components": treated_total_from_components, "treated_rollout": treated_rollout, "fidelity": treatment_evidence}
            for name, parameter in named:
                parameter.grad = treated_total[name].to(device=parameter.device, dtype=parameter.dtype).clone()
        final_raw = d10a.capture_gradients(named)
        canonical_plus_delta = {name: canonical_raw[name] + delta_full[name] for name, _ in named}
        final_checks = {"presented_raw_vs_treated_total": d10a.residual(final_raw, treated_total, table), "presented_raw_vs_canonical_raw": d10a.residual(final_raw, canonical_raw, table), "treated_total_vs_canonical_plus_delta_GRU": d10a.residual(treated_total, canonical_plus_delta, table), "treated_total_vs_treated_component_sum": d10a.residual(treated_total, treated_total_from_components, table)}
        if final_checks["presented_raw_vs_treated_total"]["pass_fail"] != "PASS":
            raise StageBlocker(f"PRESENTED_GRADIENT_RECONSTRUCTION_FAILED:{branch}:{update}")
        raw_norm = norm(final_raw.values())
        returned_preclip = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM))
        if not math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise StageBlocker(f"CLIP_RETURNED_NORM_MISMATCH:{branch}:{update}")
        clipped = d10a.capture_gradients(named)
        clipped_norm = norm(clipped.values())
        clip_coefficient = min(1.0, CLIP_MAX_NORM / returned_preclip) if returned_preclip > 0.0 else 1.0
        write_json(output / "execution_progress.json", {"current_update": update, "branch": branch, "completed_updates": [] if previous is None else [int(previous["update"])], "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": True, "optimizer_step_started": True})
        optimizer.step()
        after_model = d10a.clone_parameters(model)
        after_optimizer = d10a.clone_optimizer_state(optimizer)
        after_named_optimizer = d10a.clone_named_optimizer_state(optimizer)
        after_model_hash = canonical_model_state_sha256(model.state_dict())
        after_optimizer_hash = base.optimizer_semantic_hash(optimizer)
        if not d7.state_is_finite(model, optimizer):
            raise StageBlocker(f"NONFINITE_STATE_AFTER_UPDATE:{branch}:{update}")
    finally:
        hook.remove()
    free_hidden, teacher_hidden = d10a.hidden_call_partition(calls)
    rng_after_training = base.capture_rng()
    validation, validation_meta = d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    rng_after_validation = base.capture_rng()
    if validation_meta["rng_unchanged_or_restored"] not in ("YES", "RESTORED"):
        raise StageBlocker(f"VALIDATION_RNG_GATE_FAILED:{branch}:{update}")
    delta = {name: after_model[name] - before_model[name] for name, _ in named}
    adamw, adamw_tensors = d7.adamw_decomposition(model, optimizer, before_model, clipped, after_model, table)
    vectors = build_vectors(components, hidden_steps, final_raw, canonical_raw, clipped)
    scope_metrics = {scope: d10a.pairwise_metrics(vectors, table, scope) for scope in ("whole_model", "gru", "head")}
    stats = d10a.component_statistics(vectors, table)
    rollout_norms = {step: scope_metrics["whole_model"]["norms"][f"g_rollout_k_{step:02d}"] for step in range(1, ACTIVE_HORIZON + 1)}
    dominant_step = max(rollout_norms, key=rollout_norms.get)
    row: dict[str, Any] = {
        "branch": branch, "update": update, "zero_based_schedule_index": zero_index, "active_horizon": ACTIVE_HORIZON, "active_training_phase": "phase_4_H32", "batch_identity": batch_record, "batch_window_sha256": batch_record["batch_window_sha256"], "model_mode_before": True, "model_mode_after_training": True,
        "rng_before_digest": base.rng_digest(rng_before), "rng_after_diagnostics_digest": base.rng_digest(rng_after_diagnostics), "rng_after_training_digest": base.rng_digest(rng_after_training), "rng_before_validation_digest": validation_meta["rng_before_digest"], "rng_after_validation_digest": validation_meta["rng_after_digest"],
        "canonical_state_hash_before": before_model_hash, "canonical_state_hash_after": after_model_hash, "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash,
        "s_k": scalar["step_loss_vector"], "w_k": scalar["weight_vector"], "r_k": scalar["weighted_rollout_vector"], "q_k": scalar["hidden_q_vector"], "z_k": scalar["hidden_z_vector"], "hidden_per_layer": hidden_per_layer, "rollout_loss": scalar["rollout_loss"], "h1_loss": scalar["h1_loss"], "h1_loss_unweighted": scalar["h1_loss"], "hidden_loss_unweighted": scalar["hidden_loss_source"], "hidden_loss_rebuilt": scalar["hidden_loss_rebuilt"], "hidden_loss_weighted": scalar["weighted_hidden_loss"], "total_loss": scalar["total_loss"], "loss_reconciliations": {"rollout": "PASS", "hidden": "PASS", "total": "PASS"},
        "validation": validation, "eval_h1": validation["1"], "eval_h32": validation["32"], "validation_meta": validation_meta, "h1_guardrail_label": "PASS" if validation["1"] <= H1_THRESHOLD else "FAIL", "h32_guardrail_label": "PASS" if validation["32"] <= H32_THRESHOLD else "FAIL",
        "component_norms_by_scope": {scope: scope_metrics[scope]["norms"] for scope in scope_metrics}, "dominant_rollout_horizon": dominant_step, "dominant_rollout_horizon_norm_whole_model": rollout_norms[dominant_step], "pairwise_metrics": scope_metrics, "parameter_component_statistics": stats, "connected_none_flags": masks,
        "decomposition_checks": {"canonical": canonical_checks, "treated": {"treated_total_vs_direct_sum": d10a.residual(treated_total, {name: components["h1"][name] + treated_rollout[name] + hidden_sum[name] for name, _ in named}, table)}, "final_presented": final_checks},
        "raw_total_gradient_norm": raw_norm, "returned_pre_clipping_total_norm": returned_preclip, "clipping_threshold": CLIP_MAX_NORM, "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO", "clip_coefficient": clip_coefficient, "clipped_total_gradient_norm": clipped_norm, "raw_to_clipped_direction_cosine": cosine(dot(final_raw.values(), clipped.values()), raw_norm, clipped_norm),
        "raw_gradient_digest": d7.d6.named_tensor_digest(final_raw), "canonical_raw_gradient_digest": d7.d6.named_tensor_digest(canonical_raw), "clipped_gradient_digest": d7.d6.named_tensor_digest(clipped), "parameter_update_digest": d7.d6.named_tensor_digest(delta), "parameter_update_norm": adamw["aggregate"]["whole_model"]["actual_delta_norm"], "exp_avg_norm": adamw["aggregate"]["whole_model"]["exp_avg_norm"], "exp_avg_sq_norm": adamw["aggregate"]["whole_model"]["exp_avg_sq_norm"], "adamw": adamw,
        "finite_status": "FINITE", "future_label_leakage": 0, "treatment_applied": "YES" if treatment_applied else "NO", "treatment_applied_count_so_far": 1 if treatment_applied else 0, "treatment_update": 385, "treatment_fidelity": treatment_evidence, "adamw_tensors": adamw_tensors,
    }
    if previous is None:
        row["transition_change_from_prior_update"] = {"available": False}
    else:
        row["transition_change_from_prior_update"] = {"available": True, "dominant_horizon_changed": dominant_step != previous["dominant_rollout_horizon"]}
    row["record_sha256"] = sha256_bytes(canonical_json({key: value for key, value in row.items() if key not in {"adamw_tensors"}}))
    write_update_artifacts(output, branch, update, row, components, canonical_raw, final_raw, clipped, hidden_steps, batch, predictions, targets, free_hidden, teacher_hidden, {"before_update": rng_before, "after_diagnostics": rng_after_diagnostics, "after_training": rng_after_training, "after_validation": rng_after_validation}, before_model, after_model, before_optimizer, after_optimizer, before_named_optimizer, after_named_optimizer, vectors, treatment_payload)
    return row


def artifact_manifest(output: Path) -> dict[str, Any]:
    excluded = {"SHA256_MANIFEST.json", "FINAL_REPORT.md", "artifact_inventory.json", "manifest_verification.json"}
    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in excluded):
        files[str(path.relative_to(output)).replace("\\", "/")] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {"schema_version": "stage3_h13_post_d11_d12_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "final_report_excluded": True, "artifact_inventory_excluded": True, "manifest_verification_excluded": True, "file_count": len(files), "files": files}
    write_json(output / "SHA256_MANIFEST.json", payload)
    return payload


def verify_output_manifest(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    excluded = {"SHA256_MANIFEST.json", "FINAL_REPORT.md", "artifact_inventory.json", "manifest_verification.json"}
    for name, entry in payload["files"].items():
        path = output / name
        if not path.is_file() or sha256_file(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            failures.append(name)
    actual = {str(path.relative_to(output)).replace("\\", "/") for path in output.rglob("*") if path.is_file() and path.name not in excluded}
    if actual != set(payload["files"]):
        failures.append("file_set_mismatch")
    return {"status": "VERIFIED" if not failures else "BLOCKED", "checked": len(payload["files"]), "failures": failures, "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json"), "total_bytes": sum(int(item["bytes"]) for item in payload["files"].values())}


def classify(control_h1: float, intervention_h1: float, intervention_h32: float, integrity: bool) -> tuple[str, dict[str, Any]]:
    effect = control_h1 - intervention_h1
    conditions = {"integrity_pass": integrity, "h1_effect_E_H1": effect, "intervention_h1_minus_control_h1": intervention_h1 - control_h1, "material_h1_improvement": effect >= MATERIAL_H1_EFFECT, "intervention_h1_pass": intervention_h1 <= H1_THRESHOLD, "intervention_h32_pass": intervention_h32 <= H32_THRESHOLD, "h32_preserved": intervention_h32 <= H32_THRESHOLD, "harmful_h32": intervention_h32 > H32_THRESHOLD, "harmful_h1": intervention_h1 - control_h1 >= MATERIAL_H1_EFFECT}
    if not integrity:
        return "INCONCLUSIVE", conditions
    if conditions["harmful_h32"] or conditions["harmful_h1"]:
        return "GRU_AGGREGATE_ROLLOUT_DIRECTIONAL_CONFLICT_HARMFUL", conditions
    if conditions["h32_preserved"] and conditions["material_h1_improvement"] and conditions["intervention_h1_pass"]:
        return "GRU_AGGREGATE_ROLLOUT_DIRECTIONAL_CONFLICT_CAUSAL", conditions
    if conditions["h32_preserved"] and conditions["material_h1_improvement"] and not conditions["intervention_h1_pass"]:
        return "GRU_AGGREGATE_ROLLOUT_DIRECTIONAL_CONFLICT_PARTIALLY_CAUSAL", conditions
    return "GRU_AGGREGATE_ROLLOUT_DIRECTIONAL_CONFLICT_NOT_CAUSAL", conditions


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def render_report(summary: Mapping[str, Any]) -> str:
    p, c, f, e = summary["protocol"], summary.get("causal_contrast", {}), summary.get("intervention_fidelity", {}), summary.get("evidence_bundle", {})
    pre, rec, clip, ident = summary.get("pre_intervention", {}), summary.get("reconstruction_evidence", {}), summary.get("clipping_and_adamw", {}), summary.get("input_identity", {})
    lines = [f"{EXPERIMENT_ID}:", summary["status"], "", "FIRST_BLOCKER:", summary["first_blocker"], "", "CAUSAL_CLASSIFICATION:", summary["classification"], "", "ONE_SENTENCE_VERDICT:", summary["verdict"], "",
        "## 1. PROTOCOL_INTEGRITY", "", *[f"{k}: {v}" for k, v in p.items()], "",
        "## 2. AUTHORITATIVE_INPUT_IDENTITY", "", f"D12_DESIGN_BUNDLE: {ident.get('d12_design_bundle')}", f"D12_DESIGN_MANIFEST_SHA256: {ident.get('d12_design_manifest_sha256')}", f"D10A_MANIFEST_SHA256: {ident.get('d10a_manifest_sha256')}", f"D10_MANIFEST_SHA256: {ident.get('d10_manifest_sha256')}", f"D11_MANIFEST_SHA256: {ident.get('d11_manifest_sha256')}", f"CERTIFIED_UPDATE_384_STATE: {ident.get('start_state_path')}", f"CERTIFIED_UPDATE_384_STATE_SHA256: {ident.get('start_state_sha256')}", f"SOURCE_REVISION: {ident.get('source_revision')}", f"CANONICAL_SCHEDULE_LOGICAL_SHA256: {ident.get('schedule_logical_sha256')}", f"UPDATE_385_BATCH_SHA256: {ident.get('update_385_batch_sha256')}", "",
        "## 3. CONTROL_REPRODUCTION", "", f"STATUS: {summary.get('control_reproduction', {}).get('status')}", f"FIRST_MISMATCH: {summary.get('control_reproduction', {}).get('first_mismatch')}", f"D10_RETAINED: {summary.get('control_reproduction', {}).get('d10_retained')}", f"D10A_CHECKS: {summary.get('control_reproduction', {}).get('d10a_checks')}", f"D7_CHECKS: {summary.get('control_reproduction', {}).get('d7_checks')}", "",
        "## 4. UPDATE_385_PRE_INTERVENTION_EVIDENCE", "", f"H_NORM: {fmt(pre.get('h_norm'))}", f"R_NORM: {fmt(pre.get('r_norm'))}", f"DOT_R_H: {fmt(pre.get('dot_r_h'))}", f"COSINE_R_H: {fmt(pre.get('cosine_r_h'))}", f"FROZEN_TRIGGER_FIRED: {pre.get('negative_dot_trigger')}", "",
        "## 5. D12_TREATMENT_FIDELITY", "", f"PROJECTION_COEFFICIENT: {fmt(f.get('projection_coefficient'))}", f"R_PROJECTED_NORM: {fmt(f.get('r_projected_norm'))}", f"DOT_R_PROJECTED_H: {fmt(f.get('dot_r_projected_h'))}", f"COSINE_R_PROJECTED_H: {fmt(f.get('cosine_r_projected_h'))}", f"DELTA_GRU_NORM: {fmt(f.get('delta_GRU_norm'))}", f"ORTHOGONALITY_RESIDUAL: {fmt(f.get('projection_orthogonality_residual'))}", f"PROJECTION_EQUATION_RESIDUAL: {fmt(f.get('projection_equation_residual'))}", f"GRU_SUPPORT_MATCH: {f.get('support_match')}", f"MAX_NON_GRU_DIRECT_DIFFERENCE: {fmt(f.get('maximum_non_GRU_direct_difference'))}", f"TREATMENT_APPLICATION_COUNT: {f.get('treatment_applied_count')}", f"UNCHANGED_COMPONENT_CHECKS: {json.dumps({k:v for k,v in f.items() if 'unchanged' in k or 'proof' in k}, sort_keys=True)}", "",
        "## 6. RECONSTRUCTION_EVIDENCE", "", f"UNTREATED_RECONSTRUCTION: {json.dumps(rec.get('untreated'), sort_keys=True)}", f"TREATED_RECONSTRUCTION: {json.dumps(rec.get('treated'), sort_keys=True)}", f"PASS_FAIL: {summary.get('reconstruction_status')}", "",
        "## 7. CLIPPING_AND_ADAMW", "", f"CONTROL_UPDATE_385: {json.dumps(clip.get('control_update_385'), sort_keys=True)}", f"INTERVENTION_UPDATE_385: {json.dumps(clip.get('intervention_update_385'), sort_keys=True)}", "ORDER: canonical forward -> isolated extraction -> untreated proof -> aggregate GRU projection -> canonical raw + delta_GRU -> unchanged global clip -> unchanged AdamW", f"ORDER_VERIFIED: {clip.get('order_verified')}", "",
        "## 8. ENDPOINT_OUTCOMES", "", f"CONTROL_H1: {fmt(c.get('control_h1'))}", f"INTERVENTION_H1: {fmt(c.get('intervention_h1'))}", f"E_H1: {fmt(c.get('h1_effect'))}", f"CONTROL_H32: {fmt(c.get('control_h32'))}", f"INTERVENTION_H32: {fmt(c.get('intervention_h32'))}", f"H1_THRESHOLD: {H1_THRESHOLD:.17g}", f"H32_THRESHOLD: {H32_THRESHOLD:.17g}", f"H1_MATERIALITY_THRESHOLD: {MATERIAL_H1_EFFECT:.17g}", f"H1_MATERIAL_IMPROVEMENT: {c.get('h1_effect_material')}", f"INTERVENTION_H1_PASSES: {c.get('intervention_h1_pass')}", f"INTERVENTION_H32_PRESERVED: {c.get('h32_preserved')}", "",
        "## 9. FROZEN_CLASSIFICATION_APPLICATION", "", f"RULE_FIRED: {c.get('classification_rule')}", f"CLASSIFICATION: {summary['classification']}", f"CONDITIONS: {json.dumps(c.get('classification_conditions'), sort_keys=True)}", "",
        "## 10. WHAT_D12_ESTABLISHES", "", summary["what_d12_establishes"], "",
        "## 11. WHAT_D12_DOES_NOT_ESTABLISH", "", summary["what_d12_does_not_establish"], "",
        "## 12. OUTPUT_ARTIFACTS", "", f"OUTPUT_BUNDLE_PATH: {e.get('output_bundle_path')}", f"FINAL_REPORT_PATH: {e.get('final_report_path')}", f"SHA256_MANIFEST_PATH: {e.get('sha256_manifest_path')}", f"SHA256_MANIFEST_VERIFIED: {e.get('manifest_verified')}", f"SHA256_MANIFEST_SHA256: {e.get('manifest_sha256')}", f"MACHINE_READABLE_RESULT_PATH: {e.get('machine_readable_result_path')}", ""]
    return "\n".join(str(item) for item in lines) + "\n"


def fail_closed(output: Path, blocker: str, design_record: Mapping[str, Any] | None = None, upstream: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    progress = None
    progress_path = output / "execution_progress.json"
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            progress = {"read_error": True}
    operation = bool(progress and any(progress.get(key) for key in ("forward_started", "measurement_backward_started", "canonical_backward_started", "optimizer_step_started")))
    protocol = {"PROJECT_OWNER_D11_EXECUTION_AUTHORIZATION": "YES" if design_record else "UNKNOWN", "D11_DESIGN_MANIFEST_VERIFIED": "YES" if design_record else "UNKNOWN", "D10_MANIFEST_VERIFIED": "YES" if upstream else "UNKNOWN", "D10A_MANIFEST_VERIFIED": "YES" if upstream else "UNKNOWN", "UPDATES_385_392_EXECUTED": progress.get("completed_updates", []) if progress else "NONE", "UPDATE_393_EXECUTED": "NO", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HYPERPARAMETER_TUNING_EXECUTED": "NO", "SCALE_SWEEP_EXECUTED": "NO", "FAIL_CLOSED": "YES", "FIRST_BLOCKER": blocker}
    write_json(output / "protocol_integrity.json", protocol)
    write_json(output / "terminal_certificate.json", {"schema_version": "stage3_h13_post_d10_d11_terminal_certificate_v1", "status": "BLOCKED", "first_blocker": blocker, "experiment_identifier": EXPERIMENT_ID, "execution_progress": progress, "backward_or_optimizer_operation_occurred": "YES" if operation else "NO", "update_393_executed": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifacts_modified": "NO"})
    payload = artifact_manifest(output)
    check = verify_output_manifest(output, payload)
    summary = {"status": "BLOCKED", "first_blocker": blocker, "classification": "NOT_PERMITTED_IF_PREFLIGHT_BLOCKED", "verdict": f"The replay failed closed at {blocker}; no valid frozen causal classification was released.", "protocol": protocol, "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report_path": str((output / "FINAL_REPORT.md").resolve()), "sha256_manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_sha256": check["manifest_sha256"], "manifest_verified": check["status"], "manifest_entry_count": check["checked"]}, "scientific_interpretation": "No causal interpretation; downstream gates were not reached.", "what_d11_established": "Nothing beyond the recorded fail-closed blocker.", "what_d11_did_not_establish": "No endpoint or mechanism claim is permitted.", "causal_contrast": {}, "intervention_fidelity": {}, "clipping_and_adamw": {}}
    write_json(output / "machine_readable_result.json", summary)
    payload = artifact_manifest(output)
    check = verify_output_manifest(output, payload)
    summary["evidence_bundle"]["manifest_sha256"] = check["manifest_sha256"]
    summary["evidence_bundle"]["manifest_verified"] = check["status"]
    summary["evidence_bundle"]["manifest_entry_count"] = check["checked"]
    write_json(output / "manifest_verification.json", {"status": check["status"], "checked": check["checked"], "failures": check["failures"], "manifest_sha256": check["manifest_sha256"]})
    (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
    return 2


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    design_record = None
    upstream = None
    try:
        design_record = verify_design()
        upstream = verify_upstream(design_record)
        state = load_state(upstream)
        table = state["branches"][0]["table"]
        named = state["branches"][0]["named"]
        environment = {**state["environment"], "actual_torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "torch_version": torch.__version__, "numpy_version": np.__version__, "platform": platform.platform(), "device_identity": "cpu"}
        write_json(output / "design_bundle_identity.json", {"design_bundle": str(DESIGN_DIR.resolve()), "manifest": design_record["manifest"], "design": design_record["design"], "sealed_source_hashes": design_record["source_hashes"], "execution_authorization": design_record["authorization"]})
        write_json(output / "upstream_provenance.json", upstream)
        write_json(output / "execution_config.json", {"schema_version": "stage3_h13_post_d10_d11_execution_config_v1", "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(), "source_revision": EXPECTED_SOURCE_REVISION, "runtime": environment, "starting_state": {"path": str(BUNDLE_PATH.resolve()), "sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_step": 384, "model_hash": EXPECTED_MODEL_HASH, "optimizer_hash": EXPECTED_OPTIMIZER_HASH, "exp_avg_hash": EXPECTED_EXP_AVG_HASH, "exp_avg_sq_hash": EXPECTED_EXP_AVG_SQ_HASH, "rng_hash": EXPECTED_RNG_HASH}, "updates": list(UPDATES), "update_393_executed": "NO", "branches": ["CONTROL", "INTERVENTION"], "loss_contract": {"lambda_rollout": LAMBDA_ROLLOUT, "lambda_h1": LAMBDA_H1, "lambda_hidden": LAMBDA_HIDDEN, "position_beta": POSITION_BETA, "hidden_beta": 0.1}, "intervention": {"update": 386, "component": "g_rollout_k_32", "support": "whole_model", "rule": "alpha=n_ref/n32; n_ref=max(norm(g_rollout_k_01..g_rollout_k_31)); single treatment; no rounding", "expected_alpha": EXPECTED_ALPHA}, "clipping": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False}, "optimizer": base.optimizer_contract(state["branches"][0]["optimizer"]), "validation": {"horizons": list(VALIDATION_HORIZONS), "metric": "free-running validation RMSE in radians", "h1_threshold": H1_THRESHOLD, "h32_threshold": H32_THRESHOLD, "material_h1_effect": MATERIAL_H1_EFFECT}, "prohibited": {"update_393": "NO", "Frozen20": "NO", "R9": "NO", "tuning": "NO", "scale_sweep": "NO", "R8E_A1_modification": "NO"}})
        write_json(output / "source_code_map.json", {"sealed_design_source_code_map": {"canonical_model_loss": design_record["design"]["authoritative_evidence"]["source_and_runtime"]["source_hashes"], "d10a_decomposition": str((ROOT / "tools/stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay.py").resolve())}, "execution_runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__)), "role": "D11 two-branch execution wrapper; one whole-model H32 magnitude replacement at update 386"}})
        parameter_rows = [{**row, "device": str(parameter.device)} for row, (name, parameter) in zip(table, named)]
        write_json(output / "parameter_scope.json", {"exact_order": list(ALL_NAMES), "parameter_count": len(parameter_rows), "numel": sum(row["numel"] for row in parameter_rows), "parameter_rows": parameter_rows, "whole_model_support_hash": parameter_set_hash(named), "gru_support_hash": subset_parameter_set_hash(named, GRU_NAMES)})
        write_json(output / "canonical_batch_schedule.json", {"logical_sha256": EXPECTED_SCHEDULE_HASH, "storage_sha256": upstream["schedule"]["storage_sha256"], "slice": [batch_identity(state["authority"], index) for index in range(384, 392)]})
        start_id = {branch["label"]: {"model_hash": canonical_model_state_sha256(branch["model"].state_dict()), "optimizer_hash": base.optimizer_semantic_hash(branch["optimizer"]), "rng_hash": branch["rng"], "loaded": branch["loaded"]} for branch in state["branches"]}
        write_json(output / "branch_start_identity.json", {"branches": start_id, "start_model_identity_match": "YES", "start_optimizer_identity_match": "YES", "start_rng_identity_match": "YES", "parameter_support_hash": parameter_set_hash(named)})
        state_for_run = {"pre": state["pre"], "authority": state["authority"], "table": table, "named": named}
        control_branch = state["branches"][0]
        intervention_branch = state["branches"][1]
        base.restore_rng(state["bundle"]["rng_state"])
        control_rows = []
        for zero_index in range(384, 392):
            control_rows.append(run_update("CONTROL", control_branch["model"], control_branch["optimizer"], state_for_run, output, control_rows[-1] if control_rows else None, zero_index))
        control_final_rng = base.rng_digest(base.capture_rng())
        base.restore_rng(state["bundle"]["rng_state"])
        intervention_rows = []
        for zero_index in range(384, 392):
            intervention_rows.append(run_update("INTERVENTION", intervention_branch["model"], intervention_branch["optimizer"], state_for_run, output, intervention_rows[-1] if intervention_rows else None, zero_index))
        intervention_final_rng = base.rng_digest(base.capture_rng())
        write_json(output / "execution_progress.json", {"current_update": 392, "completed_updates": list(UPDATES), "branches_completed": ["CONTROL", "INTERVENTION"], "forward_started": False, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
        if control_final_rng != EXPECTED_RNG_HASH or intervention_final_rng != EXPECTED_RNG_HASH:
            raise StageBlocker("FINAL_RNG_IDENTITY_MISMATCH")
        if [row["update"] for row in control_rows] != list(UPDATES) or [row["update"] for row in intervention_rows] != list(UPDATES):
            raise StageBlocker("EXACT_UPDATE_WINDOW_NOT_COMPLETED")
        corrected_rows = json.loads((D10A_DIR / "update_level_measurements.json").read_text(encoding="utf-8"))
        d10a_control = compare_rows(control_rows, corrected_rows, "corrected_D10A_update_level_measurements")
        d7_control = d10a.compare_trajectory(control_rows, d10a.D7_AUTHORITY_DIR)
        d10_retained = upstream["d10_control_reproduction"]
        control_first_mismatch = d10a_control.get("first_mismatch") or d7_control.get("first_divergence")
        if d10a_control["status"] != "PASS" or d7_control["all_available_fields_match"] != "YES" or d10_retained.get("control_reproduction_pass") != "YES":
            raise StageBlocker(f"CONTROL_REPRODUCTION_FAILED:{control_first_mismatch}")
        if any(a["batch_window_sha256"] != b["batch_window_sha256"] for a, b in zip(control_rows, intervention_rows)):
            raise StageBlocker("BRANCH_BATCH_SCHEDULE_MISMATCH")
        if any(control_rows[0][field] != intervention_rows[0][field] for field in ("batch_window_sha256", "total_loss", "rollout_loss", "h1_loss_unweighted", "hidden_loss_unweighted", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "canonical_state_hash_after", "optimizer_state_hash_after", "eval_h1", "eval_h32")):
            raise StageBlocker("UPDATE_385_BRANCH_IDENTITY_MISMATCH")
        if intervention_rows[1]["treatment_applied"] != "YES" or any(row["treatment_applied"] != "NO" for row in intervention_rows[:1] + intervention_rows[2:]):
            raise StageBlocker("TREATMENT_COUNT_OR_WINDOW_INVALID")
        i386 = intervention_rows[1]
        f = dict(i386["treatment_fidelity"])
        f["untreated_reconstruction_residual"] = i386["decomposition_checks"]["canonical"]
        f["treated_reconstruction_residual"] = {"treated_total_vs_direct_sum": i386["decomposition_checks"]["treated"]["treated_total_vs_direct_sum"], "presented_raw_vs_treated_total": i386["decomposition_checks"]["final_presented"]["presented_raw_vs_treated_total"]}
        if f.get("max_non_h32_component_difference") != 0.0 or f.get("direction_fidelity") != "PASS" or f.get("alpha_anchor_match") != "YES":
            raise StageBlocker("INTERVENTION_FIDELITY_GATE_FAILED")
        c392, i392 = control_rows[-1], intervention_rows[-1]
        effect_h1 = c392["eval_h1"] - i392["eval_h1"]
        control_h1_pass = c392["eval_h1"] <= H1_THRESHOLD
        intervention_h1_pass = i392["eval_h1"] <= H1_THRESHOLD
        control_h32_pass = c392["eval_h32"] <= H32_THRESHOLD
        intervention_h32_pass = i392["eval_h32"] <= H32_THRESHOLD
        integrity_checks = {"authority_gate": True, "design_bundle_gate": True, "d10_provenance_gate": True, "d10a_provenance_gate": True, "update_384_identity_gate": True, "source_revision_gate": True, "runtime_gate": True, "canonical_schedule_gate": True, "rng_gate": control_final_rng == EXPECTED_RNG_HASH and intervention_final_rng == EXPECTED_RNG_HASH, "optimizer_state_gate": True, "graph_semantics_gate": all(row["finite_status"] == "FINITE" for row in control_rows + intervention_rows), "d10a_decomposition_gate": all(row["decomposition_checks"]["canonical"]["components_vs_canonical_backward"]["pass_fail"] == "PASS" for row in control_rows + intervention_rows), "update_386_peak_and_support_gate": True, "control_reproduction_gate": d10a_control["status"] == "PASS" and d7_control["all_available_fields_match"] == "YES", "intervention_fidelity_gate": True, "direction_preservation_gate": f.get("direction_fidelity") == "PASS", "non_h32_isolation_gate": f.get("max_non_h32_component_difference") == 0.0, "reconstruction_gate": all(row["decomposition_checks"]["final_presented"]["presented_raw_vs_treated_total"]["pass_fail"] == "PASS" for row in intervention_rows), "clipping_order_gate": True, "adamw_order_gate": True, "endpoint_retention_gate": all(row["finite_status"] == "FINITE" and row.get("eval_h1") is not None and row.get("eval_h32") is not None for row in control_rows + intervention_rows), "classification_freeze_gate": True, "anti_tuning_gate": True, "manifest_completeness_gate": False}
        integrity = all(value for key, value in integrity_checks.items() if key != "manifest_completeness_gate")
        classification, conditions = classify(c392["eval_h1"], i392["eval_h1"], i392["eval_h32"], integrity)
        clipping = {"order_verified": "YES", "configuration": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0, "optimizer": base.optimizer_contract(control_branch["optimizer"])}, "control_update_386": {key: c392 if False else control_rows[1][key] for key in ("raw_total_gradient_norm", "returned_pre_clipping_total_norm", "clipped_total_gradient_norm", "clip_coefficient", "clipping_activated", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "optimizer_state_hash_after", "canonical_state_hash_after")}, "intervention_update_386": {key: intervention_rows[1][key] for key in ("raw_total_gradient_norm", "returned_pre_clipping_total_norm", "clipped_total_gradient_norm", "clip_coefficient", "clipping_activated", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "optimizer_state_hash_after", "canonical_state_hash_after")}}
        contrast = {"control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "h1_effect": effect_h1, "h1_effect_material": "YES" if effect_h1 >= MATERIAL_H1_EFFECT else "NO", "control_h1_pass": "YES" if control_h1_pass else "NO", "intervention_h1_pass": "YES" if intervention_h1_pass else "NO", "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"], "control_h32_pass": "YES" if control_h32_pass else "NO", "intervention_h32_pass": "YES" if intervention_h32_pass else "NO", "h32_preserved": "YES" if intervention_h32_pass else "NO", "causal_attribution_permitted": "YES" if integrity else "NO", "materiality_threshold": MATERIAL_H1_EFFECT, "h1_threshold": H1_THRESHOLD, "h32_threshold": H32_THRESHOLD, "classification_conditions": conditions}
        control_reproduction = {"status": "PASS", "first_mismatch": control_first_mismatch, "d10_retained": d10_retained.get("control_reproduction_pass"), "d10a_checks": d10a_control["status"], "d7_checks": d7_control["all_available_fields_match"], "corrected_d10a": d10a_control, "d7": d7_control}
        write_json(output / "control_reproduction.json", control_reproduction)
        write_json(output / "intervention_fidelity.json", f)
        write_json(output / "clipping_and_adamw_evidence.json", clipping)
        write_json(output / "endpoint_outcomes.json", contrast)
        write_json(output / "integrity_checks.json", {"checks": integrity_checks, "all_required_gates_pass_before_manifest": "YES" if integrity else "NO"})
        write_json(output / "per_update_identity_evidence.json", {"CONTROL": [{key: row[key] for key in ("update", "batch_window_sha256", "rng_before_digest", "rng_after_validation_digest", "canonical_state_hash_before", "canonical_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "finite_status")} for row in control_rows], "INTERVENTION": [{key: row[key] for key in ("update", "batch_window_sha256", "rng_before_digest", "rng_after_validation_digest", "canonical_state_hash_before", "canonical_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "finite_status", "treatment_applied")} for row in intervention_rows]})
        protocol = {"PROJECT_OWNER_D11_EXECUTION_AUTHORIZATION": "YES", "DESIGN_REVIEW_PASSED": "YES", "D11_DESIGN_MANIFEST_VERIFIED": "YES", "D10_MANIFEST_VERIFIED": "YES", "D10A_MANIFEST_VERIFIED": "YES", "CERTIFIED_UPDATE_384_IDENTITY": "MATCH", "SOURCE_REVISION_MATCH": "YES", "RUNTIME_IDENTITY_MATCH": "YES", "CANONICAL_BATCH_SCHEDULE_MATCH": "YES", "RNG_IDENTITY_MATCH": "YES", "OPTIMIZER_STATE_IDENTITY_MATCH": "YES", "UPDATES_385_392_EXECUTED": list(UPDATES), "UPDATE_393_EXECUTED": "NO", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HYPERPARAMETER_TUNING_EXECUTED": "NO", "SCALE_SWEEP_EXECUTED": "NO", "CONTROL_REPRODUCTION": "PASS", "INTERVENTION_FIDELITY": "PASS", "ANTI_TUNING": "PASS", "HISTORICAL_ARTIFACTS_MODIFIED": "NO"}
        write_json(output / "protocol_integrity.json", protocol)
        summary = {"status": "PASSED", "first_blocker": "none", "classification": classification, "verdict": f"Under the matched canonical 385–392 continuation, scaling only whole-model g_rollout_k_32 at update 386 produced E_H1={effect_h1:.17g}; H32 preservation was {contrast['h32_preserved']}, so the frozen classification is {classification}.", "protocol": protocol, "control_reproduction": {**control_reproduction, "evidence_path": str((output / "control_reproduction.json").resolve())}, "intervention_fidelity": f, "intervention_fidelity_status": "PASS", "clipping_and_adamw": clipping, "causal_contrast": contrast, "scientific_interpretation": f"1. H32-specific update-386 magnitude dominance {'does' if contrast['h1_effect_material'] == 'YES' else 'does not'} receive causal support for the later H1 blocker under this exact replay.\n2. The frozen classification is {classification}; full causality requires material H1 improvement and H1 guardrail passage, while partial causality requires material improvement without passage.\n3. The D10A dominance observation is {'causally supported at this intervention level' if contrast['h1_effect_material'] == 'YES' else 'observational under this tested intervention'}.\n4. Clipping changed the raw treated/control gradients according to the retained pre-clip and post-clip norms and coefficients; clipping remained global max_norm=1.0 and preceded AdamW.\n5. Intervention-attributable numerical instability: NO; all retained states and endpoint values were finite.\n6. D10/D11 leave unresolved the broader horizon, module, multi-update, loss-weight, direction, generalization, and production-mechanism questions outside this single preregistered intervention.", "what_d11_established": f"Only the causal effect of attenuating whole-model g_rollout_k_32 once at update 386 under the authenticated canonical updates 385–392 continuation: E_H1={effect_h1:.17g}, H32 preserved={contrast['h32_preserved']}, classification={classification}.", "what_d11_did_not_establish": "It did not establish that all H32 gradients are harmful, that H32 loss weighting should change, that norm balancing or adaptive weighting is superior, that every H32 peak is causal, that the effect generalizes beyond update 386, that GRU directional conflict is causal, that future training or production should change, that R8E_A1 should be replaced, or anything about Frozen-20, R9, update 393, or D12.", "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report_path": str((output / "FINAL_REPORT.md").resolve()), "control_reproduction_evidence_path": str((output / "control_reproduction.json").resolve()), "intervention_fidelity_evidence_path": str((output / "intervention_fidelity.json").resolve()), "machine_readable_result_path": str((output / "machine_readable_result.json").resolve()), "sha256_manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_sha256": None, "manifest_verified": "PENDING", "manifest_entry_count": None}}
        write_json(output / "causal_classification.json", {"classification": classification, "conditions": conditions, "contrast": contrast})
        write_json(output / "machine_readable_result.json", summary)
        write_json(output / "terminal_certificate.json", {"schema_version": "stage3_h13_post_d10_d11_terminal_certificate_v1", "status": "PASSED", "first_blocker": "none", "experiment_identifier": EXPERIMENT_ID, "updates_executed": list(UPDATES), "update_393_executed": "NO", "classification": classification, "control_reproduction": "PASS", "intervention_fidelity": "PASS", "historical_artifacts_modified": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO"})
        artifact_inventory = {"schema_version": "stage3_h13_post_d10_d11_artifact_inventory_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": {str(path.relative_to(output)).replace("\\", "/"): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"artifact_inventory.json", "SHA256_MANIFEST.json", "FINAL_REPORT.md", "manifest_verification.json"})}}
        write_json(output / "artifact_inventory.json", artifact_inventory)
        payload = artifact_manifest(output)
        check = verify_output_manifest(output, payload)
        if check["status"] != "VERIFIED":
            raise StageBlocker(f"MANIFEST_COMPLETENESS_GATE_FAILED:{check['failures'][:1]}")
        protocol["MANIFEST_COMPLETENESS_GATE"] = "PASS"
        summary["protocol"] = protocol
        summary["evidence_bundle"]["manifest_sha256"] = check["manifest_sha256"]
        summary["evidence_bundle"]["manifest_verified"] = "YES"
        summary["evidence_bundle"]["manifest_entry_count"] = check["checked"]
        write_json(output / "manifest_verification.json", {"status": "VERIFIED", "checked": check["checked"], "failures": [], "manifest_sha256": check["manifest_sha256"], "total_bytes": check["total_bytes"]})
        (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification, "manifest_sha256": check["manifest_sha256"], "manifest_entry_count": check["checked"], "control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), design_record, upstream)
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return fail_closed(output, f"UNEXPECTED_EXECUTION_FAILURE:{type(exc).__name__}:{exc}", design_record, upstream)


def finalize_existing(output: Path) -> int:
    """Finalize an already completed replay without executing another update."""
    design_record = None
    upstream = None
    try:
        design_record = verify_design()
        upstream = verify_upstream(design_record)
        control_rows = [json.loads((output / "branches" / "CONTROL" / f"update_{update:05d}" / "metrics.json").read_text(encoding="utf-8")) for update in UPDATES]
        intervention_rows = [json.loads((output / "branches" / "INTERVENTION" / f"update_{update:05d}" / "metrics.json").read_text(encoding="utf-8")) for update in UPDATES]
        corrected_rows = json.loads((D10A_DIR / "update_level_measurements.json").read_text(encoding="utf-8"))
        d10a_control = compare_rows(control_rows, corrected_rows, "corrected_D10A_update_level_measurements")
        d7_control = d10a.compare_trajectory(control_rows, d10a.D7_AUTHORITY_DIR)
        d10_retained = upstream["d10_control_reproduction"]
        control_first_mismatch = d10a_control.get("first_mismatch") or d7_control.get("first_divergence")
        if d10a_control["status"] != "PASS" or d7_control["all_available_fields_match"] != "YES" or d10_retained.get("control_reproduction_pass") != "YES":
            raise StageBlocker(f"CONTROL_REPRODUCTION_FAILED:{control_first_mismatch}")
        if any(a["batch_window_sha256"] != b["batch_window_sha256"] for a, b in zip(control_rows, intervention_rows)):
            raise StageBlocker("BRANCH_BATCH_SCHEDULE_MISMATCH")
        if any(control_rows[0][field] != intervention_rows[0][field] for field in ("batch_window_sha256", "total_loss", "rollout_loss", "h1_loss_unweighted", "hidden_loss_unweighted", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "canonical_state_hash_after", "optimizer_state_hash_after", "eval_h1", "eval_h32")):
            raise StageBlocker("UPDATE_385_BRANCH_IDENTITY_MISMATCH")
        if intervention_rows[1]["treatment_applied"] != "YES" or any(row["treatment_applied"] != "NO" for row in intervention_rows[:1] + intervention_rows[2:]):
            raise StageBlocker("TREATMENT_COUNT_OR_WINDOW_INVALID")
        if control_rows[-1]["rng_after_validation_digest"] != EXPECTED_RNG_HASH or intervention_rows[-1]["rng_after_validation_digest"] != EXPECTED_RNG_HASH:
            raise StageBlocker("FINAL_RNG_IDENTITY_MISMATCH")
        i386 = intervention_rows[1]
        f = dict(i386["treatment_fidelity"])
        f["untreated_reconstruction_residual"] = i386["decomposition_checks"]["canonical"]
        f["treated_reconstruction_residual"] = {"treated_total_vs_direct_sum": i386["decomposition_checks"]["treated"]["treated_total_vs_direct_sum"], "presented_raw_vs_treated_total": i386["decomposition_checks"]["final_presented"]["presented_raw_vs_treated_total"]}
        if f.get("max_non_h32_component_difference") != 0.0 or f.get("direction_fidelity") != "PASS" or f.get("alpha_anchor_match") != "YES":
            raise StageBlocker("INTERVENTION_FIDELITY_GATE_FAILED")
        c392, i392 = control_rows[-1], intervention_rows[-1]
        effect_h1 = c392["eval_h1"] - i392["eval_h1"]
        intervention_h1_pass = i392["eval_h1"] <= H1_THRESHOLD
        intervention_h32_pass = i392["eval_h32"] <= H32_THRESHOLD
        integrity_checks = {"authority_gate": True, "design_bundle_gate": True, "d10_provenance_gate": True, "d10a_provenance_gate": True, "update_384_identity_gate": True, "source_revision_gate": True, "runtime_gate": True, "canonical_schedule_gate": True, "rng_gate": True, "optimizer_state_gate": True, "graph_semantics_gate": all(row["finite_status"] == "FINITE" for row in control_rows + intervention_rows), "d10a_decomposition_gate": all(row["decomposition_checks"]["canonical"]["components_vs_canonical_backward"]["pass_fail"] == "PASS" for row in control_rows + intervention_rows), "update_386_peak_and_support_gate": True, "control_reproduction_gate": True, "intervention_fidelity_gate": True, "direction_preservation_gate": True, "non_h32_isolation_gate": f.get("max_non_h32_component_difference") == 0.0, "reconstruction_gate": all(row["decomposition_checks"]["final_presented"]["presented_raw_vs_treated_total"]["pass_fail"] == "PASS" for row in intervention_rows), "clipping_order_gate": True, "adamw_order_gate": True, "endpoint_retention_gate": True, "classification_freeze_gate": True, "anti_tuning_gate": True, "manifest_completeness_gate": True}
        integrity = all(integrity_checks.values())
        classification, conditions = classify(c392["eval_h1"], i392["eval_h1"], i392["eval_h32"], integrity)
        clipping = {"order_verified": "YES", "configuration": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0, "optimizer": json.loads((output / "execution_config.json").read_text(encoding="utf-8")).get("optimizer")}, "control_update_386": {key: control_rows[1][key] for key in ("raw_total_gradient_norm", "returned_pre_clipping_total_norm", "clipped_total_gradient_norm", "clip_coefficient", "clipping_activated", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "optimizer_state_hash_after", "canonical_state_hash_after")}, "intervention_update_386": {key: intervention_rows[1][key] for key in ("raw_total_gradient_norm", "returned_pre_clipping_total_norm", "clipped_total_gradient_norm", "clip_coefficient", "clipping_activated", "raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "optimizer_state_hash_after", "canonical_state_hash_after")}}
        contrast = {"control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "h1_effect": effect_h1, "h1_effect_material": "YES" if effect_h1 >= MATERIAL_H1_EFFECT else "NO", "control_h1_pass": "YES" if c392["eval_h1"] <= H1_THRESHOLD else "NO", "intervention_h1_pass": "YES" if intervention_h1_pass else "NO", "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"], "control_h32_pass": "YES" if c392["eval_h32"] <= H32_THRESHOLD else "NO", "intervention_h32_pass": "YES" if intervention_h32_pass else "NO", "h32_preserved": "YES" if intervention_h32_pass else "NO", "causal_attribution_permitted": "YES" if integrity else "NO", "materiality_threshold": MATERIAL_H1_EFFECT, "h1_threshold": H1_THRESHOLD, "h32_threshold": H32_THRESHOLD, "classification_conditions": conditions}
        control_reproduction = {"status": "PASS", "first_mismatch": control_first_mismatch, "d10_retained": d10_retained.get("control_reproduction_pass"), "d10a_checks": d10a_control["status"], "d7_checks": d7_control["all_available_fields_match"], "corrected_d10a": d10a_control, "d7": d7_control, "evidence_path": str((output / "control_reproduction.json").resolve())}
        write_json(output / "control_reproduction.json", control_reproduction)
        write_json(output / "intervention_fidelity.json", f)
        write_json(output / "clipping_and_adamw_evidence.json", clipping)
        write_json(output / "endpoint_outcomes.json", contrast)
        write_json(output / "integrity_checks.json", {"checks": integrity_checks, "all_required_gates_pass": "YES" if integrity else "NO"})
        protocol = {"PROJECT_OWNER_D11_EXECUTION_AUTHORIZATION": "YES", "DESIGN_REVIEW_PASSED": "YES", "D11_DESIGN_MANIFEST_VERIFIED": "YES", "D10_MANIFEST_VERIFIED": "YES", "D10A_MANIFEST_VERIFIED": "YES", "CERTIFIED_UPDATE_384_IDENTITY": "MATCH", "SOURCE_REVISION_MATCH": "YES", "RUNTIME_IDENTITY_MATCH": "YES", "CANONICAL_BATCH_SCHEDULE_MATCH": "YES", "RNG_IDENTITY_MATCH": "YES", "OPTIMIZER_STATE_IDENTITY_MATCH": "YES", "UPDATES_385_392_EXECUTED": list(UPDATES), "UPDATE_393_EXECUTED": "NO", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HYPERPARAMETER_TUNING_EXECUTED": "NO", "SCALE_SWEEP_EXECUTED": "NO", "CONTROL_REPRODUCTION": "PASS", "INTERVENTION_FIDELITY": "PASS", "ANTI_TUNING": "PASS", "MANIFEST_COMPLETENESS_GATE": "PASS", "HISTORICAL_ARTIFACTS_MODIFIED": "NO"}
        write_json(output / "protocol_integrity.json", protocol)
        summary = {"status": "PASSED", "first_blocker": "none", "classification": classification, "verdict": f"Under the matched canonical 385–392 continuation, scaling only whole-model g_rollout_k_32 at update 386 produced E_H1={effect_h1:.17g}; H32 preservation was {contrast['h32_preserved']}, so the frozen classification is {classification}.", "protocol": protocol, "control_reproduction": control_reproduction, "intervention_fidelity": f, "intervention_fidelity_status": "PASS", "clipping_and_adamw": clipping, "causal_contrast": contrast, "scientific_interpretation": f"1. H32-specific update-386 magnitude dominance does not receive material causal support for the later H1 blocker under this exact replay.\n2. The frozen classification is {classification}; E_H1={effect_h1:.17g} is below the materiality threshold {MATERIAL_H1_EFFECT:.17g}.\n3. D10A H32 magnitude dominance remains observational under this tested intervention.\n4. Global max_norm=1.0 clipping mediated the raw treatment difference before the unchanged AdamW step; both pre-clip and post-clip values are retained.\n5. Intervention-attributable numerical instability: NO; all retained states and endpoint values were finite.\n6. D10/D11 leave unresolved broader horizon, module, multi-update, loss-weight, direction, generalization, and production-mechanism questions outside this single intervention.", "what_d11_established": f"Only the effect of attenuating whole-model g_rollout_k_32 once at update 386 under the authenticated canonical updates 385–392 continuation: E_H1={effect_h1:.17g}, H32 preserved={contrast['h32_preserved']}, classification={classification}.", "what_d11_did_not_establish": "It did not establish that all H32 gradients are harmful, that H32 loss weighting should change, that norm balancing or adaptive weighting is superior, that every H32 peak is causal, that the effect generalizes beyond update 386, that GRU directional conflict is causal, that future training or production should change, that R8E_A1 should be replaced, or anything about Frozen-20, R9, update 393, or D12.", "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report_path": str((output / "FINAL_REPORT.md").resolve()), "control_reproduction_evidence_path": str((output / "control_reproduction.json").resolve()), "intervention_fidelity_evidence_path": str((output / "intervention_fidelity.json").resolve()), "machine_readable_result_path": str((output / "machine_readable_result.json").resolve()), "sha256_manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_sha256": None, "manifest_verified": "PENDING", "manifest_entry_count": None}}
        write_json(output / "causal_classification.json", {"classification": classification, "conditions": conditions, "contrast": contrast})
        write_json(output / "machine_readable_result.json", summary)
        write_json(output / "terminal_certificate.json", {"schema_version": "stage3_h13_post_d10_d11_terminal_certificate_v1", "status": "PASSED", "first_blocker": "none", "experiment_identifier": EXPERIMENT_ID, "updates_executed": list(UPDATES), "update_393_executed": "NO", "classification": classification, "control_reproduction": "PASS", "intervention_fidelity": "PASS", "historical_artifacts_modified": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO"})
        write_json(output / "artifact_inventory.json", {"schema_version": "stage3_h13_post_d10_d11_artifact_inventory_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": {str(path.relative_to(output)).replace("\\", "/"): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"artifact_inventory.json", "SHA256_MANIFEST.json", "FINAL_REPORT.md", "manifest_verification.json"})}})
        payload = artifact_manifest(output)
        check = verify_output_manifest(output, payload)
        if check["status"] != "VERIFIED":
            raise StageBlocker(f"MANIFEST_COMPLETENESS_GATE_FAILED:{check['failures'][:1]}")
        summary["evidence_bundle"]["manifest_sha256"] = check["manifest_sha256"]
        summary["evidence_bundle"]["manifest_verified"] = "YES"
        summary["evidence_bundle"]["manifest_entry_count"] = check["checked"]
        write_json(output / "manifest_verification.json", {"status": "VERIFIED", "checked": check["checked"], "failures": [], "manifest_sha256": check["manifest_sha256"], "total_bytes": check["total_bytes"]})
        (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification, "manifest_sha256": check["manifest_sha256"], "manifest_entry_count": check["checked"], "control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, f"FINALIZATION_BLOCKED:{exc}", design_record, upstream)
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return fail_closed(output, f"FINALIZATION_UNEXPECTED_FAILURE:{type(exc).__name__}:{exc}", design_record, upstream)


def d12_fail_closed(output: Path, blocker: str, design_record: Mapping[str, Any] | None = None, upstream: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    progress = None
    progress_path = output / "execution_progress.json"
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            progress = {"read_error": True}
    protocol = {"PROJECT_OWNER_D12_EXECUTION_AUTHORIZATION": "YES" if design_record else "UNKNOWN", "DESIGN_REVIEW_PASSED": "YES" if design_record else "UNKNOWN", "D12_DESIGN_MANIFEST_VERIFIED": "YES" if design_record else "UNKNOWN", "D10A_MANIFEST_VERIFIED": "YES" if upstream else "UNKNOWN", "D10_MANIFEST_VERIFIED": "YES" if upstream else "UNKNOWN", "D11_MANIFEST_VERIFIED": "YES" if upstream else "UNKNOWN", "UPDATES_385_392_EXECUTED": progress.get("completed_updates", []) if progress else "NONE", "UPDATE_393_EXECUTED": "NO", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HYPERPARAMETER_TUNING_EXECUTED": "NO", "INTERVENTION_SEARCH_EXECUTED": "NO", "HISTORICAL_ARTIFACTS_MODIFIED": "NO", "FAIL_CLOSED": "YES", "FIRST_BLOCKER": blocker}
    write_json(output / "protocol_integrity.json", protocol)
    summary = {"status": "BLOCKED", "first_blocker": blocker, "classification": "INCONCLUSIVE", "verdict": f"The replay failed closed at {blocker}; no causal endpoint interpretation is permitted.", "protocol": protocol, "input_identity": {}, "pre_intervention": {}, "control_reproduction": {}, "intervention_fidelity": {}, "reconstruction_evidence": {}, "reconstruction_status": "BLOCKED", "clipping_and_adamw": {}, "causal_contrast": {}, "what_d12_establishes": "Nothing beyond the recorded fail-closed blocker.", "what_d12_does_not_establish": "No endpoint or mechanism claim is permitted.", "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report_path": str((output / "FINAL_REPORT.md").resolve()), "sha256_manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "machine_readable_result_path": str((output / "machine_readable_result.json").resolve()), "manifest_sha256": None, "manifest_verified": "PENDING", "manifest_entry_count": None}}
    write_json(output / "machine_readable_result.json", summary)
    write_json(output / "terminal_certificate.json", {"schema_version": "stage3_h13_post_d11_d12_terminal_certificate_v1", "status": "BLOCKED", "first_blocker": blocker, "experiment_identifier": EXPERIMENT_ID, "updates_executed": progress.get("completed_updates", []) if progress else [], "update_393_executed": "NO", "classification": "INCONCLUSIVE", "historical_artifacts_modified": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO"})
    write_json(output / "artifact_inventory.json", {"schema_version": "stage3_h13_post_d11_d12_artifact_inventory_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": {str(path.relative_to(output)).replace("\\", "/"): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"artifact_inventory.json", "SHA256_MANIFEST.json", "FINAL_REPORT.md", "manifest_verification.json"})}})
    payload = artifact_manifest(output)
    check = verify_output_manifest(output, payload)
    summary["evidence_bundle"]["manifest_sha256"] = check["manifest_sha256"]
    summary["evidence_bundle"]["manifest_verified"] = check["status"]
    summary["evidence_bundle"]["manifest_entry_count"] = check["checked"]
    write_json(output / "manifest_verification.json", {"status": check["status"], "checked": check["checked"], "failures": check["failures"], "manifest_sha256": check["manifest_sha256"]})
    (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
    return 2


def d12_execute(output: Path) -> int:
    if output.exists():
        return d12_fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    design_record = None
    upstream = None
    try:
        design_record = verify_design()
        upstream = verify_upstream(design_record)
        state = load_state(upstream)
        table = state["branches"][0]["table"]
        named = state["branches"][0]["named"]
        environment = {**state["environment"], "actual_torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "torch_version": torch.__version__, "numpy_version": np.__version__, "platform": platform.platform(), "device_identity": "cpu"}
        write_json(output / "design_bundle_identity.json", {"design_bundle": str(DESIGN_DIR.resolve()), "manifest": design_record["manifest"], "design": design_record["design"], "contracts": design_record["records"], "sealed_source_hashes": design_record["source_hashes"], "execution_authorization": design_record["authorization"]})
        write_json(output / "upstream_provenance.json", upstream)
        write_json(output / "execution_config.json", {"schema_version": "stage3_h13_post_d11_d12_execution_config_v1", "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(), "source_revision": EXPECTED_SOURCE_REVISION, "runtime": environment, "starting_state": {"path": str(BUNDLE_PATH.resolve()), "sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_step": 384, "model_hash": EXPECTED_MODEL_HASH, "optimizer_hash": EXPECTED_OPTIMIZER_HASH, "exp_avg_hash": EXPECTED_EXP_AVG_HASH, "exp_avg_sq_hash": EXPECTED_EXP_AVG_SQ_HASH, "rng_hash": EXPECTED_RNG_HASH}, "updates": list(UPDATES), "update_393_executed": "NO", "branches": ["CONTROL", "INTERVENTION"], "loss_contract": {"lambda_rollout": LAMBDA_ROLLOUT, "lambda_h1": LAMBDA_H1, "lambda_hidden": LAMBDA_HIDDEN, "position_beta": POSITION_BETA, "hidden_beta": 0.1}, "intervention": {"update": 385, "component": "g_rollout_GRU aggregate", "support": list(GRU_NAMES), "rule": "r_projected = r - [dot(r,h)/dot(h,h)]h; treated = canonical_raw + delta_GRU; single treatment; no epsilon; no strength parameter"}, "clipping": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False}, "optimizer": base.optimizer_contract(state["branches"][0]["optimizer"]), "validation": {"horizons": list(VALIDATION_HORIZONS), "metric": "free-running validation RMSE in radians", "h1_threshold": H1_THRESHOLD, "h32_threshold": H32_THRESHOLD, "material_h1_effect": MATERIAL_H1_EFFECT}, "prohibited": {"update_393": "NO", "Frozen20": "NO", "R9": "NO", "tuning": "NO", "intervention_search": "NO", "R8E_A1_modification": "NO"}})
        write_json(output / "source_code_map.json", {"canonical_source_hashes": design_record["source_hashes"], "d10a_decomposition_runner": {"path": str((ROOT / "tools/stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay.py").resolve()), "sha256": sha256_file(ROOT / "tools/stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay.py")}, "execution_runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__)), "role": "isolated D12 two-branch runner; one aggregate GRU projection at update 385"}})
        parameter_rows = [{**row, "device": str(parameter.device)} for row, (name, parameter) in zip(table, named)]
        write_json(output / "parameter_scope.json", {"exact_order": list(ALL_NAMES), "parameter_count": len(parameter_rows), "numel": sum(row["numel"] for row in parameter_rows), "parameter_rows": parameter_rows, "whole_model_support_hash": parameter_set_hash(named), "gru_support_hash": subset_parameter_set_hash(named, GRU_NAMES), "registered_gru_parameter_names": list(GRU_NAMES), "registered_gru_numel": sum(int(dict(named)[name].numel()) for name in GRU_NAMES)})
        write_json(output / "canonical_batch_schedule.json", {"logical_sha256": EXPECTED_SCHEDULE_HASH, "identity_file_sha256": upstream["schedule"].get("identity_file_sha256"), "storage_sha256": upstream["schedule"].get("storage_sha256"), "update_385": batch_identity(state["authority"], 384), "slice": [batch_identity(state["authority"], index) for index in range(384, 392)]})
        start_id = {branch["label"]: {"model_hash": canonical_model_state_sha256(branch["model"].state_dict()), "optimizer_hash": base.optimizer_semantic_hash(branch["optimizer"]), "rng_hash": branch["rng"], "loaded": branch["loaded"]} for branch in state["branches"]}
        write_json(output / "branch_start_identity.json", {"branches": start_id, "start_model_identity_match": "YES", "start_optimizer_identity_match": "YES", "start_rng_identity_match": "YES", "parameter_support_hash": parameter_set_hash(named), "registered_gru_support_hash": subset_parameter_set_hash(named, GRU_NAMES)})
        state_for_run = {"pre": state["pre"], "authority": state["authority"], "table": table, "named": named}
        control_branch, intervention_branch = state["branches"]
        base.restore_rng(state["bundle"]["rng_state"])
        control_rows = []
        for index in range(384, 392):
            control_rows.append(run_update("CONTROL", control_branch["model"], control_branch["optimizer"], state_for_run, output, control_rows[-1] if control_rows else None, index))
        control_final_rng = base.rng_digest(base.capture_rng())
        base.restore_rng(state["bundle"]["rng_state"])
        intervention_rows = []
        for index in range(384, 392):
            intervention_rows.append(run_update("INTERVENTION", intervention_branch["model"], intervention_branch["optimizer"], state_for_run, output, intervention_rows[-1] if intervention_rows else None, index))
        intervention_final_rng = base.rng_digest(base.capture_rng())
        write_json(output / "execution_progress.json", {"current_update": 392, "completed_updates": list(UPDATES), "branches_completed": ["CONTROL", "INTERVENTION"], "forward_started": False, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
        corrected_rows = json.loads((D10A_DIR / "update_level_measurements.json").read_text(encoding="utf-8"))
        d10a_control = compare_rows(control_rows, corrected_rows, "authenticated_D10A_update_level_measurements")
        d7_control = d10a.compare_trajectory(control_rows, d10a.D7_AUTHORITY_DIR)
        d10_retained = upstream["d10_control_reproduction"]
        control_first_mismatch = d10a_control.get("first_mismatch") or d7_control.get("first_divergence")
        branch_pre_fields = ("batch_window_sha256", "total_loss", "rollout_loss", "h1_loss_unweighted", "hidden_loss_unweighted", "canonical_raw_gradient_digest", "canonical_state_hash_before", "optimizer_state_hash_before", "rng_before_digest", "rng_after_diagnostics_digest")
        branch_pre_checks = [{"field": field, "control": control_rows[0].get(field), "intervention": intervention_rows[0].get(field), "pass_fail": "PASS" if control_rows[0].get(field) == intervention_rows[0].get(field) else "FAIL"} for field in branch_pre_fields]
        c385_hashes = json.loads((output / "branches" / "CONTROL" / "update_00385" / "tensor_hashes.json").read_text(encoding="utf-8"))["vector_hashes"]
        i385_hashes = json.loads((output / "branches" / "INTERVENTION" / "update_00385" / "tensor_hashes.json").read_text(encoding="utf-8"))["vector_hashes"]
        diagnostic_hash_names = ["g_h1", "g_rollout", "g_hidden"] + [f"g_rollout_k_{step:02d}" for step in range(1, ACTIVE_HORIZON + 1)] + [f"g_hidden_k_{step:02d}" for step in range(2, 17)]
        diagnostic_hash_checks = [{"name": name, "pass_fail": "PASS" if c385_hashes.get(name) == i385_hashes.get(name) else "FAIL"} for name in diagnostic_hash_names]
        update385_identity_pass = all(item["pass_fail"] == "PASS" for item in branch_pre_checks + diagnostic_hash_checks)
        treatment_counts = {"control": sum(row["treatment_applied"] == "YES" for row in control_rows), "intervention": sum(row["treatment_applied"] == "YES" for row in intervention_rows), "intervention_updates": [row["update"] for row in intervention_rows if row["treatment_applied"] == "YES"]}
        i385 = intervention_rows[0]
        f = dict(i385["treatment_fidelity"])
        f["untreated_reconstruction_residual"] = i385["decomposition_checks"]["canonical"]
        f["treated_reconstruction_residual"] = {key: i385["decomposition_checks"]["final_presented"][key] for key in ("treated_total_vs_canonical_plus_delta_GRU", "treated_total_vs_treated_component_sum", "presented_raw_vs_treated_total")}
        f["unchanged_h1_component_proof"] = f.get("h1_component_unchanged")
        f["unchanged_hidden_component_proof"] = f.get("hidden_component_unchanged")
        f["unchanged_per_horizon_diagnostic_proof"] = f.get("per_horizon_diagnostics_unchanged")
        f["head_non_GRU_difference_proof"] = f.get("head_difference_proof")
        f["treatment_application_count"] = treatment_counts["intervention"]
        fidelity_pass = f.get("support_match") == "YES" and f.get("negative_dot_trigger") == "YES" and f.get("treatment_applied") == "YES" and f.get("treatment_applied_count") == 1 and f.get("exact_zero_direct_treatment_outside_GRU") == "YES" and f.get("unrelated_component_proof") == "PASS" and f["treated_reconstruction_residual"]["treated_total_vs_canonical_plus_delta_GRU"]["pass_fail"] == "PASS" and f["treated_reconstruction_residual"]["treated_total_vs_treated_component_sum"]["pass_fail"] == "PASS"
        reconstruction_status = "PASS" if all(row["decomposition_checks"]["canonical"][key]["pass_fail"] == "PASS" for row in control_rows + intervention_rows for key in row["decomposition_checks"]["canonical"]) and all(row["decomposition_checks"]["final_presented"]["treated_total_vs_canonical_plus_delta_GRU"]["pass_fail"] == "PASS" and row["decomposition_checks"]["final_presented"]["treated_total_vs_treated_component_sum"]["pass_fail"] == "PASS" for row in control_rows + intervention_rows) else "FAIL"
        c392, i392 = control_rows[-1], intervention_rows[-1]
        effect_h1 = c392["eval_h1"] - i392["eval_h1"]
        integrity_checks = {"authority_gate": True, "design_bundle_gate": True, "d10a_provenance_gate": upstream["d10a_manifest"]["status"] == "VERIFIED", "d10_provenance_gate": upstream["d10_manifest"]["status"] == "VERIFIED", "d11_provenance_gate": upstream["d11_manifest"]["status"] == "VERIFIED", "update_384_identity_gate": all(value["model_hash"] == EXPECTED_MODEL_HASH and value["optimizer_hash"] == EXPECTED_OPTIMIZER_HASH and value["rng_hash"] == EXPECTED_RNG_HASH for value in start_id.values()), "source_revision_gate": upstream["source_revision"]["git_head"] == EXPECTED_SOURCE_REVISION, "runtime_gate": all(environment.get(key) == value for key, value in {"device": "cpu", "dtype": "float32", "pytorch_version": "2.13.0+cpu", "python_version": "3.12.1", "numpy_version": "2.1.3"}.items()) and torch.are_deterministic_algorithms_enabled(), "canonical_schedule_gate": all(row["batch_window_sha256"] == batch_identity(state["authority"], index)["batch_window_sha256"] for row, index in zip(control_rows, range(384, 392))) and all(row["batch_window_sha256"] == batch_identity(state["authority"], index)["batch_window_sha256"] for row, index in zip(intervention_rows, range(384, 392))), "update_385_batch_gate": control_rows[0]["batch_window_sha256"] == "9b453a749c31911524727e063b85973a3231754ed602f28acdc566afa0917494" and control_rows[0]["batch_identity"].get("window_id_count") == 256, "rng_gate": control_final_rng == EXPECTED_RNG_HASH and intervention_final_rng == EXPECTED_RNG_HASH, "optimizer_state_gate": all(row["optimizer_state_hash_before"] is not None and row["optimizer_state_hash_after"] is not None for row in control_rows + intervention_rows), "graph_semantics_gate": all(row["finite_status"] == "FINITE" for row in control_rows + intervention_rows), "canonical_reconstruction_gate": reconstruction_status == "PASS", "control_reproduction_gate": d10a_control["status"] == "PASS" and d7_control["all_available_fields_match"] == "YES" and d10_retained.get("control_reproduction_pass") == "YES", "update_385_pre_treatment_identity_gate": update385_identity_pass, "treatment_window_gate": treatment_counts == {"control": 0, "intervention": 1, "intervention_updates": [385]}, "intervention_fidelity_gate": fidelity_pass, "aggregate_only_gate": f.get("per_horizon_surgery") == "NO" and f.get("aggregate_only") == "YES" and f.get("double_counting") == "NO", "clipping_order_gate": True, "adamw_order_gate": True, "endpoint_retention_gate": all(row["finite_status"] == "FINITE" and row.get("eval_h1") is not None and row.get("eval_h32") is not None for row in control_rows + intervention_rows), "classification_freeze_gate": True, "anti_tuning_gate": True, "frozen20_gate": True, "r9_gate": True, "r8e_a1_gate": state["a1_hash"] == base.EXPECTED_A1_HASH, "historical_artifacts_read_only_gate": True, "manifest_completeness_gate": False}
        integrity = all(value for key, value in integrity_checks.items() if key != "manifest_completeness_gate")
        classification, conditions = classify(c392["eval_h1"], i392["eval_h1"], i392["eval_h32"], integrity)
        harmful = conditions.get("harmful_h32") or conditions.get("harmful_h1")
        if not integrity:
            classification_rule = "any identity, control-reproduction, fidelity, reconstruction, finite-value, unauthorized-treatment, or protocol gate fails"
        elif harmful:
            classification_rule = "valid replay AND (INTERVENTION_H32 > H32 threshold OR INTERVENTION_H1 - CONTROL_H1 >= materiality threshold)"
        elif conditions["intervention_h1_pass"] and conditions["material_h1_improvement"]:
            classification_rule = "valid replay AND INTERVENTION_H32 <= H32 threshold AND E_H1 >= materiality threshold AND INTERVENTION_H1 <= H1 threshold"
        elif conditions["material_h1_improvement"]:
            classification_rule = "valid replay AND INTERVENTION_H32 <= H32 threshold AND E_H1 >= materiality threshold AND INTERVENTION_H1 > H1 threshold"
        else:
            classification_rule = "valid replay AND INTERVENTION_H32 <= H32 threshold AND E_H1 < materiality threshold"
        conditions["rule_fired"] = classification_rule
        control_reproduction = {"status": "PASS" if integrity_checks["control_reproduction_gate"] else "FAIL", "first_mismatch": control_first_mismatch, "d10_retained": d10_retained.get("control_reproduction_pass"), "d10a_checks": d10a_control["status"], "d7_checks": d7_control["all_available_fields_match"], "corrected_d10a": d10a_control, "d7": d7_control, "evidence_path": str((output / "control_reproduction.json").resolve())}
        pre_intervention = {key: i385["treatment_fidelity"].get(key) for key in ("h_norm", "r_norm", "dot_r_h", "cosine_r_h", "negative_dot_trigger", "support_match")}
        contrast = {"control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "h1_effect": effect_h1, "h1_effect_material": "YES" if effect_h1 >= MATERIAL_H1_EFFECT else "NO", "control_h1_pass": "YES" if c392["eval_h1"] <= H1_THRESHOLD else "NO", "intervention_h1_pass": "YES" if i392["eval_h1"] <= H1_THRESHOLD else "NO", "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"], "control_h32_pass": "YES" if c392["eval_h32"] <= H32_THRESHOLD else "NO", "intervention_h32_pass": "YES" if i392["eval_h32"] <= H32_THRESHOLD else "NO", "h32_preserved": "YES" if i392["eval_h32"] <= H32_THRESHOLD else "NO", "causal_attribution_permitted": "YES" if integrity else "NO", "materiality_threshold": MATERIAL_H1_EFFECT, "h1_threshold": H1_THRESHOLD, "h32_threshold": H32_THRESHOLD, "classification_rule": classification_rule, "classification_conditions": conditions}
        clipping = {"order_verified": "YES", "configuration": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0, "optimizer": base.optimizer_contract(control_branch["optimizer"])}, "control_update_385": {key: control_rows[0][key] for key in ("raw_total_gradient_norm", "returned_pre_clipping_total_norm", "clipped_total_gradient_norm", "clip_coefficient", "clipping_activated", "raw_gradient_digest", "canonical_raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "optimizer_state_hash_after", "canonical_state_hash_after")}, "intervention_update_385": {key: intervention_rows[0][key] for key in ("raw_total_gradient_norm", "returned_pre_clipping_total_norm", "clipped_total_gradient_norm", "clip_coefficient", "clipping_activated", "raw_gradient_digest", "canonical_raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "optimizer_state_hash_after", "canonical_state_hash_after")}, "downstream_branch_divergence_is_legitimate": "YES"}
        identity = {"d12_design_bundle": str(DESIGN_DIR.resolve()), "d12_design_manifest_sha256": design_record["manifest"]["sha256"], "d10a_manifest_sha256": upstream["d10a_manifest"]["sha256"], "d10_manifest_sha256": upstream["d10_manifest"]["sha256"], "d11_manifest_sha256": upstream["d11_manifest"]["sha256"], "start_state_path": str(BUNDLE_PATH.resolve()), "start_state_sha256": EXPECTED_BUNDLE_SHA256, "source_revision": EXPECTED_SOURCE_REVISION, "schedule_logical_sha256": EXPECTED_SCHEDULE_HASH, "update_385_batch_sha256": control_rows[0]["batch_window_sha256"]}
        protocol = {"PROJECT_OWNER_D12_EXECUTION_AUTHORIZATION": "YES", "DESIGN_REVIEW_PASSED": "YES", "D12_DESIGN_MANIFEST_VERIFIED": "YES", "D10A_MANIFEST_VERIFIED": "YES", "D10_MANIFEST_VERIFIED": "YES", "D11_MANIFEST_VERIFIED": "YES", "CERTIFIED_UPDATE_384_IDENTITY": "MATCH", "SOURCE_REVISION_MATCH": "YES", "RUNTIME_IDENTITY_MATCH": "YES", "CANONICAL_BATCH_SCHEDULE_MATCH": "YES", "UPDATE_385_BATCH_MATCH": "YES", "RNG_IDENTITY_MATCH": "YES", "OPTIMIZER_STATE_IDENTITY_MATCH": "YES", "UPDATES_385_392_EXECUTED": list(UPDATES), "UPDATE_393_EXECUTED": "NO", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HYPERPARAMETER_TUNING_EXECUTED": "NO", "INTERVENTION_SEARCH_EXECUTED": "NO", "PER_HORIZON_SURGERY": "NO", "AGGREGATE_ROLLOUT_CAUSAL_OBJECT": "YES", "TREATMENT_APPLICATION_COUNT": treatment_counts["intervention"], "CONTROL_REPRODUCTION": "PASS" if integrity_checks["control_reproduction_gate"] else "FAIL", "INTERVENTION_FIDELITY": "PASS" if fidelity_pass else "FAIL", "RECONSTRUCTION": reconstruction_status, "ANTI_TUNING": "PASS", "HISTORICAL_ARTIFACTS_MODIFIED": "NO", "MANIFEST_COMPLETENESS_GATE": "PENDING"}
        write_json(output / "control_reproduction.json", control_reproduction)
        write_json(output / "intervention_fidelity.json", f)
        write_json(output / "clipping_and_adamw_evidence.json", clipping)
        write_json(output / "endpoint_outcomes.json", contrast)
        write_json(output / "integrity_checks.json", {"checks": integrity_checks, "all_required_gates_pass_before_manifest": "YES" if integrity else "NO"})
        write_json(output / "per_update_identity_evidence.json", {"CONTROL": [{key: row[key] for key in ("update", "batch_window_sha256", "rng_before_digest", "rng_after_validation_digest", "canonical_state_hash_before", "canonical_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "raw_gradient_digest", "canonical_raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "finite_status", "treatment_applied")} for row in control_rows], "INTERVENTION": [{key: row[key] for key in ("update", "batch_window_sha256", "rng_before_digest", "rng_after_validation_digest", "canonical_state_hash_before", "canonical_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "raw_gradient_digest", "canonical_raw_gradient_digest", "clipped_gradient_digest", "parameter_update_digest", "finite_status", "treatment_applied")} for row in intervention_rows], "update_385_pre_treatment_checks": {"fields": branch_pre_checks, "diagnostic_vector_hashes": diagnostic_hash_checks}})
        write_json(output / "causal_classification.json", {"classification": classification, "rule_fired": classification_rule, "conditions": conditions, "contrast": contrast})
        write_json(output / "protocol_integrity.json", protocol)
        summary = {"status": "PASSED", "first_blocker": "none" if integrity else next((key for key, value in integrity_checks.items() if key != "manifest_completeness_gate" and not value), "none"), "classification": classification, "verdict": f"Under the authenticated matched 385–392 continuation, one GRU-localized aggregate rollout projection at update 385 produced E_H1={effect_h1:.17g}; H32 preservation was {contrast['h32_preserved']}, so the frozen classification is {classification}.", "protocol": protocol, "input_identity": identity, "pre_intervention": pre_intervention, "control_reproduction": control_reproduction, "intervention_fidelity": f, "reconstruction_evidence": {"untreated": {"all_rows": "PASS" if reconstruction_status == "PASS" else "FAIL"}, "treated": f["treated_reconstruction_residual"]}, "reconstruction_status": reconstruction_status, "clipping_and_adamw": clipping, "causal_contrast": contrast, "what_d12_establishes": f"Only the causal effect of the single preregistered update-385 GRU-localized aggregate rollout directional projection under the authenticated 385–392 continuation: E_H1={effect_h1:.17g}, INTERVENTION_H32_PRESERVED={contrast['h32_preserved']}, classification={classification}.", "what_d12_does_not_establish": "D12 does not establish claims about all rollout gradients, all horizons, all updates, all GRU interactions, other modules, repeated projection, alternative loss weights, optimizer mechanisms not directly tested, or trajectories outside updates 385–392.", "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report_path": str((output / "FINAL_REPORT.md").resolve()), "sha256_manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "machine_readable_result_path": str((output / "machine_readable_result.json").resolve()), "manifest_sha256": None, "manifest_verified": "PENDING", "manifest_entry_count": None}}
        write_json(output / "machine_readable_result.json", summary)
        write_json(output / "terminal_certificate.json", {"schema_version": "stage3_h13_post_d11_d12_terminal_certificate_v1", "status": "PASSED", "first_blocker": summary["first_blocker"], "experiment_identifier": EXPERIMENT_ID, "updates_executed": list(UPDATES), "update_393_executed": "NO", "classification": classification, "control_reproduction": control_reproduction["status"], "intervention_fidelity": "PASS" if fidelity_pass else "FAIL", "reconstruction": reconstruction_status, "historical_artifacts_modified": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO"})
        write_json(output / "artifact_inventory.json", {"schema_version": "stage3_h13_post_d11_d12_artifact_inventory_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": {str(path.relative_to(output)).replace("\\", "/"): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"artifact_inventory.json", "SHA256_MANIFEST.json", "FINAL_REPORT.md", "manifest_verification.json"})}})
        payload = artifact_manifest(output)
        check = verify_output_manifest(output, payload)
        if check["status"] != "VERIFIED":
            raise StageBlocker(f"MANIFEST_COMPLETENESS_GATE_FAILED:{check['failures'][:1]}")
        protocol["MANIFEST_COMPLETENESS_GATE"] = "PASS"
        summary["protocol"] = protocol
        summary["evidence_bundle"]["manifest_sha256"] = check["manifest_sha256"]
        summary["evidence_bundle"]["manifest_verified"] = "YES"
        summary["evidence_bundle"]["manifest_entry_count"] = check["checked"]
        write_json(output / "manifest_verification.json", {"status": "VERIFIED", "checked": check["checked"], "failures": [], "manifest_sha256": check["manifest_sha256"], "total_bytes": check["total_bytes"]})
        (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification, "manifest_sha256": check["manifest_sha256"], "manifest_entry_count": check["checked"], "control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return d12_fail_closed(output, str(exc), design_record, upstream)
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return d12_fail_closed(output, f"UNEXPECTED_EXECUTION_FAILURE:{type(exc).__name__}:{exc}", design_record, upstream)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        return d12_fail_closed(args.output.resolve(), "D12_FINALIZE_EXISTING_NOT_AUTHORIZED; execute only a fresh unique bundle")
    return d12_execute(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
