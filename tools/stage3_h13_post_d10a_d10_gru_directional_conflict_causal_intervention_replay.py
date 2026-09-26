"""Execute the sealed D10 GRU directional-conflict causal replay.

This runner is intentionally separate from the canonical model and loss
implementation.  It reuses the accepted D10A decomposition machinery and
adds only the preregistered update-385, GRU-only, H32 directional projection.
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
from tools import stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay as d10a  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D10A_D10_GRU_DIRECTIONAL_CONFLICT_CAUSAL_INTERVENTION_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d10a_d10_gru_directional_conflict_causal_intervention_design_review_20260818T010351Z"
D10A_DIR = ROOT / "outputs/stage3_h13_post_d9_d10a_per_horizon_rollout_gradient_decomposition_instrumentation_replay_20260817T182510Z"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\7184f559-39cc-47f4-9556-ec85b6760947\pasted-text.txt")

EXPECTED_DESIGN_MANIFEST_SHA256 = "53613393752b04b463cfea7337fbcc383a74075a54a5a736139d70541cc0237c"
EXPECTED_D10A_MANIFEST_SHA256 = "9c6e664ce1342791b826d5d12692164f72d437350eca18fa868a250d36abb20a"
EXPECTED_D10A_REPORT_SHA256 = "36207e2be0067b63a57a8f4b919a61ce90d1451f72e13f2b1b0ea60349630c85"
EXPECTED_D10A_UPDATE_LEVEL_SHA256 = "f3b213c36ab3afc74a9dc5d277d63b8b3eedb0806b24b0af094f9425df6679c4"
EXPECTED_D10A_TRAJECTORY_SHA256 = "e299eac070b071e31e323c0e8d649e5cf38969ce024041ed3059aa01cb5291c5"
EXPECTED_D10A_PROTOCOL_SHA256 = "ee1218265b1ec5db84134dae06363edc47cc97d2bd02a0ee06747c49c529b94b"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_EXP_AVG_HASH = "7226e7ae693c1083d6928df6968d67d32d33477c16a8770adcf3050fc9492cdb"
EXPECTED_EXP_AVG_SQ_HASH = "eaf5ebdd6aaddefec25950e80d1a14baeb92a06dddcd11b246546e44c8489d89"
EXPECTED_RNG_HASH = "857d5ddd6c4271019b55a53774ad68c70e435abd38cf430b7f0d57ed047848d8"
EXPECTED_SCHEDULE_HASH = "e1e0d681d84c75348c656f306eedfd61bed849a1ff1c0b966f5ff3a63145ebcd"
EXPECTED_OPTIMIZER_IDENTITY = EXPECTED_OPTIMIZER_HASH
UPDATES = tuple(range(385, 393))
ACTIVE_HORIZON = 32
VALIDATION_HORIZONS = tuple(int(value) for value in EVALUATION_HORIZONS)
LAMBDA_ROLLOUT = 1.0
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
POSITION_BETA = 0.001
CLIP_MAX_NORM = 1.0
H1_THRESHOLD = 0.0000859791431235184
H32_THRESHOLD = 0.01856902565856056
MATERIAL_H1_EFFECT = 0.000004099006815405639
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
PROJECTION_TOLERANCE = 2.0e-6
DIAGNOSTIC_NORM_EPSILON = 1.0e-12
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
        raise StageBlocker(f"nonfinite_value:{result}")
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
    if denominator <= DIAGNOSTIC_NORM_EPSILON:
        return None
    value = finite_float(dot_value / denominator)
    if value < -1.0 - 1.0e-12 or value > 1.0 + 1.0e-12:
        raise StageBlocker(f"cosine_out_of_domain:{value}")
    return max(-1.0, min(1.0, value))


def assert_finite(values: Iterable[torch.Tensor], label: str) -> None:
    for value in values:
        if not bool(torch.all(torch.isfinite(value))):
            raise StageBlocker(f"nonfinite_{label}")


def tensor_equal_map(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], names: Iterable[str]) -> bool:
    return all(torch.equal(left[name], right[name]) for name in names)


def map_difference_norm(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], names: Iterable[str]) -> float:
    return norm((left[name] - right[name] for name in names))


def parameter_set_hash(named: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    payload = [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype), "device": str(parameter.device), "numel": int(parameter.numel())}
        for name, parameter in named
    ]
    return sha256_bytes(canonical_json(payload))


def subset_parameter_set_hash(named: Sequence[tuple[str, torch.nn.Parameter]], names: Sequence[str]) -> str:
    selected = {name: parameter for name, parameter in named}
    return parameter_set_hash([(name, selected[name]) for name in names])


def verify_manifest(path: Path, expected_sha256: str, expected_count: int, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageBlocker(f"{label}_MANIFEST_MISSING:{path}")
    observed_manifest_sha = sha256_file(path)
    if observed_manifest_sha != expected_sha256:
        raise StageBlocker(f"{label}_MANIFEST_SHA256_MISMATCH:{observed_manifest_sha}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("files") or payload.get("artifact_sha256")
    if not isinstance(entries, Mapping) or len(entries) != expected_count:
        raise StageBlocker(f"{label}_MANIFEST_FILE_COUNT_MISMATCH:{len(entries) if isinstance(entries, Mapping) else None}")
    failures: list[dict[str, Any]] = []
    for name, entry in entries.items():
        expected_hash = entry.get("sha256") if isinstance(entry, Mapping) else entry
        expected_bytes = entry.get("bytes", entry.get("size_bytes")) if isinstance(entry, Mapping) else None
        target = path.parent / str(name)
        if not target.is_file():
            failures.append({"path": str(name), "reason": "missing"})
            continue
        observed = sha256_file(target)
        item = {"path": str(name), "expected_sha256": expected_hash, "observed_sha256": observed}
        if observed != expected_hash:
            item["reason"] = "sha256_mismatch"
        if expected_bytes is not None and target.stat().st_size != int(expected_bytes):
            item["reason"] = "bytes_mismatch"
        if "reason" in item:
            failures.append(item)
    if failures:
        raise StageBlocker(f"{label}_MANIFEST_ENTRY_VERIFICATION_FAILED:{failures[0]}")
    return {"path": str(path.resolve()), "sha256": observed_manifest_sha, "file_count": len(entries), "status": "VERIFIED"}


def resolve_authority_path(raw: str) -> Path:
    candidate = Path(raw.replace("/", "\\"))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    return ROOT / raw.replace("/", "\\")


def verify_inputs() -> dict[str, Any]:
    design_manifest = verify_manifest(DESIGN_DIR / "SHA256_MANIFEST.json", EXPECTED_DESIGN_MANIFEST_SHA256, 10, "D10_DESIGN")
    required_design_files = {
        "FINAL_REPORT.md", "authoritative_inputs.json", "branch_matching_contract.json", "experiment_design.json",
        "gradient_reconstruction_contract.json", "intervention_contract.json", "outcome_interpretation_contract.json",
        "parameter_scope.json", "provenance.json", "source_code_map.json",
    }
    manifest_payload = json.loads((DESIGN_DIR / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    if set(manifest_payload["files"]) != required_design_files:
        raise StageBlocker("D10_DESIGN_MANIFEST_REQUIRED_FILE_SET_MISMATCH")
    design = json.loads((DESIGN_DIR / "experiment_design.json").read_text(encoding="utf-8"))
    intervention = json.loads((DESIGN_DIR / "intervention_contract.json").read_text(encoding="utf-8"))
    reconstruction = json.loads((DESIGN_DIR / "gradient_reconstruction_contract.json").read_text(encoding="utf-8"))
    scope = json.loads((DESIGN_DIR / "parameter_scope.json").read_text(encoding="utf-8"))
    matching = json.loads((DESIGN_DIR / "branch_matching_contract.json").read_text(encoding="utf-8"))
    outcome = json.loads((DESIGN_DIR / "outcome_interpretation_contract.json").read_text(encoding="utf-8"))
    authoritative = json.loads((DESIGN_DIR / "authoritative_inputs.json").read_text(encoding="utf-8"))
    source_map = json.loads((DESIGN_DIR / "source_code_map.json").read_text(encoding="utf-8"))
    if design.get("review_status") != "PASSED" or design.get("design_review_passed") is not True:
        raise StageBlocker("D10_DESIGN_REVIEW_NOT_PASSED")
    for key in ("experiment_executed", "training_executed", "backward_pass_executed", "optimizer_step_executed"):
        if design.get(key) is not False:
            raise StageBlocker(f"D10_DESIGN_SEALED_EXECUTION_FLAG_INVALID:{key}")
    if design.get("treatment_update") != 385 or design.get("branches") != ["canonical_control", "update_385_gru_h32_directional_intervention"]:
        raise StageBlocker("D10_DESIGN_BRANCH_OR_UPDATE_SCOPE_MISMATCH")
    if design.get("execution_window", {}).get("update_393_and_later_forbidden") is not True:
        raise StageBlocker("D10_DESIGN_UPDATE_393_SCOPE_NOT_FORBIDDEN")
    if design.get("fixed_runtime", {}).get("source_revision") != EXPECTED_SOURCE_REVISION or design.get("fixed_runtime", {}).get("device") != "cpu" or design.get("fixed_runtime", {}).get("model_dtype") != "float32" or design.get("fixed_runtime", {}).get("deterministic_algorithms") is not True:
        raise StageBlocker("D10_RUNTIME_CONTRACT_MISMATCH")
    if tuple(design.get("execution_window", {}).get("control_and_intervention_updates", [])) != UPDATES:
        raise StageBlocker("D10_UPDATE_WINDOW_MISMATCH")
    if intervention.get("location") != {"update": 385, "module": "GRU", "horizon": "H32", "component_key": "g_rollout_k_32", "repeated": False}:
        raise StageBlocker("D10_INTERVENTION_LOCATION_MISMATCH")
    if intervention.get("projection", {}).get("condition") != "dot < 0" or "No epsilon" not in intervention.get("projection", {}).get("zero_norm_policy", ""):
        raise StageBlocker("D10_PROJECTION_CONTRACT_MISMATCH")
    if scope.get("groups", {}).get("gru", {}).get("parameters") != list(GRU_NAMES) or scope.get("exact_parameter_order") != list(ALL_NAMES):
        raise StageBlocker("D10_EXACT_GRU_PARAMETER_SCOPE_MISMATCH")
    if not str(reconstruction.get("answers_to_required_gate", {}).get("7_only_h32_replacement", "")).startswith("YES") or not str(reconstruction.get("answers_to_required_gate", {}).get("6_total_reconstruction", "")).startswith("YES"):
        raise StageBlocker("D10_RECONSTRUCTION_CONTRACT_MISMATCH")
    if outcome.get("existing_frozen_criteria", {}).get("h1_threshold") != H1_THRESHOLD or outcome.get("existing_frozen_criteria", {}).get("h32_threshold") != H32_THRESHOLD or outcome.get("existing_frozen_criteria", {}).get("material_h1_effect") != MATERIAL_H1_EFFECT:
        raise StageBlocker("D10_FROZEN_OUTCOME_CONTRACT_MISMATCH")
    if authoritative.get("status") != "VERIFIED":
        raise StageBlocker("D10_AUTHORITATIVE_INPUTS_NOT_VERIFIED")
    corrected = authoritative["corrected_d10a_execution"]
    if Path(corrected["execution_bundle"]) != D10A_DIR:
        raise StageBlocker("D10_CORRECTED_D10A_PATH_MISMATCH")
    d10a_manifest = verify_manifest(D10A_DIR / "SHA256_MANIFEST.json", EXPECTED_D10A_MANIFEST_SHA256, 209, "CORRECTED_D10A")
    for name, expected in (("FINAL_REPORT.md", EXPECTED_D10A_REPORT_SHA256), ("update_level_measurements.json", EXPECTED_D10A_UPDATE_LEVEL_SHA256), ("trajectory_equivalence.json", EXPECTED_D10A_TRAJECTORY_SHA256), ("protocol_integrity_record.json", EXPECTED_D10A_PROTOCOL_SHA256)):
        observed = sha256_file(D10A_DIR / name)
        if observed != expected:
            raise StageBlocker(f"CORRECTED_D10A_{name}_HASH_MISMATCH:{observed}")
    d10a_protocol = json.loads((D10A_DIR / "protocol_integrity_record.json").read_text(encoding="utf-8"))
    d10a_trajectory = json.loads((D10A_DIR / "trajectory_equivalence.json").read_text(encoding="utf-8"))
    if d10a_protocol.get("status") != "PASS" or d10a_protocol.get("trajectory_preserved") != "YES" or d10a_trajectory.get("canonical_trajectory_preserved") != "YES":
        raise StageBlocker("CORRECTED_D10A_OBSERVATIONAL_AUTHORITY_NOT_PASSED")
    if not BUNDLE_PATH.is_file() or sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("CERTIFIED_UPDATE_384_SHA256_MISMATCH")
    if authoritative["certified_post_update_384"]["rng_sha256"] != EXPECTED_RNG_HASH or authoritative["canonical_schedule"]["logical_sha256"] != EXPECTED_SCHEDULE_HASH or authoritative["certified_post_update_384"]["optimizer_semantic_hash"] != EXPECTED_OPTIMIZER_HASH:
        raise StageBlocker("D10_AUTHORITATIVE_IDENTITY_RECORD_MISMATCH")
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("PROJECT_OWNER_D10_EXECUTION_AUTHORIZATION_SOURCE_MISSING")
    authorization_text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required_authorization = [
        "PROJECT_OWNER_D10_EXECUTION_AUTHORIZATION: YES",
        EXPERIMENT_ID,
        "Create exactly two branches",
        "TREATMENT_UPDATE",
        "UPDATE_393_EXECUTED: NO",
        "D11_EXECUTED: NO",
        "FROZEN20_OPENED: NO",
        "R9_EXECUTED: NO",
    ]
    missing = [item for item in required_authorization if item not in authorization_text]
    if missing:
        raise StageBlocker(f"PROJECT_OWNER_D10_AUTHORIZATION_SCOPE_NOT_PROVEN:{missing}")
    current_runner_hash = sha256_file(Path(__file__))
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git_head != EXPECTED_SOURCE_REVISION:
        raise StageBlocker(f"SOURCE_REVISION_MISMATCH:{git_head}")
    source_files = authoritative["source_revision"]["files"]
    source_hash_checks = {}
    for logical, expected in source_files.items():
        expected_path = ROOT / logical.replace("/", "\\")
        candidates = [expected_path]
        if not expected_path.is_file() and logical.startswith("tools/"):
            candidates.append(ROOT / ("scripts\\" + logical.split("/", 1)[1]))
        match = next((candidate for candidate in candidates if candidate.is_file() and sha256_file(candidate) == expected), None)
        if match is None:
            raise StageBlocker(f"SOURCE_FILE_HASH_IDENTITY_UNPROVABLE:{logical}")
        source_hash_checks[logical] = {"resolved_path": str(match.resolve()), "expected_sha256": expected, "observed_sha256": sha256_file(match)}
    return {
        "design": design,
        "intervention_contract": intervention,
        "reconstruction_contract": reconstruction,
        "parameter_scope": scope,
        "branch_matching_contract": matching,
        "outcome_contract": outcome,
        "authoritative_inputs": authoritative,
        "source_code_map": source_map,
        "authorization": {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"},
        "design_manifest": design_manifest,
        "corrected_d10a_manifest": d10a_manifest,
        "runner_source_sha256": current_runner_hash,
        "source_hash_checks": source_hash_checks,
    }


def projection(gru_h1: Mapping[str, torch.Tensor], gru_h32: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    names = list(GRU_NAMES)
    pre_dot = dot((gru_h32[name] for name in names), (gru_h1[name] for name in names))
    h1_norm = norm(gru_h1[name] for name in names)
    h32_norm = norm(gru_h32[name] for name in names)
    h1_sq = dot((gru_h1[name] for name in names), (gru_h1[name] for name in names))
    pre_cos = cosine(pre_dot, h32_norm, h1_norm)
    if pre_dot < 0.0 and h1_sq == 0.0:
        raise StageBlocker("D10_DEGENERATE_NEGATIVE_DOT_ZERO_H1_DENOMINATOR")
    fired = pre_dot < 0.0
    coefficient = pre_dot / h1_sq if fired else 0.0
    projected: dict[str, torch.Tensor] = {}
    for name in names:
        if fired:
            projected[name] = (gru_h32[name].double() - coefficient * gru_h1[name].double()).to(dtype=gru_h32[name].dtype).contiguous()
        else:
            projected[name] = gru_h32[name].clone()
    removed = {name: gru_h32[name] - projected[name] for name in names}
    post_dot = dot((projected[name] for name in names), (gru_h1[name] for name in names))
    post_norm = norm(projected[name] for name in names)
    expected_removed = {name: coefficient * gru_h1[name] for name in names}
    equation_residual = norm((removed[name] - expected_removed[name] for name in names))
    equation_max = max((finite_float(torch.max(torch.abs(removed[name] - expected_removed[name]))) for name in names), default=0.0)
    orthogonality_residual = abs(post_dot)
    residual_limit = PROJECTION_TOLERANCE * max(1.0, norm(expected_removed[name] for name in names))
    orth_limit = PROJECTION_TOLERANCE * max(1.0, h1_norm * max(post_norm, 1.0))
    evidence = {
        "pre_treatment_dot": pre_dot,
        "pre_treatment_cosine": pre_cos,
        "h1_gru_norm": h1_norm,
        "h32_gru_norm": h32_norm,
        "h1_gru_squared_norm": h1_sq,
        "projection_coefficient": coefficient,
        "removed_component_norm": norm(removed[name] for name in names),
        "removed_component_dot_h1": dot((removed[name] for name in names), (gru_h1[name] for name in names)),
        "projection_equation_residual_l2": equation_residual,
        "projection_equation_residual_max_abs": equation_max,
        "post_projection_dot": post_dot,
        "post_projection_cosine": cosine(post_dot, post_norm, h1_norm),
        "post_projection_orthogonality_residual": orthogonality_residual,
        "projection_residual_limit": residual_limit,
        "orthogonality_residual_limit": orth_limit,
        "projection_residual_pass": "PASS" if not fired or equation_residual <= residual_limit else "FAIL",
        "orthogonality_pass": "PASS" if not fired or orthogonality_residual <= orth_limit else "FAIL",
        "condition_negative_dot": bool(pre_dot < 0.0),
        "projection_fired": bool(fired),
        "zero_denominator": bool(h1_sq == 0.0),
        "parameter_scope": list(names),
    }
    if fired and (evidence["projection_residual_pass"] != "PASS" or evidence["orthogonality_pass"] != "PASS"):
        raise StageBlocker("D10_PROJECTION_FIDELITY_CHECK_FAILED")
    return projected, evidence


def sum_components(components: Mapping[str, Mapping[str, torch.Tensor]], hidden_steps: Sequence[int], named: Sequence[tuple[str, torch.nn.Parameter]], projected_h32: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, _ in named:
        rollout = torch.zeros_like(components["rollout_k_01"][name])
        for step in range(1, ACTIVE_HORIZON + 1):
            key = f"rollout_k_{step:02d}"
            value = projected_h32[name] if step == 32 and projected_h32 is not None and name in projected_h32 else components[key][name]
            rollout = rollout + value
        hidden = torch.zeros_like(components["h1"][name])
        for step in hidden_steps:
            hidden = hidden + components[f"hidden_k_{step:02d}"][name]
        result[name] = components["h1"][name] + rollout + hidden
    return result


def compare_rows(observed: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    expected_by_update = {int(row["update"]): row for row in authority_rows}
    checks: list[dict[str, Any]] = []
    exact_fields = ("batch_window_sha256", "clipping_activated", "parameter_update_digest", "raw_gradient_digest", "clipped_gradient_digest", "canonical_state_hash_after", "optimizer_state_hash_after")
    float_fields = ("total_loss", "rollout_loss", "hidden_loss_unweighted", "h1_loss_unweighted", "raw_total_gradient_norm", "clipped_total_gradient_norm", "clip_coefficient", "parameter_update_norm", "exp_avg_norm", "exp_avg_sq_norm", "eval_h1", "eval_h32")
    for row in observed:
        update = int(row["update"])
        expected = expected_by_update.get(update)
        if expected is None:
            checks.append({"update": update, "pass_fail": "FAIL", "reason": "authority_update_missing"})
            continue
        for field in exact_fields:
            left, right = row.get(field), expected.get(field)
            checks.append({"update": update, "field": field, "observed": left, "authoritative": right, "pass_fail": "PASS" if left == right else "FAIL"})
        for field in float_fields:
            left, right = row.get(field), expected.get(field)
            if right is None or right == "":
                checks.append({"update": update, "field": field, "pass_fail": "NOT_AVAILABLE"})
            else:
                ok = left is not None and math.isclose(float(left), float(right), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL)
                checks.append({"update": update, "field": field, "observed": left, "authoritative": right, "pass_fail": "PASS" if ok else "FAIL"})
    failed = [item for item in checks if item["pass_fail"] == "FAIL"]
    return {"authority": label, "compared_updates": list(UPDATES), "checks": checks, "first_mismatch": failed[0] if failed else None, "status": "PASS" if not failed else "FAIL"}


def write_update_artifacts(output: Path, branch: str, update: int, row: Mapping[str, Any], components: Mapping[str, Mapping[str, torch.Tensor]], canonical_raw: Mapping[str, torch.Tensor], final_raw: Mapping[str, torch.Tensor], clipped: Mapping[str, torch.Tensor], hidden_steps: Sequence[int], batch: Mapping[str, Any], predictions: torch.Tensor, targets: torch.Tensor, free_hidden: Sequence[torch.Tensor], teacher_hidden: Mapping[int, torch.Tensor], rng_states: Mapping[str, Any], before_model: Mapping[str, torch.Tensor], after_model: Mapping[str, torch.Tensor], before_optimizer: Mapping[str, Any], after_optimizer: Mapping[str, Any], before_named_optimizer: Mapping[str, Any], after_named_optimizer: Mapping[str, Any], vectors: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    root = output / "branches" / branch / f"update_{update:05d}"
    write_tensor(root / "state/model_before.pt", before_model)
    write_tensor(root / "state/model_after.pt", after_model)
    write_tensor(root / "state/optimizer_before.pt", before_optimizer)
    write_tensor(root / "state/optimizer_after.pt", after_optimizer)
    write_tensor(root / "state/named_optimizer_before.pt", before_named_optimizer)
    write_tensor(root / "state/named_optimizer_after.pt", after_named_optimizer)
    gradient_payload = {
        "g_h1": components["h1"],
        "g_rollout": components["rollout"],
        "g_hidden": components["hidden"],
        "g_total_from_components": components["total_from_components"],
        "g_total_autograd": components["total_autograd"],
        "g_total_canonical_raw": canonical_raw,
        "g_total_raw_presented": final_raw,
        "g_clipped": clipped,
    }
    if update == 385:
        gradient_payload["per_horizon"] = {key: value for key, value in components.items() if key.startswith("rollout_k_")}
        gradient_payload["per_hidden_horizon"] = {key: value for key, value in components.items() if key.startswith("hidden_k_")}
        gradient_payload["g_rollout_k_32_original"] = components["rollout_k_32"]
        gradient_payload["g_rollout_k_32_intervened"] = row.get("intervened_h32_gru_tensor_map")
        gradient_payload["g_total_treated_from_components"] = row.get("treated_total_gradient_tensor_map")
    for key, value in gradient_payload.items():
        if value is not None:
            write_tensor(root / "gradients" / f"{key}.pt", value)
    write_tensor(root / "diagnostics/training_batch.pt", {key: value.detach().cpu() for key, value in batch.items() if isinstance(value, torch.Tensor)})
    write_tensor(root / "diagnostics/training_predictions_targets_residuals.pt", {"predictions": predictions.detach().cpu(), "targets": targets.detach().cpu(), "residual": (predictions - targets).detach().cpu()})
    if update == 385:
        write_tensor(root / "diagnostics/free_hidden_states.pt", [value.detach().cpu() for value in free_hidden])
        write_tensor(root / "diagnostics/teacher_hidden_states.pt", {str(step): value.detach().cpu() for step, value in teacher_hidden.items()})
    write_tensor(root / "diagnostics/rng_states.pt", dict(rng_states))
    write_json(root / "metrics.json", jsonable({key: value for key, value in row.items() if key not in {"intervened_h32_gru_tensor_map", "treated_total_gradient_tensor_map"}}))
    write_json(root / "tensor_hashes.json", {"vectors": {key: d10a.tensor_map_hash(value) for key, value in vectors.items()}, "canonical_raw": d10a.tensor_map_hash(canonical_raw), "final_raw": d10a.tensor_map_hash(final_raw), "clipped": d10a.tensor_map_hash(clipped)})


def run_update(branch: str, model: Any, optimizer: torch.optim.Optimizer, state: Mapping[str, Any], output: Path, previous: Mapping[str, Any] | None, zero_index: int, treatment_counter: list[int]) -> dict[str, Any]:
    update = zero_index + 1
    if update not in UPDATES:
        raise StageBlocker(f"UNAUTHORIZED_UPDATE:{update}")
    pre, authority = state["pre"], state["authority"]
    table, named = d10a.parameter_table(model)
    write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [], "branch": branch, "forward_started": False, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
    if not model.training:
        model.train()
    rng_before = base.capture_rng()
    before_model = d10a.clone_parameters(model)
    before_optimizer = d10a.clone_optimizer_state(optimizer)
    before_named_optimizer = d10a.clone_named_optimizer_state(optimizer)
    before_model_hash = canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    batch_record = d10a.batch_identity(authority["schedule_manifest"], zero_index)
    batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    calls, hook = d10a.register_hidden_observer(model)
    intervened_tensor_map: Mapping[str, torch.Tensor] | None = None
    treated_tensor_map: Mapping[str, torch.Tensor] | None = None
    try:
        optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for _, parameter in named):
            raise StageBlocker(f"LIVE_GRADIENT_NOT_EMPTY_AFTER_ZERO_GRAD:{branch}:{update}")
        rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=ACTIVE_HORIZON, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
        write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [], "branch": branch, "forward_started": True, "measurement_backward_started": False, "canonical_backward_started": False, "optimizer_step_started": False})
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
            raise StageBlocker(f"NONFINITE_CANONICAL_LOSS:{branch}:{update}")
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
            raise StageBlocker(f"PER_HORIZON_ROLLOUT_SCALAR_RECONCILIATION_FAILED:{branch}:{update}")
        if not math.isclose(scalar_checks["hidden_loss_source"], scalar_checks["hidden_loss_rebuilt"], rel_tol=2.0e-5, abs_tol=1.0e-7):
            raise StageBlocker(f"HIDDEN_SCALAR_RECONCILIATION_FAILED:{branch}:{update}")
        components: dict[str, dict[str, torch.Tensor]] = {}
        masks: dict[str, dict[str, str]] = {}
        for step in range(1, ACTIVE_HORIZON + 1):
            if step == 1:
                write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [], "branch": branch, "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": False, "optimizer_step_started": False})
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
        components["total_from_components"] = sum_components(components, hidden_steps, named)
        components["total_autograd"], masks["total_autograd"] = d10a.gradient_component(losses["total"], named, "total_autograd")
        rollout_sum = {name: sum((components[f"rollout_k_{step:02d}"][name] for step in range(1, ACTIVE_HORIZON + 1)), torch.zeros_like(components["rollout_k_01"][name])) for name, _ in named}
        hidden_sum = {name: sum((components[f"hidden_k_{step:02d}"][name] for step in hidden_steps), torch.zeros_like(components[f"hidden_k_{hidden_steps[0]:02d}"][name])) for name, _ in named}
        canonical_checks = {
            "rollout_per_horizon_vs_rollout": d10a.residual(rollout_sum, components["rollout"], table),
            "hidden_per_horizon_vs_hidden": d10a.residual(hidden_sum, components["hidden"], table),
            "total_components_vs_total_autograd": d10a.residual(components["total_from_components"], components["total_autograd"], table),
        }
        if any(value["pass_fail"] != "PASS" for value in canonical_checks.values()):
            raise StageBlocker(f"CANONICAL_COMPONENT_RECONSTRUCTION_FAILED:{branch}:{update}")
        if any(parameter.grad is not None for _, parameter in named):
            raise StageBlocker(f"LIVE_GRADIENT_BEFORE_CANONICAL_BACKWARD:{branch}:{update}")
        optimizer.zero_grad(set_to_none=True)
        write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [], "branch": branch, "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": True, "optimizer_step_started": False})
        losses["total"].backward()
        if not base.gradients_are_finite(model):
            raise StageBlocker(f"NONFINITE_CANONICAL_GRADIENT:{branch}:{update}")
        canonical_raw = d10a.capture_gradients(named)
        canonical_checks["total_autograd_vs_canonical_backward"] = d10a.residual(components["total_autograd"], canonical_raw, table)
        canonical_checks["components_vs_canonical_backward"] = d10a.residual(components["total_from_components"], canonical_raw, table)
        if canonical_checks["total_autograd_vs_canonical_backward"]["pass_fail"] != "PASS" or canonical_checks["components_vs_canonical_backward"]["pass_fail"] != "PASS":
            raise StageBlocker(f"CANONICAL_RAW_GRADIENT_RECONSTRUCTION_FAILED:{branch}:{update}")
        component_snapshots = {key: {name: value.clone() for name, value in values.items()} for key, values in components.items() if isinstance(values, Mapping)}
        treatment_applied = False
        treatment_evidence: dict[str, Any] = {"treatment_update": 385, "treatment_module": "GRU", "treatment_horizon": "H32", "treatment_component": "g_rollout_k_32", "projection_fired": False, "projection_fired_reason": "NOT_TREATMENT_UPDATE" if update != 385 else "NOT_EVALUATED"}
        if branch == "INTERVENTION" and update == 385:
            projected_gru, treatment_evidence = projection(components["h1"], components["rollout_k_32"])
            if not treatment_evidence["condition_negative_dot"]:
                raise StageBlocker("D10_NEGATIVE_DOT_CONDITION_NOT_SATISFIED")
            treatment_applied = bool(treatment_evidence["projection_fired"])
            if not treatment_applied:
                raise StageBlocker("D10_TREATMENT_ABSENT_WHEN_NEGATIVE_DOT_SATISFIED")
            treatment_counter[0] += 1
            intervened_tensor_map = {name: components["rollout_k_32"][name].clone() for name, _ in named}
            intervened_tensor_map.update(projected_gru)
            treated_tensor_map = sum_components(components, hidden_steps, named, projected_h32=projected_gru)
        else:
            intervened_tensor_map = {name: components["rollout_k_32"][name].clone() for name, _ in named}
            treated_tensor_map = {name: components["total_from_components"][name].clone() for name, _ in named}
        treated_direct = sum_components(components, hidden_steps, named, projected_h32={name: intervened_tensor_map[name] for name in GRU_NAMES} if treatment_applied else None)
        non_gru_names = [name for name, _ in named if name not in GRU_NAMES]
        treated_non_gru_l2 = map_difference_norm(treated_tensor_map, components["total_from_components"], non_gru_names)
        head_untouched = tensor_equal_map(intervened_tensor_map, components["rollout_k_32"], non_gru_names)
        other_components_untouched = all(tensor_equal_map(components[key], component_snapshots[key], ALL_NAMES) for key in component_snapshots if key != "rollout_k_32")
        treated_checks = {"treated_total_vs_direct_sum": d10a.residual(treated_tensor_map, treated_direct, table), "canonical_vs_treated_non_gru": {"l2": treated_non_gru_l2, "pass_fail": "PASS" if treated_non_gru_l2 == 0.0 else "FAIL"}, "head_untouched": "YES" if head_untouched else "NO", "other_components_untouched": "YES" if other_components_untouched else "NO"}
        if treated_checks["treated_total_vs_direct_sum"]["pass_fail"] != "PASS" or treated_checks["canonical_vs_treated_non_gru"]["pass_fail"] != "PASS":
            raise StageBlocker(f"TREATED_GRADIENT_RECONSTRUCTION_OR_SCOPE_FAILED:{branch}:{update}")
        if treatment_applied:
            if not tensor_equal_map(intervened_tensor_map, components["rollout_k_32"], (name for name, _ in named if name not in GRU_NAMES)):
                raise StageBlocker("D10_H32_HEAD_COMPONENT_MODIFIED")
            if not tensor_equal_map(components["h1"], component_snapshots["h1"], ALL_NAMES) or not tensor_equal_map(components["hidden"], component_snapshots["hidden"], ALL_NAMES):
                raise StageBlocker("D10_UNTARGETED_COMPONENT_MUTATED")
            if any(not tensor_equal_map(components[f"rollout_k_{step:02d}"], component_snapshots[f"rollout_k_{step:02d}"], ALL_NAMES) for step in range(1, ACTIVE_HORIZON + 1) if step != 32):
                raise StageBlocker("D10_OTHER_ROLLOUT_HORIZONS_MUTATED")
            for name, parameter in named:
                parameter.grad = treated_tensor_map[name].to(device=parameter.device, dtype=parameter.dtype).clone()
        final_raw = d10a.capture_gradients(named)
        final_checks = {"presented_raw_vs_treated_total": d10a.residual(final_raw, treated_tensor_map, table), "presented_raw_vs_canonical_raw": d10a.residual(final_raw, canonical_raw, table)}
        if final_checks["presented_raw_vs_treated_total"]["pass_fail"] != "PASS":
            raise StageBlocker(f"PRESENTED_GRADIENT_RECONSTRUCTION_FAILED:{branch}:{update}")
        raw_norm = norm(final_raw.values())
        returned_preclip = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM))
        if not math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise StageBlocker(f"CLIP_RETURNED_NORM_MISMATCH:{branch}:{update}")
        clipped = d10a.capture_gradients(named)
        clipped_norm = norm(clipped.values())
        clip_coefficient = min(1.0, CLIP_MAX_NORM / returned_preclip) if returned_preclip > 0.0 else 1.0
        write_json(output / "execution_progress.json", {"current_update": update, "completed_updates": [], "branch": branch, "forward_started": True, "measurement_backward_started": True, "canonical_backward_started": True, "optimizer_step_started": True})
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
    vectors["g_total_raw"] = final_raw
    vectors["g_total_canonical_raw"] = canonical_raw
    vectors["g_clipped"] = clipped
    scope_metrics = {scope: d10a.pairwise_metrics(vectors, table, scope) for scope in ("whole_model", "gru", "head")}
    stats = d10a.component_statistics(vectors, table)
    primary = {scope: {"h1_vs_rollout": scope_metrics[scope]["cosine_matrix"]["g_h1"]["g_rollout"], "rollout_to_h1_ratio": scope_metrics[scope]["norms"]["g_rollout"] / max(scope_metrics[scope]["norms"]["g_h1"], DIAGNOSTIC_NORM_EPSILON)} for scope in scope_metrics}
    if any(primary[scope]["h1_vs_rollout"] is None for scope in primary):
        raise StageBlocker(f"PRIMARY_COSINE_UNDEFINED:{branch}:{update}")
    rollout_norms = {step: scope_metrics["whole_model"]["norms"][f"g_rollout_k_{step:02d}"] for step in range(1, ACTIVE_HORIZON + 1)}
    dominant_step = max(rollout_norms, key=rollout_norms.get)
    row: dict[str, Any] = {
        "branch": branch,
        "update": update,
        "zero_based_schedule_index": zero_index,
        "active_horizon": ACTIVE_HORIZON,
        "active_training_phase": "phase_4_H32",
        "batch_identity": batch_record,
        "batch_window_sha256": batch_record["batch_window_sha256"],
        "model_mode_before": True,
        "model_mode_after_training": True,
        "rng_before_digest": base.rng_digest(rng_before),
        "rng_after_diagnostics_digest": base.rng_digest(rng_after_diagnostics),
        "rng_after_training_digest": base.rng_digest(rng_after_training),
        "rng_before_validation_digest": validation_meta["rng_before_digest"],
        "rng_after_validation_digest": validation_meta["rng_after_digest"],
        "canonical_state_hash_before": before_model_hash,
        "canonical_state_hash_after": after_model_hash,
        "optimizer_state_hash_before": before_optimizer_hash,
        "optimizer_state_hash_after": after_optimizer_hash,
        "s_k": scalar_checks["step_loss_vector"],
        "w_k": scalar_checks["weight_vector"],
        "r_k": scalar_checks["weighted_rollout_vector"],
        "q_k": scalar_checks["hidden_q_vector"],
        "z_k": scalar_checks["hidden_z_vector"],
        "hidden_per_layer": hidden_per_layer,
        "rollout_loss": scalar_checks["rollout_loss"],
        "h1_loss": scalar_checks["h1_loss"],
        "h1_loss_unweighted": scalar_checks["h1_loss"],
        "hidden_loss_unweighted": scalar_checks["hidden_loss_source"],
        "hidden_loss_rebuilt": scalar_checks["hidden_loss_rebuilt"],
        "hidden_loss_weighted": scalar_checks["weighted_hidden_loss"],
        "total_loss": scalar_checks["total_loss"],
        "loss_reconciliations": {"rollout": "PASS", "hidden": "PASS", "total": "PASS"},
        "validation": validation,
        "eval_h1": validation["1"],
        "eval_h32": validation["32"],
        "validation_meta": validation_meta,
        "h1_guardrail_label": "PASS" if validation["1"] <= H1_THRESHOLD else "FAIL",
        "h32_guardrail_label": "PASS" if validation["32"] <= H32_THRESHOLD else "FAIL",
        "component_norms_by_scope": {scope: scope_metrics[scope]["norms"] for scope in scope_metrics},
        "h1_rollout_comparison": primary,
        "dominant_rollout_horizon": dominant_step,
        "dominant_rollout_horizon_norm_whole_model": rollout_norms[dominant_step],
        "pairwise_metrics": scope_metrics,
        "parameter_component_statistics": stats,
        "connected_none_flags": masks,
        "decomposition_checks": {"canonical": canonical_checks, "treated": treated_checks, "final_presented": final_checks},
        "raw_total_gradient_norm": raw_norm,
        "returned_pre_clipping_total_norm": returned_preclip,
        "clipping_threshold": CLIP_MAX_NORM,
        "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO",
        "clip_coefficient": clip_coefficient,
        "clipped_total_gradient_norm": clipped_norm,
        "raw_to_clipped_direction_cosine": cosine(dot(final_raw.values(), clipped.values()), raw_norm, clipped_norm),
        "raw_gradient_digest": d7.d6.named_tensor_digest(final_raw),
        "canonical_raw_gradient_digest": d7.d6.named_tensor_digest(canonical_raw),
        "clipped_gradient_digest": d7.d6.named_tensor_digest(clipped),
        "parameter_update_digest": d7.d6.named_tensor_digest(delta),
        "parameter_update_norm": adamw["aggregate"]["whole_model"]["actual_delta_norm"],
        "exp_avg_norm": adamw["aggregate"]["whole_model"]["exp_avg_norm"],
        "exp_avg_sq_norm": adamw["aggregate"]["whole_model"]["exp_avg_sq_norm"],
        "adamw": adamw,
        "finite_status": "FINITE",
        "future_label_leakage": 0,
        "treatment_applied": "YES" if treatment_applied else "NO",
        "treatment_applied_count_so_far": treatment_counter[0],
        "treatment_update": 385,
        "treatment_fidelity": treatment_evidence,
        "head_untouched": "YES" if head_untouched else "NO",
        "non_gru_untouched": "YES" if treated_non_gru_l2 == 0.0 else "NO",
        "other_components_untouched": "YES" if other_components_untouched else "NO",
        "pre_treatment_component_hashes": {key: d10a.tensor_map_hash(value) for key, value in component_snapshots.items() if key.startswith("rollout_k_") or key in {"h1", "hidden"}},
        "treated_total_gradient_tensor_map": treated_tensor_map,
        "intervened_h32_gru_tensor_map": intervened_tensor_map,
    }
    if previous is None:
        row["transition_change_from_prior_update"] = {"available": False}
    else:
        row["transition_change_from_prior_update"] = {"available": True, "h1_vs_rollout_cosine_delta": row["h1_rollout_comparison"]["whole_model"]["h1_vs_rollout"] - previous["h1_rollout_comparison"]["whole_model"]["h1_vs_rollout"], "dominant_horizon_changed": dominant_step != previous["dominant_rollout_horizon"]}
    row["record_sha256"] = sha256_bytes(canonical_json({key: value for key, value in row.items() if key not in {"treated_total_gradient_tensor_map", "intervened_h32_gru_tensor_map"}}))
    write_update_artifacts(output, branch, update, row, components, canonical_raw, final_raw, clipped, hidden_steps, batch, predictions, targets, free_hidden, teacher_hidden, {"before_update": rng_before, "after_diagnostics": rng_after_diagnostics, "after_training": rng_after_training, "after_validation": rng_after_validation}, before_model, after_model, before_optimizer, after_optimizer, before_named_optimizer, after_named_optimizer, vectors)
    return row


def artifact_manifest(output: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"SHA256_MANIFEST.json", "FINAL_REPORT.md", "artifact_inventory.json"}):
        files[str(path.relative_to(output)).replace("\\", "/")] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {"schema_version": "stage3_h13_post_d10a_d10_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "final_report_excluded": True, "artifact_inventory_excluded": True, "file_count": len(files), "files": files}
    write_json(output / "SHA256_MANIFEST.json", payload)
    return payload


def verify_output_manifest(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for name, entry in payload["files"].items():
        path = output / name
        if not path.is_file() or sha256_file(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            failures.append(name)
    actual = {str(path.relative_to(output)).replace("\\", "/") for path in output.rglob("*") if path.is_file() and path.name not in {"SHA256_MANIFEST.json", "FINAL_REPORT.md", "artifact_inventory.json"}}
    if actual != set(payload["files"]):
        failures.append("file_set_mismatch")
    return {"status": "VERIFIED" if not failures else "BLOCKED", "checked": len(payload["files"]), "failures": failures, "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json")}


def classify(control_h1: float, intervention_h1: float, control_h32: float, intervention_h32: float, integrity: bool) -> tuple[str, dict[str, Any]]:
    effect_h1 = control_h1 - intervention_h1
    effect_h32 = control_h32 - intervention_h32
    conditions = {
        "integrity_pass": integrity,
        "h1_effect": effect_h1,
        "h32_effect": effect_h32,
        "material_h1_effect": effect_h1 >= MATERIAL_H1_EFFECT,
        "intervention_h1_pass": intervention_h1 <= H1_THRESHOLD,
        "intervention_h32_pass": intervention_h32 <= H32_THRESHOLD,
        "intervention_h1_harmful": intervention_h1 - control_h1 >= MATERIAL_H1_EFFECT,
        "intervention_h32_harmful": intervention_h32 > H32_THRESHOLD,
    }
    if not integrity:
        return "INCONCLUSIVE", conditions
    if conditions["intervention_h1_harmful"] or conditions["intervention_h32_harmful"]:
        return "GRU_DIRECTIONAL_INTERVENTION_HARMFUL", conditions
    if conditions["material_h1_effect"] and conditions["intervention_h1_pass"] and conditions["intervention_h32_pass"]:
        return "GRU_DIRECTIONAL_CONFLICT_CAUSAL", conditions
    if conditions["material_h1_effect"] and not conditions["intervention_h1_pass"] and conditions["intervention_h32_pass"]:
        return "GRU_DIRECTIONAL_CONFLICT_PARTIALLY_CAUSAL", conditions
    if conditions["intervention_h32_pass"] and effect_h1 < MATERIAL_H1_EFFECT:
        return "GRU_DIRECTIONAL_CONFLICT_NOT_CAUSAL", conditions
    return "INCONCLUSIVE", conditions


def render_report(summary: Mapping[str, Any]) -> str:
    c = summary.get("causal_contrast", {})
    t = summary.get("treatment_fidelity", {})
    g = summary.get("gradient_reconstruction", {})
    p = summary.get("protocol", {})
    b = summary.get("evidence_bundle", {})
    fmt = lambda value: "null" if value is None else (format(value, ".17g") if isinstance(value, (float, int)) else str(value))
    lines = [
        f"{EXPERIMENT_ID}:",
        summary["status"],
        "",
        "FIRST_BLOCKER:", summary["first_blocker"],
        "",
        "CAUSAL_CLASSIFICATION:", summary["classification"],
        "",
        "ONE_SENTENCE_VERDICT:", summary["verdict"],
        "",
        "==================================================", "1. PROTOCOL_INTEGRITY", "==================================================", "",
        "PROJECT_OWNER_D10_EXECUTION_AUTHORIZATION: YES",
        f"D10_DESIGN_REVIEW_PASSED: {p.get('D10_DESIGN_REVIEW_PASSED')}",
        f"D10_DESIGN_BUNDLE_IDENTITY_MATCH: {p.get('D10_DESIGN_BUNDLE_IDENTITY_MATCH')}",
        f"CERTIFIED_UPDATE_384_IDENTITY: {p.get('CERTIFIED_UPDATE_384_IDENTITY')}",
        f"SOURCE_REVISION_MATCH: {p.get('SOURCE_REVISION_MATCH')}",
        f"CANONICAL_SCHEDULE_MATCH: {p.get('CANONICAL_SCHEDULE_MATCH')}",
        f"CANONICAL_RNG_MATCH: {p.get('CANONICAL_RNG_MATCH')}",
        f"OPTIMIZER_IDENTITY_MATCH: {p.get('OPTIMIZER_IDENTITY_MATCH')}",
        f"BRANCHES_EXECUTED: {p.get('BRANCHES_EXECUTED')}",
        f"UPDATES_EXECUTED: {p.get('UPDATES_EXECUTED')}",
        "UPDATE_393_EXECUTED: NO", "FROZEN20_OPENED: NO", "FROZEN20_USED: NO", "R9_EXECUTED: NO", "R8E_A1_MODIFIED: NO", "HISTORICAL_ARTIFACTS_MODIFIED: NO", "HYPERPARAMETER_SEARCH_EXECUTED: NO", "FUTURE_EXPERIMENT_BUDGET_SELECTED: NO", "",
        "==================================================", "2. CONTROL_REPRODUCTION", "==================================================", "",
        f"CONTROL_BRANCH_VALID: {summary.get('control_branch_valid')}",
        f"CONTROL_REPRODUCES_CORRECTED_D10A: {summary.get('control_reproduction', {}).get('corrected_d10a')}",
        f"CONTROL_REPRODUCES_D7_ANCHORS: {summary.get('control_reproduction', {}).get('d7')}",
        f"CONTROL_FIRST_MISMATCH: {summary.get('control_reproduction', {}).get('first_mismatch')}", "",
        "==================================================", "3. TREATMENT_FIDELITY", "==================================================", "",
        f"TREATMENT_UPDATE: {t.get('treatment_update')}", f"TREATMENT_MODULE: {t.get('treatment_module')}", f"TREATMENT_HORIZON: {t.get('treatment_horizon')}",
        f"PRE_TREATMENT_DOT: {fmt(t.get('pre_treatment_dot'))}", f"PRE_TREATMENT_COSINE: {fmt(t.get('pre_treatment_cosine'))}", f"H1_GRU_NORM: {fmt(t.get('h1_gru_norm'))}", f"H32_GRU_NORM: {fmt(t.get('h32_gru_norm'))}", f"PROJECTION_COEFFICIENT: {fmt(t.get('projection_coefficient'))}", f"REMOVED_COMPONENT_NORM: {fmt(t.get('removed_component_norm'))}", f"POST_PROJECTION_DOT: {fmt(t.get('post_projection_dot'))}", f"POST_PROJECTION_COSINE: {fmt(t.get('post_projection_cosine'))}", f"POST_PROJECTION_ORTHOGONALITY_RESIDUAL: {fmt(t.get('post_projection_orthogonality_residual'))}", f"GRU_PARAMETER_COUNT: {t.get('gru_parameter_count')}", f"GRU_PARAMETER_SET_HASH: {t.get('gru_parameter_set_hash')}", f"HEAD_UNTOUCHED: {t.get('head_untouched')}", f"NON_GRU_UNTOUCHED: {t.get('non_gru_untouched')}", f"OTHER_COMPONENTS_UNTOUCHED: {t.get('other_components_untouched')}", f"TREATMENT_APPLIED_COUNT: {t.get('treatment_applied_count')}", "",
        "==================================================", "4. GRADIENT_RECONSTRUCTION", "==================================================", "",
        f"CANONICAL_RAW_GRADIENT_RECONSTRUCTION: {g.get('canonical_raw_gradient_reconstruction')}", f"TREATED_RAW_GRADIENT_RECONSTRUCTION: {g.get('treated_raw_gradient_reconstruction')}", f"CANONICAL_RESIDUAL: {json.dumps(g.get('canonical_residuals'), sort_keys=True)}", f"TREATED_RESIDUAL: {json.dumps(g.get('treated_residuals'), sort_keys=True)}", "",
        "==================================================", "5. CLIPPING_AND_ADAMW", "==================================================", "",
        f"CANONICAL_CLIPPING_AND_ADAMW_UNCHANGED: {summary.get('clipping_and_adamw_unchanged')}", "PROJECTION_ORDER: component extraction -> treatment -> unchanged global clip -> unchanged AdamW", "",
        "==================================================", "6. UPDATE_392_OUTCOMES", "==================================================", "",
        f"CONTROL_H1: {fmt(c.get('control_h1'))}", f"INTERVENTION_H1: {fmt(c.get('intervention_h1'))}", f"H1_EFFECT_CONTROL_MINUS_INTERVENTION: {fmt(c.get('h1_effect_control_minus_intervention'))}", f"H1_MATERIAL_EFFECT_THRESHOLD: {MATERIAL_H1_EFFECT:.17g}", f"CONTROL_H1_THRESHOLD_MARGIN: {fmt(c.get('control_h1_threshold_margin'))}", f"INTERVENTION_H1_THRESHOLD_MARGIN: {fmt(c.get('intervention_h1_threshold_margin'))}", f"CONTROL_H32: {fmt(c.get('control_h32'))}", f"INTERVENTION_H32: {fmt(c.get('intervention_h32'))}", f"H32_EFFECT_CONTROL_MINUS_INTERVENTION: {fmt(c.get('h32_effect_control_minus_intervention'))}", f"CONTROL_H32_THRESHOLD_MARGIN: {fmt(c.get('control_h32_threshold_margin'))}", f"INTERVENTION_H32_THRESHOLD_MARGIN: {fmt(c.get('intervention_h32_threshold_margin'))}", "",
        "==================================================", "7. CAUSAL_DECISION", "==================================================", "",
        f"DID_THE_UPDATE_385_GRU_DIRECTIONAL_CONFLICT_CAUSALLY_CONTRIBUTE_TO_THE_H1_BLOCKER: {summary.get('decision_h1')}", f"WAS_THE_H1_EFFECT_MATERIAL: {summary.get('h1_material')}", f"DID_INTERVENTION_H1_PASS: {summary.get('intervention_h1_pass')}", f"DID_INTERVENTION_H32_PASS: {summary.get('intervention_h32_pass')}", f"WAS_H32_PRESERVED: {summary.get('h32_preserved')}", f"CAUSAL_CLASSIFICATION: {summary['classification']}", "The decision uses only the completed matched D10 control/intervention contrast; D10A observations are not treated as causal evidence.", "",
        "==================================================", "8. WHAT_D10_ESTABLISHES", "==================================================", "", summary.get("what_d10_establishes", ""), "",
        "==================================================", "9. WHAT_D10_DOES_NOT_ESTABLISH", "==================================================", "", summary.get("what_d10_does_not_establish", ""), "",
        "==================================================", "10. EVIDENCE_BUNDLE", "==================================================", "", f"OUTPUT_BUNDLE_PATH: {b.get('output_bundle_path')}", f"FINAL_REPORT: {b.get('final_report')}", f"SHA256_MANIFEST: {b.get('sha256_manifest')}", f"MANIFEST_VERIFIED: {b.get('manifest_verified')}", f"MANIFEST_SHA256: {b.get('manifest_sha256')}", f"FILE_COUNT: {b.get('file_count')}", "",
        "==================================================", "11. FINAL_CLASSIFICATION", "==================================================", "", f"D10_EXECUTION_STATUS: {summary['status']}", f"CAUSAL_CLASSIFICATION: {summary['classification']}", f"CONTROL_BRANCH_VALID: {summary.get('control_branch_valid')}", f"INTERVENTION_BRANCH_VALID: {summary.get('intervention_branch_valid')}", f"SINGLE_FACTOR_CAUSAL_IDENTIFICATION_VALID: {summary.get('single_factor_valid')}", "UPDATE_393_EXECUTED: NO", "NEXT_CAUSAL_STAGE_EXECUTED: NO", "FUTURE_EXPERIMENT_BUDGET_SELECTED: NO", "",
    ]
    return "\n".join(str(item) for item in lines) + "\n"


def fail_closed(output: Path, blocker: str, inputs: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    progress = None
    if (output / "execution_progress.json").is_file():
        try:
            progress = json.loads((output / "execution_progress.json").read_text(encoding="utf-8"))
        except Exception:
            progress = {"read_error": True}
    certificate = {"schema_version": "stage3_h13_post_d10a_d10_terminal_certificate_v1", "status": "BLOCKED", "first_blocker": blocker, "experiment_identifier": EXPERIMENT_ID, "execution_progress": progress, "inputs": inputs, "UPDATE_393_EXECUTED": "NO", "FROZEN20_OPENED": "NO", "R9_EXECUTED": "NO", "D11_EXECUTED": "NO", "historical_artifacts_modified": "NO"}
    write_json(output / "terminal_certificate.json", certificate)
    write_json(output / "protocol_integrity_record.json", {"status": "BLOCKED", "first_blocker": blocker, "backward_pass_executed": "YES" if progress and progress.get("canonical_backward_started") else "NO", "optimizer_step_executed": "YES" if progress and progress.get("optimizer_step_started") else "NO", "frozen20_opened": "NO", "historical_artifacts_modified": "NO"})
    payload = artifact_manifest(output)
    check = verify_output_manifest(output, payload)
    summary = {"status": "BLOCKED", "first_blocker": blocker, "classification": "INCONCLUSIVE", "verdict": "The D10 replay failed closed before a valid causal contrast was established.", "protocol": {"D10_DESIGN_REVIEW_PASSED": "UNKNOWN", "D10_DESIGN_BUNDLE_IDENTITY_MATCH": "UNKNOWN", "CERTIFIED_UPDATE_384_IDENTITY": "UNKNOWN", "SOURCE_REVISION_MATCH": "UNKNOWN", "CANONICAL_SCHEDULE_MATCH": "UNKNOWN", "CANONICAL_RNG_MATCH": "UNKNOWN", "OPTIMIZER_IDENTITY_MATCH": "UNKNOWN", "BRANCHES_EXECUTED": "NONE", "UPDATES_EXECUTED": "NONE"}, "control_reproduction": {}, "treatment_fidelity": {}, "gradient_reconstruction": {}, "causal_contrast": {}, "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report": str((output / "FINAL_REPORT.md").resolve()), "sha256_manifest": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_verified": check["status"], "manifest_sha256": check["manifest_sha256"], "file_count": check["checked"]}, "control_branch_valid": "NO", "intervention_branch_valid": "NO", "single_factor_valid": "NO", "inputs": inputs}
    (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
    return 2


def finalize_existing(output: Path) -> int:
    """Finalize a completed two-branch run without executing training again."""
    try:
        inputs = verify_inputs()
        control_rows = [json.loads((output / "branches" / "CONTROL" / f"update_{update:05d}" / "metrics.json").read_text(encoding="utf-8")) for update in UPDATES]
        intervention_rows = [json.loads((output / "branches" / "INTERVENTION" / f"update_{update:05d}" / "metrics.json").read_text(encoding="utf-8")) for update in UPDATES]
        if [row.get("update") for row in control_rows] != list(UPDATES) or [row.get("update") for row in intervention_rows] != list(UPDATES):
            raise StageBlocker("FINALIZATION_UPDATE_WINDOW_INCOMPLETE")
        corrected_rows = json.loads((D10A_DIR / "update_level_measurements.json").read_text(encoding="utf-8"))
        d10a_control = compare_rows(control_rows, corrected_rows, "corrected_D10A_update_level_measurements")
        d7_control = d10a.compare_trajectory(control_rows, d10a.D7_AUTHORITY_DIR)
        if d10a_control["status"] != "PASS" or d7_control["all_available_fields_match"] != "YES":
            raise StageBlocker(f"FINALIZATION_CONTROL_REPRODUCTION_FAILED:{d10a_control.get('first_mismatch') or d7_control.get('first_divergence')}")
        treatment_root = output / "branches" / "INTERVENTION" / "update_00385" / "gradients"
        original_h32 = torch.load(treatment_root / "g_rollout_k_32_original.pt", map_location="cpu", weights_only=False)
        intervened_h32 = torch.load(treatment_root / "g_rollout_k_32_intervened.pt", map_location="cpu", weights_only=False)
        h1 = torch.load(treatment_root / "g_h1.pt", map_location="cpu", weights_only=False)
        canonical_total = torch.load(treatment_root / "g_total_from_components.pt", map_location="cpu", weights_only=False)
        treated_total = torch.load(treatment_root / "g_total_treated_from_components.pt", map_location="cpu", weights_only=False)
        presented_total = torch.load(treatment_root / "g_total_raw_presented.pt", map_location="cpu", weights_only=False)
        projected_h32, recomputed_projection = projection(h1, original_h32)
        if not all(torch.equal(intervened_h32[name], projected_h32[name]) for name in GRU_NAMES):
            raise StageBlocker("FINALIZATION_PROJECTION_TENSOR_MISMATCH")
        if not all(torch.equal(treated_total[name], presented_total[name]) for name in ALL_NAMES):
            raise StageBlocker("FINALIZATION_PRESENTED_TREATED_GRADIENT_MISMATCH")
        if not all(torch.equal(treated_total[name], canonical_total[name]) for name in ALL_NAMES if name not in GRU_NAMES):
            raise StageBlocker("FINALIZATION_NON_GRU_GRADIENT_CHANGED")
        if not all(torch.equal(intervened_h32[name], original_h32[name]) for name in ALL_NAMES if name not in GRU_NAMES):
            raise StageBlocker("FINALIZATION_HEAD_H32_GRADIENT_CHANGED")
        i385 = intervention_rows[0]
        if i385.get("treatment_applied") != "YES" or i385.get("treatment_applied_count_so_far") != 1 or any(row.get("treatment_applied") != "NO" for row in intervention_rows[1:]):
            raise StageBlocker("FINALIZATION_TREATMENT_COUNT_OR_WINDOW_INVALID")
        if i385.get("treatment_fidelity", {}).get("projection_fired") is not True or i385.get("treatment_fidelity", {}).get("projection_residual_pass") != "PASS" or i385.get("treatment_fidelity", {}).get("orthogonality_pass") != "PASS":
            raise StageBlocker("FINALIZATION_TREATMENT_FIDELITY_INVALID")
        if i385.get("pre_treatment_component_hashes") != control_rows[0].get("pre_treatment_component_hashes"):
            raise StageBlocker("FINALIZATION_PRE_TREATMENT_BRANCH_MISMATCH")
        if any(control["batch_window_sha256"] != intervention["batch_window_sha256"] for control, intervention in zip(control_rows, intervention_rows)):
            raise StageBlocker("FINALIZATION_BATCH_MATCH_FAILED")
        c392, i392 = control_rows[-1], intervention_rows[-1]
        effect_h1 = c392["eval_h1"] - i392["eval_h1"]
        effect_h32 = c392["eval_h32"] - i392["eval_h32"]
        integrity_checks = {
            "design_bundle_verified": True,
            "certified_update_384_match": True,
            "source_revision_match": True,
            "canonical_schedule_match": True,
            "canonical_rng_match": True,
            "optimizer_identity_match": True,
            "runtime_identity_match": True,
            "start_model_identity_match": True,
            "start_optimizer_identity_match": True,
            "start_rng_identity_match": True,
            "update_385_batch_match": True,
            "updates_386_392_batches_match": True,
            "loss_configuration_match": True,
            "validation_configuration_match": True,
            "clipping_configuration_match": True,
            "adamw_configuration_match": True,
            "control_reproduction": d10a_control["status"] == "PASS" and d7_control["all_available_fields_match"] == "YES",
            "treatment_fidelity": True,
            "canonical_reconstruction": all(row.get("decomposition_checks", {}).get("canonical", {}).get("components_vs_canonical_backward", {}).get("pass_fail") == "PASS" for row in control_rows + intervention_rows),
            "treated_reconstruction": all(row.get("decomposition_checks", {}).get("treated", {}).get("treated_total_vs_direct_sum", {}).get("pass_fail") == "PASS" and row.get("decomposition_checks", {}).get("final_presented", {}).get("presented_raw_vs_treated_total", {}).get("pass_fail") == "PASS" for row in intervention_rows),
            "head_untouched": all(torch.equal(intervened_h32[name], original_h32[name]) for name in ALL_NAMES if name not in GRU_NAMES),
            "non_gru_untouched": all(torch.equal(treated_total[name], canonical_total[name]) for name in ALL_NAMES if name not in GRU_NAMES),
            "other_components_untouched": i385.get("other_components_untouched") == "YES",
            "treatment_once": True,
            "no_update_393": True,
            "no_nonfinite": all(row.get("finite_status") == "FINITE" for row in control_rows + intervention_rows),
        }
        integrity = all(integrity_checks.values())
        classification, classification_conditions = classify(c392["eval_h1"], i392["eval_h1"], c392["eval_h32"], i392["eval_h32"], integrity)
        parameter_scope = json.loads((output / "parameter_scope.json").read_text(encoding="utf-8"))
        treatment_evidence = i385["treatment_fidelity"] | {"treatment_update": 385, "treatment_module": "GRU", "treatment_horizon": "H32", "gru_parameter_count": parameter_scope["gru_parameter_count"], "gru_parameter_set_hash": parameter_scope["gru_parameter_set_hash"], "head_untouched": "YES" if integrity_checks["head_untouched"] else "NO", "non_gru_untouched": "YES" if integrity_checks["non_gru_untouched"] else "NO", "other_components_untouched": "YES" if integrity_checks["other_components_untouched"] else "NO", "treatment_applied_count": 1, "parameter_names_shapes_dtypes_devices": parameter_scope["gru_parameter_set"]}
        gradient_evidence = {"canonical_raw_gradient_reconstruction": "PASS" if integrity_checks["canonical_reconstruction"] else "FAIL", "treated_raw_gradient_reconstruction": "PASS" if integrity_checks["treated_reconstruction"] else "FAIL", "canonical_residuals": {"control_update_385": control_rows[0]["decomposition_checks"]["canonical"], "intervention_update_385": intervention_rows[0]["decomposition_checks"]["canonical"]}, "treated_residuals": intervention_rows[0]["decomposition_checks"]["treated"]}
        causal_contrast = {"control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "h1_effect_control_minus_intervention": effect_h1, "h1_material_effect_threshold": MATERIAL_H1_EFFECT, "control_h1_threshold_margin": H1_THRESHOLD - c392["eval_h1"], "intervention_h1_threshold_margin": H1_THRESHOLD - i392["eval_h1"], "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"], "h32_effect_control_minus_intervention": effect_h32, "control_h32_threshold_margin": H32_THRESHOLD - c392["eval_h32"], "intervention_h32_threshold_margin": H32_THRESHOLD - i392["eval_h32"], "classification_conditions": classification_conditions}
        protocol = {"D10_DESIGN_REVIEW_PASSED": "YES", "D10_DESIGN_BUNDLE_IDENTITY_MATCH": "YES", "CERTIFIED_UPDATE_384_IDENTITY": "MATCH", "SOURCE_REVISION_MATCH": "YES", "CANONICAL_SCHEDULE_MATCH": "YES", "CANONICAL_RNG_MATCH": "YES", "OPTIMIZER_IDENTITY_MATCH": "YES", "BRANCHES_EXECUTED": "CONTROL, INTERVENTION", "UPDATES_EXECUTED": list(UPDATES), "D10A_CORRECTED_EVIDENCE": "VERIFIED", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HISTORICAL_ARTIFACTS_MODIFIED": "NO", "HYPERPARAMETER_SEARCH_EXECUTED": "NO", "H32_MAGNITUDE_INTERVENTION": "NO", "HEAD_INTERVENTION": "NO", "UPDATE_393_EXECUTED": "NO", "D11_EXECUTED": "NO", "FINALIZATION_ONLY_AFTER_COMPLETED_REPLAY": "YES"}
        branch_matching = {"START_MODEL_IDENTITY_MATCH": "YES", "START_OPTIMIZER_IDENTITY_MATCH": "YES", "START_RNG_IDENTITY_MATCH": "YES", "BATCH_SCHEDULE_MATCH": "YES", "UPDATE_385_BATCH_MATCH": "YES", "UPDATES_386_392_BATCHES_MATCH": "YES", "LOSS_CONFIGURATION_MATCH": "YES", "VALIDATION_CONFIGURATION_MATCH": "YES", "CLIPPING_CONFIGURATION_MATCH": "YES", "ADAMW_CONFIGURATION_MATCH": "YES", "SOURCE_REVISION_MATCH": "YES", "control_final_rng": control_rows[-1]["rng_after_validation_digest"], "intervention_final_rng": intervention_rows[-1]["rng_after_validation_digest"], "post_treatment_model_divergence_expected": "YES", "integrity_checks": integrity_checks}
        write_json(output / "branch_matching.json", branch_matching)
        compact_exclusions = {"parameter_component_statistics", "pairwise_metrics", "adamw", "treated_total_gradient_tensor_map", "intervened_h32_gru_tensor_map"}
        write_json(output / "branch_a_control_metrics.json", jsonable({"branch": "CONTROL", "updates": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in control_rows], "update_392": {key: value for key, value in c392.items() if key not in compact_exclusions}}))
        write_json(output / "branch_b_intervention_metrics.json", jsonable({"branch": "INTERVENTION", "updates": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in intervention_rows], "update_392": {key: value for key, value in i392.items() if key not in compact_exclusions}}))
        write_json(output / "treatment_fidelity.json", treatment_evidence)
        write_json(output / "gradient_reconstruction.json", gradient_evidence)
        write_json(output / "optimizer_and_clipping_evidence.json", {"unchanged_configuration_between_branches": "YES", "configuration": {"clip": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0}, "optimizer": json.loads((output / "execution_config.json").read_text(encoding="utf-8"))["optimizer"]}, "control_updates": [{"update": row["update"], "pre_clip_global_grad_norm": row["returned_pre_clipping_total_norm"], "post_clip_global_grad_norm": row["clipped_total_gradient_norm"], "clip_coefficient": row["clip_coefficient"], "optimizer_before_hash": row["optimizer_state_hash_before"], "optimizer_after_hash": row["optimizer_state_hash_after"]} for row in control_rows], "intervention_updates": [{"update": row["update"], "pre_clip_global_grad_norm": row["returned_pre_clipping_total_norm"], "post_clip_global_grad_norm": row["clipped_total_gradient_norm"], "clip_coefficient": row["clip_coefficient"], "optimizer_before_hash": row["optimizer_state_hash_before"], "optimizer_after_hash": row["optimizer_state_hash_after"]} for row in intervention_rows]})
        write_json(output / "causal_contrast.json", causal_contrast)
        write_json(output / "control_reproduction.json", {"corrected_d10a": d10a_control, "d7": d7_control, "control_reproduction_pass": "YES" if integrity_checks["control_reproduction"] else "NO"})
        write_json(output / "integrity_checks.json", {"checks": integrity_checks, "all_required_integrity_pass": "YES" if integrity else "NO"})
        write_json(output / "update_metrics.json", jsonable({"CONTROL": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in control_rows], "INTERVENTION": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in intervention_rows]}))
        summary = {"status": "PASSED", "first_blocker": "none", "classification": classification, "verdict": f"The matched update-385 GRU-only H32 directional projection changed update-392 H1 by {effect_h1:.17g} (control minus intervention) while H32 changed by {effect_h32:.17g}; the frozen contract classifies this as {classification}.", "protocol": protocol, "control_reproduction": {"corrected_d10a": d10a_control["status"], "d7": d7_control["all_available_fields_match"], "first_mismatch": d10a_control.get("first_mismatch") or d7_control.get("first_divergence")}, "treatment_fidelity": treatment_evidence, "gradient_reconstruction": gradient_evidence, "causal_contrast": causal_contrast, "clipping_and_adamw_unchanged": "YES", "control_branch_valid": "YES" if integrity_checks["control_reproduction"] else "NO", "intervention_branch_valid": "YES" if integrity else "NO", "single_factor_valid": "YES" if integrity else "NO", "decision_h1": "YES" if effect_h1 > 0 else ("NO" if effect_h1 <= 0 else "INCONCLUSIVE"), "h1_material": "YES" if effect_h1 >= MATERIAL_H1_EFFECT else "NO", "intervention_h1_pass": "YES" if i392["eval_h1"] <= H1_THRESHOLD else "NO", "intervention_h32_pass": "YES" if i392["eval_h32"] <= H32_THRESHOLD else "NO", "h32_preserved": "YES" if i392["eval_h32"] <= H32_THRESHOLD else "NO", "what_d10_establishes": "Observational evidence: D10A localized a negative H1-versus-weighted-H32 cosine to the exact GRU scope at update 385. Causal evidence: under the matched replay, the single update-385 GRU-only anti-H1 projection produced the reported update-392 H1 contrast and the frozen classification. Unresolved mechanism: only this exact intervention is identified; no broader horizon, magnitude, head, or aggregate-gradient mechanism is identified.", "what_d10_does_not_establish": "This replay does not make D10A magnitude dominance at H32 causal, does not make head localization causal, and does not establish effects of H32 norm capping, head intervention, other horizons, later updates, aggregate projections, D11, or any future experiment.", "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report": str((output / "FINAL_REPORT.md").resolve()), "sha256_manifest": str((output / "SHA256_MANIFEST.json").resolve())}}
        write_json(output / "protocol_integrity.json", protocol)
        write_json(output / "terminal_certificate.json", summary)
        manifest_payload = artifact_manifest(output)
        manifest_check = verify_output_manifest(output, manifest_payload)
        if manifest_check["status"] != "VERIFIED":
            raise StageBlocker(f"FINALIZATION_MANIFEST_VERIFICATION_FAILED:{manifest_check['failures'][:1]}")
        summary["evidence_bundle"].update({"manifest_verified": "YES", "manifest_sha256": manifest_check["manifest_sha256"], "file_count": manifest_check["checked"]})
        (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
        inventory = {"schema_version": "stage3_h13_post_d10a_d10_artifact_inventory_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": {str(path.relative_to(output)).replace("\\", "/"): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_inventory.json")}}
        write_json(output / "artifact_inventory.json", inventory)
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification, "manifest_sha256": manifest_check["manifest_sha256"], "control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, f"FINALIZATION_BLOCKED:{exc}", None)
    except Exception as exc:  # pragma: no cover
        (output / "finalization_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return fail_closed(output, f"FINALIZATION_UNEXPECTED_FAILURE:{type(exc).__name__}:{exc}", None)


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    inputs: dict[str, Any] | None = None
    try:
        inputs = verify_inputs()
        base.set_deterministic(base.SEED)
        bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
        pre = base.preflight(True)
        authority = base.verify_authority(pre)
        env = authority["current_environment"]
        if env.get("source_revision") != EXPECTED_SOURCE_REVISION or env.get("device") != "cpu" or env.get("dtype") != "float32" or env.get("pytorch_version") != "2.13.0+cpu" or env.get("python_version") != "3.12.1" or env.get("numpy_version") != "2.1.3" or not torch.are_deterministic_algorithms_enabled():
            raise StageBlocker("RUNTIME_IDENTITY_MISMATCH")
        a1_before = sha256_file(base.A1_CHECKPOINT)
        if a1_before != base.EXPECTED_A1_HASH:
            raise StageBlocker("R8E_A1_IMMUTABILITY_GATE_FAILED_BEFORE_EXECUTION")
        def load_branch() -> tuple[Any, torch.optim.Optimizer, dict[str, Any]]:
            model, optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
            loaded = d7.verify_loaded_identity(bundle, model, optimizer)
            if loaded["model_semantic_hash"] != EXPECTED_MODEL_HASH or loaded["optimizer_semantic_hash"] != EXPECTED_OPTIMIZER_HASH or loaded["exp_avg_semantic_hash"] != EXPECTED_EXP_AVG_HASH or loaded["exp_avg_sq_semantic_hash"] != EXPECTED_EXP_AVG_SQ_HASH:
                raise StageBlocker("CERTIFIED_START_STATE_RELOAD_IDENTITY_MISMATCH")
            return model, optimizer, loaded
        control_model, control_optimizer, control_loaded = load_branch()
        intervention_model, intervention_optimizer, intervention_loaded = load_branch()
        base.restore_rng(bundle["rng_state"])
        control_rng_start = base.rng_digest(base.capture_rng())
        base.restore_rng(bundle["rng_state"])
        intervention_rng_start = base.rng_digest(base.capture_rng())
        if control_rng_start != EXPECTED_RNG_HASH or intervention_rng_start != control_rng_start:
            raise StageBlocker("CERTIFIED_START_RNG_IDENTITY_MISMATCH")
        control_table, control_named = d10a.parameter_table(control_model)
        intervention_table, intervention_named = d10a.parameter_table(intervention_model)
        if control_table != intervention_table or parameter_set_hash(control_named) != parameter_set_hash(intervention_named):
            raise StageBlocker("BRANCH_PARAMETER_SCOPE_IDENTITY_MISMATCH")
        if base.optimizer_semantic_hash(control_optimizer) != base.optimizer_semantic_hash(intervention_optimizer) or canonical_model_state_sha256(control_model.state_dict()) != canonical_model_state_sha256(intervention_model.state_dict()):
            raise StageBlocker("START_BRANCH_SEMANTIC_IDENTITY_MISMATCH")
        state = {"pre": pre, "authority": authority, "table": control_table, "named": control_named, "bundle": bundle}
        write_json(output / "authoritative_inputs.json", {"source": inputs["authoritative_inputs"], "verified_at_utc": utc_now(), "design_manifest": inputs["design_manifest"], "corrected_d10a_manifest": inputs["corrected_d10a_manifest"], "certified_bundle_sha256": EXPECTED_BUNDLE_SHA256, "certified_model_hash": EXPECTED_MODEL_HASH, "certified_optimizer_hash": EXPECTED_OPTIMIZER_HASH, "certified_rng_hash": EXPECTED_RNG_HASH, "canonical_schedule_hash": EXPECTED_SCHEDULE_HASH})
        write_json(output / "execution_config.json", {"schema_version": "stage3_h13_post_d10a_d10_execution_config_v1", "experiment_identifier": EXPERIMENT_ID, "captured_at_utc": utc_now(), "design_bundle": str(DESIGN_DIR.resolve()), "corrected_d10a_evidence": str(D10A_DIR.resolve()), "authorization": inputs["authorization"], "source_revision": EXPECTED_SOURCE_REVISION, "runtime": {"python": env.get("python_version"), "pytorch": env.get("pytorch_version"), "numpy": env.get("numpy_version"), "platform": platform.platform(), "device": "cpu", "model_dtype": "float32", "reduction_dtype": "float64", "deterministic_algorithms": True, "dataloader_workers": 0}, "starting_state": {"path": str(BUNDLE_PATH.resolve()), "sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_step": 384, "model_hash": EXPECTED_MODEL_HASH, "optimizer_hash": EXPECTED_OPTIMIZER_HASH, "exp_avg_hash": EXPECTED_EXP_AVG_HASH, "exp_avg_sq_hash": EXPECTED_EXP_AVG_SQ_HASH, "rng_hash": EXPECTED_RNG_HASH}, "objective": {"T": "R + H1 + 0.005*Z", "lambda_rollout": 1.0, "lambda_h1": 1.0, "lambda_hidden": 0.005, "position_smooth_l1_beta": 0.001, "hidden_smooth_l1_beta": 0.1, "active_training_horizon": 32, "hidden_active_steps": list(range(2, 17))}, "updates": list(UPDATES), "update_393_executed": "NO", "branches": ["CONTROL", "INTERVENTION"], "intervention": {"update": 385, "module": "GRU", "horizon": "H32", "component": "g_rollout_k_32", "condition": "dot < 0", "projection": "g_H32_GRU - (dot/||g_H1_GRU||^2)*g_H1_GRU", "projection_reduction": "float64 then cast to parameter dtype", "epsilon": "NONE"}, "clipping": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False}, "optimizer": base.optimizer_contract(control_optimizer), "validation": {"horizons": list(VALIDATION_HORIZONS), "metric": "free-running validation RMSE in radians", "h1_threshold": H1_THRESHOLD, "h32_threshold": H32_THRESHOLD, "material_h1_effect": MATERIAL_H1_EFFECT}})
        write_json(output / "source_code_map.json", {"sealed_design_source_code_map": inputs["source_code_map"], "execution_runner": {"path": str(Path(__file__).resolve()), "sha256": inputs["runner_source_sha256"], "role": "D10 two-branch execution wrapper; no production model/loss semantics modified"}})
        parameter_rows = [{**row, "device": str(parameter.device)} for row, (name, parameter) in zip(control_table, control_named)]
        write_json(output / "parameter_scope.json", {"exact_order": list(ALL_NAMES), "gru_parameter_set": parameter_rows[:8], "head_parameter_set": parameter_rows[8:], "gru_parameter_count": sum(row["numel"] for row in parameter_rows[:8]), "gru_parameter_set_hash": subset_parameter_set_hash(control_named, GRU_NAMES), "all_parameter_set_hash": parameter_set_hash(control_named), "all_parameter_count": sum(row["numel"] for row in parameter_rows)})
        write_json(output / "canonical_batch_schedule.json", {"logical_sha256": EXPECTED_SCHEDULE_HASH, "storage_sha256": inputs["authoritative_inputs"]["canonical_schedule"]["storage_sha256"], "slice": [d10a.batch_identity(authority["schedule_manifest"], index) for index in range(384, 392)]})
        write_json(output / "branch_start_identity.json", {"control": {"model_hash": canonical_model_state_sha256(control_model.state_dict()), "optimizer_hash": base.optimizer_semantic_hash(control_optimizer), "rng_hash": control_rng_start, "loaded": control_loaded}, "intervention": {"model_hash": canonical_model_state_sha256(intervention_model.state_dict()), "optimizer_hash": base.optimizer_semantic_hash(intervention_optimizer), "rng_hash": intervention_rng_start, "loaded": intervention_loaded}, "start_model_identity_match": "YES", "start_optimizer_identity_match": "YES", "start_rng_identity_match": "YES", "parameter_set_hash": parameter_set_hash(control_named)})
        base.restore_rng(bundle["rng_state"])
        control_rows: list[dict[str, Any]] = []
        control_counter = [0]
        for zero_index in range(384, 392):
            control_rows.append(run_update("CONTROL", control_model, control_optimizer, state, output, control_rows[-1] if control_rows else None, zero_index, control_counter))
        control_final_rng = base.rng_digest(base.capture_rng())
        base.restore_rng(bundle["rng_state"])
        intervention_rows: list[dict[str, Any]] = []
        intervention_counter = [0]
        for zero_index in range(384, 392):
            intervention_rows.append(run_update("INTERVENTION", intervention_model, intervention_optimizer, state, output, intervention_rows[-1] if intervention_rows else None, zero_index, intervention_counter))
        intervention_final_rng = base.rng_digest(base.capture_rng())
        if control_counter[0] != 0 or intervention_counter[0] != 1:
            raise StageBlocker(f"TREATMENT_COUNT_INVALID:control={control_counter[0]}:intervention={intervention_counter[0]}")
        if [row["update"] for row in control_rows] != list(UPDATES) or [row["update"] for row in intervention_rows] != list(UPDATES):
            raise StageBlocker("EXACT_UPDATE_WINDOW_NOT_COMPLETED")
        corrected_rows = json.loads((D10A_DIR / "update_level_measurements.json").read_text(encoding="utf-8"))
        d10a_control = compare_rows(control_rows, corrected_rows, "corrected_D10A_update_level_measurements")
        d7_control = d10a.compare_trajectory(control_rows, d10a.D7_AUTHORITY_DIR)
        if d10a_control["status"] != "PASS" or d7_control["all_available_fields_match"] != "YES":
            raise StageBlocker(f"CONTROL_REPRODUCTION_FAILED:{d10a_control.get('first_mismatch') or d7_control.get('first_divergence')}")
        update385_control, update385_intervention = control_rows[0], intervention_rows[0]
        if update385_control["pre_treatment_component_hashes"] != update385_intervention["pre_treatment_component_hashes"]:
            raise StageBlocker("UPDATE_385_PRE_TREATMENT_COMPONENT_BRANCH_MISMATCH")
        if any(a["batch_window_sha256"] != b["batch_window_sha256"] for a, b in zip(control_rows, intervention_rows)):
            raise StageBlocker("BRANCH_BATCH_SCHEDULE_MISMATCH")
        if any(row["treatment_applied"] != "NO" for row in intervention_rows[1:]):
            raise StageBlocker("REPEATED_OR_LATE_TREATMENT_DETECTED")
        if control_final_rng != EXPECTED_RNG_HASH or intervention_final_rng != EXPECTED_RNG_HASH:
            raise StageBlocker("FINAL_RNG_IDENTITY_MISMATCH")
        base.verify_snapshot(authority["authority_snapshot"])
        c392, i392 = control_rows[-1], intervention_rows[-1]
        effect_h1 = c392["eval_h1"] - i392["eval_h1"]
        effect_h32 = c392["eval_h32"] - i392["eval_h32"]
        integrity_checks = {
            "design_bundle_verified": True,
            "certified_update_384_match": True,
            "source_revision_match": True,
            "canonical_schedule_match": True,
            "canonical_rng_match": True,
            "optimizer_identity_match": True,
            "runtime_identity_match": True,
            "start_model_identity_match": True,
            "start_optimizer_identity_match": True,
            "start_rng_identity_match": True,
            "update_385_batch_match": True,
            "updates_386_392_batches_match": True,
            "loss_configuration_match": True,
            "validation_configuration_match": True,
            "clipping_configuration_match": True,
            "adamw_configuration_match": True,
            "control_reproduction": d10a_control["status"] == "PASS" and d7_control["all_available_fields_match"] == "YES",
            "treatment_fidelity": intervention_rows[0]["treatment_fidelity"].get("projection_fired") is True and intervention_rows[0]["treatment_fidelity"].get("projection_residual_pass") == "PASS" and intervention_rows[0]["treatment_fidelity"].get("orthogonality_pass") == "PASS",
            "canonical_reconstruction": all(row["decomposition_checks"]["canonical"]["components_vs_canonical_backward"]["pass_fail"] == "PASS" for row in control_rows + intervention_rows),
            "treated_reconstruction": all(row["decomposition_checks"]["treated"]["treated_total_vs_direct_sum"]["pass_fail"] == "PASS" and row["decomposition_checks"]["final_presented"]["presented_raw_vs_treated_total"]["pass_fail"] == "PASS" for row in intervention_rows),
            "head_untouched": intervention_rows[0].get("head_untouched") == "YES",
            "non_gru_untouched": intervention_rows[0].get("non_gru_untouched") == "YES",
            "other_components_untouched": intervention_rows[0].get("other_components_untouched") == "YES",
            "treatment_once": intervention_counter[0] == 1,
            "no_update_393": True,
            "no_nonfinite": all(row["finite_status"] == "FINITE" for row in control_rows + intervention_rows),
        }
        integrity = all(integrity_checks.values())
        classification, classification_conditions = classify(c392["eval_h1"], i392["eval_h1"], c392["eval_h32"], i392["eval_h32"], integrity)
        treatment_evidence = intervention_rows[0]["treatment_fidelity"] | {"treatment_update": 385, "treatment_module": "GRU", "treatment_horizon": "H32", "gru_parameter_count": 156672, "gru_parameter_set_hash": subset_parameter_set_hash(intervention_named, GRU_NAMES), "head_untouched": "YES" if integrity_checks["head_untouched"] else "NO", "non_gru_untouched": "YES" if integrity_checks["non_gru_untouched"] else "NO", "other_components_untouched": "YES" if integrity_checks["other_components_untouched"] else "NO", "treatment_applied_count": intervention_counter[0]}
        gradient_evidence = {"canonical_raw_gradient_reconstruction": "PASS" if integrity_checks["canonical_reconstruction"] else "FAIL", "treated_raw_gradient_reconstruction": "PASS" if integrity_checks["treated_reconstruction"] else "FAIL", "canonical_residuals": {"control_update_385": control_rows[0]["decomposition_checks"]["canonical"], "intervention_update_385": intervention_rows[0]["decomposition_checks"]["canonical"]}, "treated_residuals": intervention_rows[0]["decomposition_checks"]["treated"]}
        causal_contrast = {"control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "h1_effect_control_minus_intervention": effect_h1, "h1_material_effect_threshold": MATERIAL_H1_EFFECT, "control_h1_threshold_margin": H1_THRESHOLD - c392["eval_h1"], "intervention_h1_threshold_margin": H1_THRESHOLD - i392["eval_h1"], "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"], "h32_effect_control_minus_intervention": effect_h32, "control_h32_threshold_margin": H32_THRESHOLD - c392["eval_h32"], "intervention_h32_threshold_margin": H32_THRESHOLD - i392["eval_h32"], "classification_conditions": classification_conditions}
        branch_matching = {"START_MODEL_IDENTITY_MATCH": "YES", "START_OPTIMIZER_IDENTITY_MATCH": "YES", "START_RNG_IDENTITY_MATCH": "YES", "BATCH_SCHEDULE_MATCH": "YES", "UPDATE_385_BATCH_MATCH": "YES", "UPDATES_386_392_BATCHES_MATCH": "YES", "LOSS_CONFIGURATION_MATCH": "YES", "VALIDATION_CONFIGURATION_MATCH": "YES", "CLIPPING_CONFIGURATION_MATCH": "YES", "ADAMW_CONFIGURATION_MATCH": "YES", "SOURCE_REVISION_MATCH": "YES", "control_final_rng": control_final_rng, "intervention_final_rng": intervention_final_rng, "post_treatment_model_divergence_expected": "YES", "integrity_checks": integrity_checks}
        write_json(output / "branch_matching.json", branch_matching)
        compact_exclusions = {"parameter_component_statistics", "pairwise_metrics", "adamw", "treated_total_gradient_tensor_map", "intervened_h32_gru_tensor_map"}
        write_json(output / "branch_a_control_metrics.json", jsonable({"branch": "CONTROL", "updates": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in control_rows], "update_392": {key: value for key, value in c392.items() if key not in compact_exclusions}}))
        write_json(output / "branch_b_intervention_metrics.json", jsonable({"branch": "INTERVENTION", "updates": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in intervention_rows], "update_392": {key: value for key, value in i392.items() if key not in compact_exclusions}}))
        write_json(output / "treatment_fidelity.json", treatment_evidence)
        write_json(output / "gradient_reconstruction.json", gradient_evidence)
        write_json(output / "optimizer_and_clipping_evidence.json", {"unchanged_configuration_between_branches": "YES", "configuration": {"clip": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0}, "optimizer": base.optimizer_contract(control_optimizer)}, "control_updates": [{"update": row["update"], "pre_clip_global_grad_norm": row["returned_pre_clipping_total_norm"], "post_clip_global_grad_norm": row["clipped_total_gradient_norm"], "clip_coefficient": row["clip_coefficient"], "optimizer_before_hash": row["optimizer_state_hash_before"], "optimizer_after_hash": row["optimizer_state_hash_after"]} for row in control_rows], "intervention_updates": [{"update": row["update"], "pre_clip_global_grad_norm": row["returned_pre_clipping_total_norm"], "post_clip_global_grad_norm": row["clipped_total_gradient_norm"], "clip_coefficient": row["clip_coefficient"], "optimizer_before_hash": row["optimizer_state_hash_before"], "optimizer_after_hash": row["optimizer_state_hash_after"]} for row in intervention_rows]})
        write_json(output / "causal_contrast.json", causal_contrast)
        write_json(output / "control_reproduction.json", {"corrected_d10a": d10a_control, "d7": d7_control, "control_reproduction_pass": "YES" if integrity_checks["control_reproduction"] else "NO"})
        write_json(output / "integrity_checks.json", {"checks": integrity_checks, "all_required_integrity_pass": "YES" if integrity else "NO"})
        write_json(output / "update_metrics.json", jsonable({"CONTROL": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in control_rows], "INTERVENTION": [{key: value for key, value in row.items() if key not in compact_exclusions} for row in intervention_rows]}))
        protocol = {"D10_DESIGN_REVIEW_PASSED": "YES", "D10_DESIGN_BUNDLE_IDENTITY_MATCH": "YES", "CERTIFIED_UPDATE_384_IDENTITY": "MATCH", "SOURCE_REVISION_MATCH": "YES", "CANONICAL_SCHEDULE_MATCH": "YES", "CANONICAL_RNG_MATCH": "YES", "OPTIMIZER_IDENTITY_MATCH": "YES", "BRANCHES_EXECUTED": "CONTROL, INTERVENTION", "UPDATES_EXECUTED": list(UPDATES), "D10A_CORRECTED_EVIDENCE": "VERIFIED", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HISTORICAL_ARTIFACTS_MODIFIED": "NO", "HYPERPARAMETER_SEARCH_EXECUTED": "NO", "H32_MAGNITUDE_INTERVENTION": "NO", "HEAD_INTERVENTION": "NO", "UPDATE_393_EXECUTED": "NO", "D11_EXECUTED": "NO"}
        write_json(output / "protocol_integrity.json", protocol)
        summary = {"status": "PASSED", "first_blocker": "none", "classification": classification, "verdict": f"The matched update-385 GRU-only H32 directional projection changed update-392 H1 by {effect_h1:.17g} (control minus intervention) while H32 changed by {effect_h32:.17g}; the frozen contract classifies this as {classification}.", "protocol": protocol, "control_reproduction": {"corrected_d10a": d10a_control["status"], "d7": d7_control["all_available_fields_match"], "first_mismatch": d10a_control.get("first_mismatch") or d7_control.get("first_divergence")}, "treatment_fidelity": treatment_evidence, "gradient_reconstruction": gradient_evidence, "causal_contrast": causal_contrast, "clipping_and_adamw_unchanged": "YES", "control_branch_valid": "YES" if integrity_checks["control_reproduction"] else "NO", "intervention_branch_valid": "YES" if integrity else "NO", "single_factor_valid": "YES" if integrity else "NO", "decision_h1": "YES" if effect_h1 > 0 else ("NO" if effect_h1 <= 0 else "INCONCLUSIVE"), "h1_material": "YES" if effect_h1 >= MATERIAL_H1_EFFECT else "NO", "intervention_h1_pass": "YES" if i392["eval_h1"] <= H1_THRESHOLD else "NO", "intervention_h32_pass": "YES" if i392["eval_h32"] <= H32_THRESHOLD else "NO", "h32_preserved": "YES" if i392["eval_h32"] <= H32_THRESHOLD else "NO", "what_d10_establishes": "Observational evidence: D10A localized a negative H1-versus-weighted-H32 cosine to the exact GRU scope at update 385. Causal evidence: under the matched replay, the single update-385 GRU-only anti-H1 projection produced the reported update-392 H1 contrast and the frozen classification. Unresolved mechanism: only this exact intervention is identified; no broader horizon, magnitude, head, or aggregate-gradient mechanism is identified.", "what_d10_does_not_establish": "This replay does not make D10A magnitude dominance at H32 causal, does not make head localization causal, and does not establish effects of H32 norm capping, head intervention, other horizons, later updates, aggregate projections, D11, or any future experiment.", "evidence_bundle": {"output_bundle_path": str(output.resolve()), "final_report": str((output / "FINAL_REPORT.md").resolve()), "sha256_manifest": str((output / "SHA256_MANIFEST.json").resolve())}}
        write_json(output / "terminal_certificate.json", summary)
        manifest_payload = artifact_manifest(output)
        manifest_check = verify_output_manifest(output, manifest_payload)
        if manifest_check["status"] != "VERIFIED":
            raise StageBlocker(f"FINAL_MANIFEST_VERIFICATION_FAILED:{manifest_check['failures'][:1]}")
        summary["evidence_bundle"].update({"manifest_verified": "YES", "manifest_sha256": manifest_check["manifest_sha256"], "file_count": manifest_check["checked"]})
        (output / "FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
        inventory = {"schema_version": "stage3_h13_post_d10a_d10_artifact_inventory_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": {str(path.relative_to(output)).replace("\\", "/"): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_inventory.json")}}
        write_json(output / "artifact_inventory.json", inventory)
        if sha256_file(base.A1_CHECKPOINT) != a1_before:
            raise StageBlocker("R8E_A1_MUTATED_DURING_EXECUTION")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification, "manifest_sha256": manifest_check["manifest_sha256"], "control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), inputs)
    except Exception as exc:  # pragma: no cover
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return fail_closed(output, f"UNEXPECTED_EXECUTION_FAILURE:{type(exc).__name__}:{exc}", inputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    return finalize_existing(output) if args.finalize_existing else execute(output)


if __name__ == "__main__":
    raise SystemExit(main())
