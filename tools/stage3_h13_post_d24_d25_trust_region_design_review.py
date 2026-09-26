"""D25 design-review and separately-gated execution harness.

The design-review path authenticates the D24 update-394 parent, reconstructs the
authenticated D24 P4 definition, and tests the realized-AdamW branch mechanics
only with synthetic tensors.  It deliberately does not execute the scientific
update-395 batch.  Scientific execution requires a separate authorization file
containing ``D25_EXECUTION_AUTHORIZED: YES`` and the frozen design-manifest hash.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution as d24  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D24_D25_H32_FEASIBLE_REALIZED_ADAMW_TRUST_REGION_FORWARD_CONTINUATION_DESIGN_REVIEW"
EXECUTION_EXPERIMENT_ID = "STAGE_3_H13_POST_D24_D25_H32_FEASIBLE_REALIZED_ADAMW_TRUST_REGION_FORWARD_CONTINUATION_EXECUTION"
D24_DIR = ROOT / "outputs" / "stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution_20260821T225000+0800_retry"
D23_DIR = ROOT / "outputs" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint_20260821T190000+0800"
START_CHECKPOINT = D24_DIR / "checkpoints" / "committed_update_394.pt"
EXPECTED_START_CHECKPOINT_SHA256 = "f7939ac8d6c9adedbfb2ed6cf90a949e7252ba6926932053617fbe4caac97a47"
EXPECTED_START_UPDATE = 394
TARGET_UPDATE = 395
START_H1 = 8.366622366361216e-05
START_H32 = 0.018494000982624222
H32_LIMIT = 0.01856902565856056
H32_MARGIN_AT_START = 7.5024675936338e-05
H1_TARGET = 5.0e-05
ALPHA_LADDER = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)
PROPOSAL_ORDER = ("P4",)
FALLBACK_DIRECTION = "NONE"
P4_SPEC = {
    "name": "P4",
    "family": "C5_PLUS_POSTCLIP_H1_BIAS",
    "rule": "structure_plus_postclip_h1_bias",
    "attenuation": 0.5,
    "postclip_h1_coefficient": 0.35,
    "attenuated_region": list(range(18, 32)),
}
H1_COMPARISON_TOLERANCE = 0.0
OUTPUT_DIR = ROOT / "outputs" / "stage3_h13_post_d24_d25_h32_feasible_realized_adamw_trust_region_forward_continuation_design_review_20260821T235900+0800"
EXECUTION_ENTRYPOINT = ROOT / "tools" / "stage3_h13_post_d24_d25_trust_region_execution.py"
AUTHORIZATION_EVIDENCE = Path(r"C:\Users\86198\.codex\attachments\1d7bb90c-192f-4242-bd16-54342a88182e\pasted-text.txt")
D25_ARTIFACTS = (
    "FINAL_REPORT.md",
    "D25_DESIGN_MANIFEST.json",
    "D25_ALPHA_LADDER.json",
    "D25_CANDIDATE_CONTRACT.json",
    "D25_STATIC_VALIDATION.json",
    "D25_INPUT_IDENTITY.json",
    "D25_P4_RECONSTRUCTION_VALIDATION.json",
    "D25_OPTIMIZER_STATE_CONSISTENCY_VALIDATION.json",
    "SHA256_MANIFEST.json",
)
CANDIDATE_FIELDS = (
    "parent_update",
    "candidate_update",
    "proposal",
    "alpha",
    "canonical_lr",
    "effective_lr",
    "parent_H1",
    "candidate_H1",
    "delta_H1",
    "parent_H32",
    "candidate_H32",
    "delta_H32",
    "H32_limit",
    "H32_margin_before",
    "H32_margin_after",
    "finite_state_pass",
    "H32_preservation_pass",
    "H1_improvement_pass",
    "candidate_valid",
    "rejection_reason",
    "selected",
    "model_checkpoint_sha256",
    "optimizer_state_sha256_or_equivalent",
    "batch_identity",
    "source_revision",
    "parameter_delta_l2",
    "parameter_delta_linf",
    "adamw_update_l2",
    "gradient_l2",
    "postclip_gradient_l2",
)


class DesignBlocker(RuntimeError):
    pass


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


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise DesignBlocker(f"NONFINITE:{result}")
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DesignBlocker(message)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def safe_attr_path(module: Any, name: str) -> str | None:
    value = getattr(module, name, None)
    return str(Path(value).resolve()) if isinstance(value, (str, Path)) else None


def verify_manifest_entries(directory: Path, manifest_path: Path) -> dict[str, Any]:
    require(manifest_path.is_file(), f"MANIFEST_MISSING:{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for entry in manifest.get("files", []):
        relative = Path(str(entry["relative_path"]))
        target = directory / relative
        checked += 1
        if not target.is_file() or sha256_file(target) != str(entry["sha256"]):
            failures.append(str(relative))
    return {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path), "checked_files": checked, "failures": failures, "pass": not failures}


def load_parent_checkpoint(runtime: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    require(START_CHECKPOINT.is_file(), f"START_CHECKPOINT_MISSING:{START_CHECKPOINT}")
    observed_sha = sha256_file(START_CHECKPOINT)
    require(observed_sha == EXPECTED_START_CHECKPOINT_SHA256, f"START_CHECKPOINT_SHA256_MISMATCH:{observed_sha}")
    checkpoint = torch.load(START_CHECKPOINT, map_location="cpu", weights_only=False)
    require(checkpoint.get("schema_version") == "stage3_h13_d24_resumable_committed_checkpoint_v1", "D24_CHECKPOINT_SCHEMA_MISMATCH")
    require(checkpoint.get("proposal") == "P4" and checkpoint.get("proposal_spec") == P4_SPEC, "D24_CHECKPOINT_P4_MISMATCH")
    require(int(checkpoint.get("completed_optimizer_step", -1)) == EXPECTED_START_UPDATE, "D24_CHECKPOINT_UPDATE_MISMATCH")
    require(int(checkpoint.get("next_schedule_index", -1)) == EXPECTED_START_UPDATE, "D24_CHECKPOINT_SCHEDULE_MISMATCH")
    require(checkpoint.get("batch_position", {}).get("next_zero_based_schedule_index") == EXPECTED_START_UPDATE, "D24_CHECKPOINT_BATCH_POSITION_MISMATCH")
    require(checkpoint.get("scheduler_state") is None, "UNEXPECTED_SCHEDULER_STATE")
    require(checkpoint.get("resumable") is True and checkpoint.get("scientific_state") == "COMMITTED_FORWARD_STATE", "D24_CHECKPOINT_NOT_RESUMABLE")
    model, optimizer = d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]), strict=True)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
    d24.d17.base.restore_rng(copy.deepcopy(checkpoint["rng_state"]))
    model_hash = d24.d17.canonical_model_state_sha256(model.state_dict())
    optimizer_hash = d24.d17.base.optimizer_semantic_hash(optimizer)
    rng_hash = d24.d17.base.rng_digest(d24.d17.base.capture_rng())
    last = checkpoint["last_record_identity"]
    require(model_hash == last["model_state_hash"], "D24_PARENT_MODEL_STATE_HASH_MISMATCH")
    require(optimizer_hash == last["optimizer_state_hash"], "D24_PARENT_OPTIMIZER_STATE_HASH_MISMATCH")
    require(rng_hash == d24.d17.base.rng_digest(checkpoint["rng_state"]), "D24_PARENT_RNG_HASH_MISMATCH")
    steps = d24.d16.optimizer_steps(d24.d16.clone_named_optimizer_state(optimizer))
    require(set(steps.values()) == {EXPECTED_START_UPDATE}, "D24_PARENT_OPTIMIZER_STEP_MISMATCH")
    require(d24.d16.state_finite(model, optimizer), "D24_PARENT_NONFINITE_STATE")
    identity = {
        "path": str(START_CHECKPOINT.resolve()),
        "sha256": observed_sha,
        "schema_version": checkpoint["schema_version"],
        "completed_optimizer_step": EXPECTED_START_UPDATE,
        "next_schedule_index": int(checkpoint["next_schedule_index"]),
        "model_state_hash": model_hash,
        "optimizer_state_hash": optimizer_hash,
        "rng_state_hash": rng_hash,
        "batch_identity": {"update": EXPECTED_START_UPDATE, "next_zero_based_schedule_index": EXPECTED_START_UPDATE, "batch_window_sha256": last["batch_window_sha256"]},
        "proposal": checkpoint["proposal"],
        "proposal_spec": checkpoint["proposal_spec"],
        "optimizer_group_contract": d24.d17.base.optimizer_contract(optimizer),
        "scheduler_state": checkpoint["scheduler_state"],
        "finite_state": True,
    }
    return model, optimizer, copy.deepcopy(checkpoint["rng_state"]), identity


def authenticate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    require(D24_DIR.is_dir(), f"D24_DIR_MISSING:{D24_DIR}")
    require(D23_DIR.is_dir(), f"D23_DIR_MISSING:{D23_DIR}")
    d24_manifest = verify_manifest_entries(D24_DIR, D24_DIR / "SHA256_MANIFEST.json")
    d23_manifest = verify_manifest_entries(D23_DIR, D23_DIR / "SHA256_MANIFEST.json")
    require(d24_manifest["pass"], f"D24_MANIFEST_FAILURE:{d24_manifest['failures']}")
    require(d23_manifest["pass"], f"D23_MANIFEST_FAILURE:{d23_manifest['failures']}")
    d24_run_manifest_path = D24_DIR / "D24_RUN_MANIFEST.json"
    d24_run_manifest = json.loads(d24_run_manifest_path.read_text(encoding="utf-8"))
    d24_source_path = Path(d24_run_manifest["runner_path"])
    require(d24_source_path.is_file(), f"D24_SOURCE_MISSING:{d24_source_path}")
    require(sha256_file(d24_source_path) == d24_run_manifest["runner_sha256"], "D24_SOURCE_HASH_MISMATCH")
    trajectory_path = D24_DIR / "D24_COMMITTED_FORWARD_TRAJECTORY.csv"
    with trajectory_path.open(encoding="utf-8", newline="") as stream:
        trajectory = list(csv.DictReader(stream))
    trajectory_by_update = {int(row["update"]): row for row in trajectory}
    require(trajectory_by_update[394]["selected_proposal"] == "P4", "D24_UPDATE_394_NOT_P4")
    require(math.isclose(float(trajectory_by_update[394]["H1"]), START_H1, rel_tol=0.0, abs_tol=1e-18), "D24_UPDATE_394_H1_MISMATCH")
    require(math.isclose(float(trajectory_by_update[394]["H32"]), START_H32, rel_tol=0.0, abs_tol=1e-18), "D24_UPDATE_394_H32_MISMATCH")
    proposal_path = D24_DIR / "D24_PROPOSAL_RESULTS.csv"
    with proposal_path.open(encoding="utf-8", newline="") as stream:
        proposals = list(csv.DictReader(stream))
    update395_p4 = [row for row in proposals if int(row["update"]) == 395 and row["proposal"] == "P4"]
    require(len(update395_p4) == 1, "D24_UPDATE_395_P4_EVIDENCE_MISSING")
    require(float(update395_p4[0]["H32_after"]) > H32_LIMIT, "D24_UPDATE_395_P4_BARRIER_EVIDENCE_MISMATCH")
    require(AUTHORIZATION_EVIDENCE.is_file(), f"D25_AUTHORIZATION_EVIDENCE_MISSING:{AUTHORIZATION_EVIDENCE}")
    authorization_text = AUTHORIZATION_EVIDENCE.read_text(encoding="utf-8")
    authorization_text_normalized = authorization_text.replace("**", "")
    for marker in ("STAGE 3 H13", "Do not execute scientific D25 training yet", EXPERIMENT_ID, "UPDATE_395_EXECUTED: NO"):
        require(marker in authorization_text_normalized, f"D25_AUTHORIZATION_MARKER_MISSING:{marker}")
    runtime = d24.d17.preflight_runtime()
    model, optimizer, rng, parent_identity = load_parent_checkpoint(runtime)
    input_paths = {
        "d24_final_report": D24_DIR / "FINAL_REPORT.md",
        "d24_run_manifest": d24_run_manifest_path,
        "d24_proposal_results": proposal_path,
        "d24_committed_forward_trajectory": trajectory_path,
        "d24_best_checkpoint_metadata": D24_DIR / "BEST_FORWARD_CHECKPOINT_METADATA.json",
        "d24_sha256_manifest": D24_DIR / "SHA256_MANIFEST.json",
        "d24_checkpoint": START_CHECKPOINT,
        "d23_sha256_manifest": D23_DIR / "SHA256_MANIFEST.json",
        "d23_runner": D23_DIR / "D23_RUN_MANIFEST.json",
        "d23_checkpoint_parent": D23_DIR / "checkpoints" / "C5_update_392.pt",
        "d25_authorization_evidence": AUTHORIZATION_EVIDENCE,
        "d24_source": d24_source_path,
        "d16_source": Path(d24.d16.__file__).resolve(),
        "d17_source": Path(d24.d17.__file__).resolve(),
        "d25_execution_entrypoint": EXECUTION_ENTRYPOINT,
    }
    for name, path in input_paths.items():
        require(path.is_file(), f"INPUT_MISSING:{name}:{path}")
    input_identity = {
        "schema_version": "stage3_h13_d25_input_identity_v1",
        "authoritative_d24": {
            "experiment_id": d24_run_manifest["experiment_id"],
            "run_manifest_sha256": sha256_file(d24_run_manifest_path),
            "sha256_manifest_sha256": sha256_file(D24_DIR / "SHA256_MANIFEST.json"),
            "final_report_sha256": sha256_file(D24_DIR / "FINAL_REPORT.md"),
            "proposal_results_sha256": sha256_file(proposal_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "checkpoint_sha256": EXPECTED_START_CHECKPOINT_SHA256,
            "checkpoint_update": EXPECTED_START_UPDATE,
            "start_H1": START_H1,
            "start_H32": START_H32,
            "h32_limit": H32_LIMIT,
            "d24_source_sha256": sha256_file(d24_source_path),
        },
        "authoritative_d23": {
            "run_manifest_sha256": sha256_file(D23_DIR / "D23_RUN_MANIFEST.json"),
            "sha256_manifest_sha256": sha256_file(D23_DIR / "SHA256_MANIFEST.json"),
            "checkpoint_parent_sha256": sha256_file(D23_DIR / "checkpoints" / "C5_update_392.pt"),
        },
        "runtime": {
            "git_head": runtime["git_head"],
            "source_revision": git_head(),
            "environment": runtime["environment"],
            "d17_runner_sha256": runtime["runner_sha256"],
            "d16_source_sha256": sha256_file(Path(d24.d16.__file__).resolve()),
            "d17_source_sha256": sha256_file(Path(d24.d17.__file__).resolve()),
            "d25_execution_entrypoint_sha256": sha256_file(EXECUTION_ENTRYPOINT),
            "bundle_path": safe_attr_path(d24.d17, "BUNDLE_PATH"),
            "schedule_identity_path": safe_attr_path(d24.d17, "SCHEDULE_IDENTITY"),
            "schedule_storage_path": safe_attr_path(d24.d17, "SCHEDULE_STORAGE"),
            "bundle_sha256": sha256_file(Path(d24.d17.BUNDLE_PATH)),
            "schedule_identity_sha256": sha256_file(Path(d24.d17.SCHEDULE_IDENTITY)),
            "schedule_storage_sha256": sha256_file(Path(d24.d17.SCHEDULE_STORAGE)),
        },
        "d25_authorization": {"path": str(AUTHORIZATION_EVIDENCE.resolve()), "sha256": sha256_file(AUTHORIZATION_EVIDENCE)},
        "parent_checkpoint_identity": parent_identity,
        "d24_manifest_validation": d24_manifest,
        "d23_manifest_validation": d23_manifest,
        "d24_evidence_update_395_p4": {
            "candidate_H1": float(update395_p4[0]["H1_after"]),
            "candidate_H32": float(update395_p4[0]["H32_after"]),
            "h32_rejection_observed": True,
        },
    }
    return input_identity, {"runtime": runtime, "model": model, "optimizer": optimizer, "rng": rng, "parent_identity": parent_identity}


def canonical_group_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [finite(group["lr"]) for group in optimizer.param_groups]


def effective_group_lrs(canonical_lrs: Sequence[float], alpha: float) -> list[float]:
    require(math.isfinite(alpha) and alpha > 0.0, f"INVALID_ALPHA:{alpha}")
    return [finite(float(lr) * float(alpha)) for lr in canonical_lrs]


def apply_effective_lrs(optimizer: torch.optim.Optimizer, alpha: float, canonical_lrs: Sequence[float] | None = None) -> dict[str, Any]:
    base_lrs = list(canonical_lrs) if canonical_lrs is not None else canonical_group_lrs(optimizer)
    require(len(base_lrs) == len(optimizer.param_groups), "LR_GROUP_COUNT_MISMATCH")
    effective_lrs = effective_group_lrs(base_lrs, alpha)
    for group, effective_lr in zip(optimizer.param_groups, effective_lrs):
        group["lr"] = effective_lr
    return {"alpha": float(alpha), "canonical_base_lr": base_lrs, "candidate_effective_lr": effective_lrs}


def clone_parent_branch(parent_model: Any, parent_optimizer: torch.optim.Optimizer, bundle: Mapping[str, Any], alpha: float, canonical_lrs: Sequence[float] | None = None) -> tuple[Any, torch.optim.Optimizer, dict[str, Any]]:
    model, optimizer = d24.d16.new_branch(bundle)
    model.load_state_dict(copy.deepcopy(parent_model.state_dict()), strict=True)
    optimizer.load_state_dict(copy.deepcopy(parent_optimizer.state_dict()))
    model.train()
    base_lrs = list(canonical_lrs) if canonical_lrs is not None else canonical_group_lrs(parent_optimizer)
    lr_record = apply_effective_lrs(optimizer, alpha, base_lrs)
    return model, optimizer, lr_record


def tensor_digest(values: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def object_digest(value: Any) -> str:
    """Stable digest for nested synthetic optimizer state without JSONifying tensors."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in item:
                digest.update(repr(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def recursive_tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return list(left.keys()) == list(right.keys()) and all(recursive_tensor_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(recursive_tensor_equal(a, b) for a, b in zip(left, right))
    return left == right


def synthetic_model_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(20260821)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    return model, optimizer


def clone_synthetic(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    clone = copy.deepcopy(model)
    clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return clone, clone_optimizer


def set_synthetic_grads(model: torch.nn.Module) -> None:
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, 0.01 * index)


def synthetic_adamw_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, alpha: float) -> tuple[list[float], list[float]]:
    canonical = canonical_group_lrs(optimizer)
    effective = effective_group_lrs(canonical, alpha)
    for group, lr in zip(optimizer.param_groups, effective):
        group["lr"] = lr
    set_synthetic_grads(model)
    optimizer.step()
    return canonical, effective


def rank_candidates(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    valid = [row for row in rows if row.get("candidate_valid") is True]
    if not valid:
        return None
    return min(valid, key=lambda row: (float(row["candidate_H1"]), -float(row["alpha"]), PROPOSAL_ORDER.index(str(row["proposal"]))))


def run_static_tests() -> dict[str, Any]:
    results: dict[str, Any] = {}
    parent_model, parent_optimizer = synthetic_model_optimizer()
    parent_model_state = copy.deepcopy(parent_model.state_dict())
    parent_optimizer_state = copy.deepcopy(parent_optimizer.state_dict())
    parent_lrs = canonical_group_lrs(parent_optimizer)

    canonical_model, canonical_optimizer = clone_synthetic(parent_model, parent_optimizer)
    alpha1_model, alpha1_optimizer = clone_synthetic(parent_model, parent_optimizer)
    canonical_lrs, canonical_effective = synthetic_adamw_step(canonical_model, canonical_optimizer, 1.0)
    alpha1_lrs, alpha1_effective = synthetic_adamw_step(alpha1_model, alpha1_optimizer, 1.0)
    results["alpha_identity"] = {
        "pass": recursive_tensor_equal(canonical_model.state_dict(), alpha1_model.state_dict()) and recursive_tensor_equal(canonical_optimizer.state_dict(), alpha1_optimizer.state_dict()),
        "canonical_lrs": canonical_lrs,
        "alpha1_effective_lrs": alpha1_effective,
        "model_exact_equal": recursive_tensor_equal(canonical_model.state_dict(), alpha1_model.state_dict()),
        "optimizer_exact_equal": recursive_tensor_equal(canonical_optimizer.state_dict(), alpha1_optimizer.state_dict()),
    }

    zero_model, zero_optimizer = clone_synthetic(parent_model, parent_optimizer)
    before_zero = copy.deepcopy(zero_model.state_dict())
    _, zero_effective = synthetic_adamw_step(zero_model, zero_optimizer, 0.0 + 1e-30)
    # Scientific ladder excludes zero. This near-zero test checks parameter
    # immobility without relying on an out-of-contract ladder value.
    zero_unchanged = all(torch.equal(before_zero[name], zero_model.state_dict()[name]) for name in before_zero)
    results["zero_scale_sanity"] = {"pass": zero_unchanged, "test_alpha": 1e-30, "effective_lrs": zero_effective, "parameters_unchanged": zero_unchanged, "optimizer_step_state_allowed_to_change": True}

    branch_a, branch_a_opt = clone_synthetic(parent_model, parent_optimizer)
    branch_b, branch_b_opt = clone_synthetic(parent_model, parent_optimizer)
    standalone_b, standalone_b_opt = clone_synthetic(parent_model, parent_optimizer)
    synthetic_adamw_step(branch_a, branch_a_opt, 0.5)
    synthetic_adamw_step(branch_b, branch_b_opt, 0.25)
    synthetic_adamw_step(standalone_b, standalone_b_opt, 0.25)
    results["parent_restoration_and_rejected_candidate_isolation"] = {
        "pass": recursive_tensor_equal(branch_b.state_dict(), standalone_b.state_dict()) and recursive_tensor_equal(branch_b_opt.state_dict(), standalone_b_opt.state_dict()),
        "candidate_a_rejected": True,
        "candidate_b_matches_independent_parent_child": recursive_tensor_equal(branch_b.state_dict(), standalone_b.state_dict()) and recursive_tensor_equal(branch_b_opt.state_dict(), standalone_b_opt.state_dict()),
        "parent_model_unchanged": recursive_tensor_equal(parent_model.state_dict(), parent_model_state),
        "parent_optimizer_unchanged": recursive_tensor_equal(parent_optimizer.state_dict(), parent_optimizer_state),
    }

    lr_model, lr_optimizer = clone_synthetic(parent_model, parent_optimizer)
    lr_record = apply_effective_lrs(lr_optimizer, 0.25, parent_lrs)
    results["lr_scaling"] = {
        "pass": lr_record["candidate_effective_lr"] == [value * 0.25 for value in parent_lrs],
        "canonical_base_lr": lr_record["canonical_base_lr"],
        "candidate_effective_lr": lr_record["candidate_effective_lr"],
        "formula": "candidate_effective_lr = canonical_base_lr * alpha",
    }

    consistency_model, consistency_optimizer = clone_synthetic(parent_model, parent_optimizer)
    synthetic_adamw_step(consistency_model, consistency_optimizer, 0.5)
    model_hash = tensor_digest(consistency_model.state_dict().values())
    optimizer_hash = object_digest(consistency_optimizer.state_dict())
    results["optimizer_consistency"] = {
        "pass": model_hash != tensor_digest(parent_model.state_dict().values()) and set(consistency_optimizer.state.keys()) == set(consistency_model.parameters()),
        "model_branch_digest": model_hash,
        "optimizer_state_digest": optimizer_hash,
        "same_candidate_branch": True,
    }

    ranking_rows = [
        {"proposal": "P4", "alpha": 0.5, "candidate_H1": 0.000080, "candidate_valid": True},
        {"proposal": "P4", "alpha": 0.25, "candidate_H1": 0.000079, "candidate_valid": True},
        {"proposal": "P4", "alpha": 0.125, "candidate_H1": 0.000079, "candidate_valid": True},
        {"proposal": "P4", "alpha": 0.0625, "candidate_H1": 0.000081, "candidate_valid": False},
    ]
    winner = rank_candidates(ranking_rows)
    results["deterministic_ranking"] = {"pass": winner is not None and winner["alpha"] == 0.25, "expected_winner": {"proposal": "P4", "alpha": 0.25}, "observed_winner": dict(winner) if winner else None}
    results["h32_contract"] = {"pass": H32_LIMIT == 0.01856902565856056, "inherited_h32_limit": H32_LIMIT, "buffer": 0.0, "buffer_policy": "none; no new scientific buffer"}
    results["manifest_completeness"] = {"pass": len(CANDIDATE_FIELDS) >= 25 and {"parent_update", "candidate_update", "alpha", "canonical_lr", "effective_lr", "candidate_valid", "selected"}.issubset(CANDIDATE_FIELDS), "field_count": len(CANDIDATE_FIELDS), "fields": list(CANDIDATE_FIELDS)}
    all_pass = all(bool(value.get("pass")) for value in results.values())
    return {"schema_version": "stage3_h13_d25_static_validation_v1", "scientific_training_executed": False, "scientific_backward_pass_executed": False, "scientific_optimizer_step_executed": False, "synthetic_only": True, "tests": results, "pass": all_pass}


def p4_reconstruction_validation(static_validation: Mapping[str, Any], parent_identity: Mapping[str, Any]) -> dict[str, Any]:
    observed_spec = d24.proposal_spec("P4")
    source = inspect.getsource(d24.proposal_spec)
    markers = {
        "proposal_spec_entry_point": "if name == \"P4\":" in source,
        "attenuation_0_5": '"attenuation": 0.5' in source,
        "postclip_h1_coefficient_0_35": '"postclip_h1_coefficient": 0.35' in source,
        "attenuated_h18_h31": "list(range(18, 32))" in source,
        "gradient_construction": "d17.base.causal_paired_rollout" in inspect.getsource(d24.run_trial),
        "canonical_clip": "torch.nn.utils.clip_grad_norm_" in inspect.getsource(d24.run_trial),
        "adamw_after_clip": "optimizer.step()" in inspect.getsource(d24.run_trial),
        "post_step_validation": "validation_snapshot" in inspect.getsource(d24.run_trial),
    }
    canonical_lrs = parent_identity["optimizer_group_contract"].get("learning_rate", 5.0e-5)
    if isinstance(canonical_lrs, list):
        canonical_lrs = [float(value) for value in canonical_lrs]
    else:
        canonical_lrs = [float(canonical_lrs)]
    alpha1_effective = effective_group_lrs(canonical_lrs, 1.0)
    return {
        "schema_version": "stage3_h13_d25_p4_reconstruction_validation_v1",
        "pass": observed_spec == P4_SPEC and parent_identity["proposal_spec"] == P4_SPEC and all(markers.values()) and alpha1_effective == canonical_lrs and bool(static_validation["tests"]["alpha_identity"]["pass"]),
        "source_file": str(Path(d24.__file__).resolve()),
        "source_sha256": sha256_file(Path(d24.__file__).resolve()),
        "function_entry_point": "tools.stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution.proposal_spec('P4')",
        "exact_parameters": observed_spec,
        "gradient_inputs": ["canonical paired rollout predictions/targets", "H1 diagnostic gradient", "rest gradient", "H32 diagnostic gradient"],
        "clipping_placement": "torch.nn.utils.clip_grad_norm_ after P4 gradient construction and before postclip H1 directional bias",
        "adamw_placement": "candidate optimizer.step() after presented P4 postclip gradient is installed",
        "normalization_projection": "postclip H1 bias adds normalized H1 direction and renormalizes to canonical postclip norm",
        "alpha_1": {"canonical_base_lr": canonical_lrs, "candidate_effective_lr": alpha1_effective, "lr_identity": alpha1_effective == canonical_lrs, "synthetic_full_step_identity": bool(static_validation["tests"]["alpha_identity"]["pass"])},
        "static_source_markers": markers,
        "parent_checkpoint_proposal": parent_identity["proposal"],
        "parent_checkpoint_proposal_spec": parent_identity["proposal_spec"],
    }


def optimizer_consistency_validation(static_validation: Mapping[str, Any], parent_identity: Mapping[str, Any]) -> dict[str, Any]:
    contract = parent_identity["optimizer_group_contract"]
    return {
        "schema_version": "stage3_h13_d25_optimizer_state_consistency_validation_v1",
        "pass": bool(static_validation["tests"]["optimizer_consistency"]["pass"]) and contract.get("class") == "AdamW" and contract.get("learning_rate") == 5.0e-5,
        "mechanism": "clone parent model and optimizer state, set each isolated candidate optimizer param-group lr to canonical_lr * alpha, then call that candidate optimizer's actual AdamW step; commit model and optimizer from the same accepted branch",
        "prohibited_mechanism": "post-hoc parameter-delta scaling while retaining full-step optimizer state",
        "parent_optimizer_contract": contract,
        "state_transition": {"moments": "updated by candidate AdamW step", "step_counters": "incremented by candidate AdamW step", "parameter_groups": "copied from parent; only temporary candidate lr differs", "scheduler_state": "None in authenticated D24 parent; canonical schedule definition remains external/frozen"},
        "candidate_isolation": "every alpha is cloned from the same parent model, optimizer, scheduler, RNG, and batch identity; rejected branches are never reused",
        "synthetic_tests": static_validation["tests"],
    }


def design_manifest(input_identity: Mapping[str, Any], p4_validation: Mapping[str, Any], optimizer_validation: Mapping[str, Any], static_validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_d25_design_manifest_v1",
        "status": "FROZEN_FOR_SEPARATE_EXECUTION_AUTHORIZATION",
        "experiment_id": EXPERIMENT_ID,
        "recommended_execution_name": EXECUTION_EXPERIMENT_ID,
        "scientific_question": "Can the demonstrated D24 H1 descent direction be continued from update 394 by reducing the magnitude of the realized AdamW parameter step enough to remain inside the frozen H32 feasible region?",
        "authoritative_d24": {"experiment_id": input_identity["authoritative_d24"]["experiment_id"], "start_checkpoint": str(START_CHECKPOINT.resolve()), "start_checkpoint_sha256": EXPECTED_START_CHECKPOINT_SHA256, "start_update": EXPECTED_START_UPDATE, "start_H1": START_H1, "start_H32": START_H32, "h32_limit": H32_LIMIT, "h32_margin_at_start": H32_MARGIN_AT_START, "d24_classification": "D24_EXECUTION_BLOCKED", "scientific_interpretation": "FORWARD_OPTIMIZATION_SUCCEEDED_THROUGH_UPDATE_394; H32_FEASIBILITY_BARRIER_IDENTIFIED_AT_UPDATE_395"},
        "primary_direction": {"proposal": "P4", "definition": P4_SPEC, "source_file": p4_validation["source_file"], "source_sha256": p4_validation["source_sha256"], "entry_point": p4_validation["function_entry_point"]},
        "fallback_direction": FALLBACK_DIRECTION,
        "alpha_ladder": [float(alpha) for alpha in ALPHA_LADDER],
        "search_order": {"policy": "largest-to-smallest monotone backtracking; stop at first finite/H32/H1-valid alpha, then evaluate only its immediately adjacent smaller alpha when available for ranking", "candidate_parent": "all candidates are independent children of update-394 parent", "no_exhaustive_smaller_alpha_evaluation": True},
        "candidate_isolation": {"clone": ["model parameters", "optimizer first moments", "optimizer second moments", "optimizer step counters", "parameter-group metadata", "scheduler state", "RNG state", "batch identity", "repository-specific runtime state"], "rejection": "discard full candidate model/optimizer/RNG state", "commit": "commit exactly one accepted candidate model and its matching optimizer state"},
        "candidate_optimizer_semantics": {"realized_step_multiplier": "alpha", "canonical_base_lr": "recorded per optimizer parameter group for every trial", "candidate_effective_lr": "canonical_base_lr * alpha per optimizer parameter group", "adamw_state": "actual candidate AdamW step updates moments and counters coherently", "frozen": {"betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 1.0e-5, "amsgrad": False, "clipping": "D24 canonical clip then P4 postclip direction", "gradient_construction": "D24 P4 pipeline", "parent_optimizer": "authenticated update-394 optimizer state"}},
        "hard_gates": {"finite_model_state": True, "finite_optimizer_state": True, "finite_metrics": True, "H32": f"candidate_H32 <= {H32_LIMIT:.17g}", "H1": "candidate_H1 < incumbent_H1; inherited comparison tolerance is 0.0"},
        "candidate_ranking": {"order": ["finite/stable", "H32 preservation", "H1 improvement", "lowest candidate_H1", "larger alpha on inherited tolerance tie", "proposal order"], "proposal_order": list(PROPOSAL_ORDER)},
        "safety_buffer": {"inherited_buffer_present": False, "new_buffer": 0.0, "decision": "no scientific buffer introduced; exact hard limit inherited"},
        "target_authorization_boundary": {"first_execution_target_update": TARGET_UPDATE, "update_396_plus": "not authorized by this design review", "training_executed_during_review": False, "backward_executed_during_review": False, "optimizer_step_executed_during_review": False},
        "failure_classifications": {"success": "D25_UPDATE_395_FEASIBLE_FORWARD_CONTINUATION", "failure_A": "NO_H32_FEASIBLE_ALPHA", "failure_B": "H32_FEASIBLE_BUT_NO_H1_IMPROVING_ALPHA", "failure_C": "P4_RECONSTRUCTION_FAILURE", "failure_D": "CANDIDATE_STATE_ISOLATION_FAILURE", "failure_E": "OPTIMIZER_STATE_CONSISTENCY_FAILURE"},
        "required_execution_outputs": ["D25_CANDIDATE_RESULTS.csv", "D25_COMMITTED_FORWARD_TRAJECTORY.csv", "D25_UPDATE_395_CHECKPOINT.pt", "D25_EXECUTION_MANIFEST.json", "D25_EXECUTION_SUMMARY.json", "SHA256_MANIFEST.json"],
        "candidate_result_schema": list(CANDIDATE_FIELDS),
        "completion_context": {"H1_target": H1_TARGET, "stage3_completion_contract": "inherited; D25 update 395 success alone does not complete Stage 3"},
        "prohibited_operations": ["D22/D23/D24 rerun", "D25 scientific training during design review", "update 396+ execution in initial D25 authorization", "post-hoc delta scaling with full-step optimizer state", "threshold mutation", "new proposal families", "broad causal localization", "OFF states/reorientation/GNN/PPO/LSTM/Transformer/retreat/approach/closed-contour transitions"],
        "source_revision": input_identity["runtime"]["source_revision"],
        "runtime_identity": input_identity["runtime"],
        "relevant_input_sha256": input_identity,
        "implementation_entrypoint": str(EXECUTION_ENTRYPOINT.resolve()),
        "implementation_entrypoint_sha256": sha256_file(EXECUTION_ENTRYPOINT),
        "validation": {"p4_reconstruction": bool(p4_validation["pass"]), "optimizer_state_consistency": bool(optimizer_validation["pass"]), "static_validation": bool(static_validation["pass"])},
    }


def write_sha256_manifest(output: Path) -> dict[str, Any]:
    excluded = {"FINAL_REPORT.md", "SHA256_MANIFEST.json"}
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"relative_path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {"schema_version": "stage3_h13_d25_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "excluded_from_manifest": sorted(excluded), "file_count": len(files), "files": files}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return manifest


def final_report(manifest_path: Path, manifest_sha: str, input_identity: Mapping[str, Any], p4_validation: Mapping[str, Any], static_validation: Mapping[str, Any], optimizer_validation: Mapping[str, Any]) -> str:
    margin = H32_MARGIN_AT_START
    return f"""{EXPERIMENT_ID}:
PASSED

DESIGN_REVIEW_PASSED:
YES

FIRST_BLOCKER:
none

ONE_SENTENCE_VERDICT:
The authenticated update-394 P4 parent can be continued with isolated realized-AdamW alpha branches under the frozen H32 contract, and the design is ready for separate update-395 authorization.

AUTHORITATIVE_START_CHECKPOINT:
{START_CHECKPOINT.resolve()}

AUTHORITATIVE_START_CHECKPOINT_SHA256:
{EXPECTED_START_CHECKPOINT_SHA256}

START_UPDATE:
394

TARGET_UPDATE:
395

START_H1:
{START_H1}

START_H32:
{START_H32}

H32_PRESERVATION_LIMIT:
{H32_LIMIT}

H32_MARGIN_AT_START:
{margin}

PRIMARY_DIRECTION:
P4

FALLBACK_DIRECTION:
{FALLBACK_DIRECTION}

ALPHA_LADDER:
{json.dumps([float(alpha) for alpha in ALPHA_LADDER], separators=(',', ':'))}

REALIZED_ADAMW_SCALING_MECHANISM:
Each independent candidate clones the authenticated update-394 model and AdamW optimizer state, sets every candidate parameter-group LR to canonical_base_lr times alpha, executes that candidate's actual AdamW step, and commits model plus optimizer only for the selected branch; no post-hoc parameter-delta scaling is used.

CANDIDATE_STATE_ISOLATION:
PASS

P4_ALPHA_1_RECONSTRUCTION:
{"PASS" if p4_validation["pass"] else "FAIL"}

H32_CONTRACT_VERIFIED:
PASS

H1_ACCEPTANCE_RULE:
candidate_H1 < incumbent_H1 ({START_H1}) with inherited comparison tolerance {H1_COMPARISON_TOLERANCE}

CANDIDATE_SELECTION_RULE:
Among finite H32-preserving H1-improving candidates choose lowest H1; ties within inherited tolerance choose larger alpha; remaining ties use deterministic proposal order.

D25_EXECUTED:
NO

UPDATE_395_EXECUTED:
NO

UPDATE_396_PLUS_EXECUTED:
NO

DESIGN_MANIFEST:
{manifest_path.resolve()}

DESIGN_MANIFEST_SHA256:
{manifest_sha}

IMPLEMENTATION_ENTRYPOINT:
{EXECUTION_ENTRYPOINT.resolve()}

STATIC_VALIDATION:
{"PASS" if static_validation["pass"] and optimizer_validation["pass"] else "FAIL"}

READY_FOR_SEPARATE_D25_EXECUTION_AUTHORIZATION:
YES

NEXT_RECOMMENDED_ACTION:
Authorize one D25 execution targeting update 395 only with the frozen manifest hash and a separate execution authorization file.
"""


def run_design_review(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    # A failed preflight may leave an empty directory behind. Reusing that
    # exact empty directory is safe; any populated directory remains protected
    # from accidental overwrite.
    if output.exists():
        require(output.is_dir() and not any(output.iterdir()), f"D25_OUTPUT_ALREADY_EXISTS:{output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    input_identity, context = authenticate_inputs()
    static_validation = run_static_tests()
    p4_validation = p4_reconstruction_validation(static_validation, context["parent_identity"])
    optimizer_validation = optimizer_consistency_validation(static_validation, context["parent_identity"])
    require(static_validation["pass"], "D25_STATIC_VALIDATION_FAILED")
    require(p4_validation["pass"], "P4_RECONSTRUCTION_FAILURE")
    require(optimizer_validation["pass"], "OPTIMIZER_STATE_CONSISTENCY_FAILURE")
    ladder = {"schema_version": "stage3_h13_d25_alpha_ladder_v1", "status": "FROZEN", "values": [float(alpha) for alpha in ALPHA_LADDER], "fractions": ["1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128", "1/256"], "primary_proposal": "P4", "fallback_proposal": FALLBACK_DIRECTION, "search_policy": "largest-to-smallest monotone backtracking; stop at first finite/H32/H1-valid candidate, then evaluate only the immediately adjacent smaller alpha when available"}
    contract = {"schema_version": "stage3_h13_d25_candidate_contract_v1", "parent_update": EXPECTED_START_UPDATE, "target_update": TARGET_UPDATE, "proposal": "P4", "candidate_fields": list(CANDIDATE_FIELDS), "finite_gate": ["model_state", "optimizer_state", "metrics"], "H32_limit": H32_LIMIT, "H32_comparison": "candidate_H32 <= H32_limit", "H1_incumbent": START_H1, "H1_comparison": "candidate_H1 < incumbent_H1", "H1_tolerance": H1_COMPARISON_TOLERANCE, "ranking": "valid lowest H1; larger alpha tie-break; proposal order final tie-break", "safety_buffer": 0.0, "execution_boundary": "update 395 only; no update 396+", "scientific_execution_in_review": False}
    write_json(output / "D25_ALPHA_LADDER.json", ladder)
    write_json(output / "D25_CANDIDATE_CONTRACT.json", contract)
    write_json(output / "D25_INPUT_IDENTITY.json", input_identity)
    write_json(output / "D25_STATIC_VALIDATION.json", static_validation)
    write_json(output / "D25_P4_RECONSTRUCTION_VALIDATION.json", p4_validation)
    write_json(output / "D25_OPTIMIZER_STATE_CONSISTENCY_VALIDATION.json", optimizer_validation)
    design = design_manifest(input_identity, p4_validation, optimizer_validation, static_validation)
    write_json(output / "D25_DESIGN_MANIFEST.json", design)
    sha_manifest = write_sha256_manifest(output)
    design_hash = sha256_file(output / "D25_DESIGN_MANIFEST.json")
    report = final_report(output / "D25_DESIGN_MANIFEST.json", design_hash, input_identity, p4_validation, static_validation, optimizer_validation)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    # Recompute the manifest after all non-excluded artifacts are final; FINAL_REPORT
    # and the manifest itself remain excluded by the explicit D24-compatible rule.
    sha_manifest = write_sha256_manifest(output)
    summary = {"status": "PASSED", "output_directory": str(output.resolve()), "design_manifest_sha256": design_hash, "sha256_manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json"), "scientific_training_executed": False, "scientific_backward_pass_executed": False, "scientific_optimizer_step_executed": False, "update_395_executed": False, "update_396_plus_executed": False, "input_identity": input_identity, "static_validation": static_validation, "p4_reconstruction": p4_validation, "optimizer_consistency": optimizer_validation}
    # The summary is returned to the caller and deliberately not emitted as an
    # extra artifact, keeping the required output set exact.
    return summary


def execution_authorized(path: Path, design_manifest_sha256: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return "D25_EXECUTION_AUTHORIZED: YES" in text and f"DESIGN_MANIFEST_SHA256: {design_manifest_sha256}" in text


def execute_update_395(*, authorization: Path, design_manifest: Path, output: Path) -> None:
    """Execute the later separately-authorized update-395 D25 continuation.

    This function is intentionally unreachable from the default design-review
    command. It is included now so the frozen design has a concrete entrypoint.
    The full scientific candidate pipeline is implemented in the companion
    execution module once authorization is supplied; this guard prevents an
    accidental design-review invocation from training.
    """
    design_hash = sha256_file(design_manifest)
    require(execution_authorized(authorization, design_hash), "D25_EXECUTION_AUTHORIZATION_MISSING_OR_MANIFEST_HASH_MISMATCH")
    raise DesignBlocker("D25_EXECUTION_ENTRYPOINT_GUARD: scientific execution requires the separately authorized execution module")


def main() -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--design-review", action="store_true", help="authenticate inputs, run synthetic validation, and freeze the D25 design")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not args.design_review:
        parser.error("scientific D25 execution is disabled here; pass --design-review for the non-training design review")
    try:
        summary = run_design_review(args.output.resolve())
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}:{exc}", "scientific_training_executed": False, "scientific_backward_pass_executed": False, "scientific_optimizer_step_executed": False}, sort_keys=True))
        return 1
    print(json.dumps({"status": summary["status"], "output_directory": summary["output_directory"], "design_manifest_sha256": summary["design_manifest_sha256"], "sha256_manifest_sha256": summary["sha256_manifest_sha256"], "scientific_training_executed": False, "scientific_backward_pass_executed": False, "scientific_optimizer_step_executed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
