"""Future D20 runner and design-contract validator.

Default invocation is static-only.  Scientific execution is fail-closed behind
an explicit authorization text marker and is intentionally not called by the
D20 design-review materializer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from itertools import combinations, product
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FACTORS = ("X32", "A", "B", "C")
BLOCKS = {
    "A": tuple(range(2, 18)),
    "B": tuple(range(18, 27)),
    "C": tuple(range(27, 32)),
}
HORIZONS = tuple(range(2, 32))
UPDATES = tuple(range(385, 393))
H1_MATERIALITY_THRESHOLD = 4.099006815405639e-06
H32_PRESERVATION_THRESHOLD = 0.01856902565856056
DECOMPOSITION_TOLERANCE = 1e-12
COMPONENTS = (
    "M_32_A",
    "M_32_B",
    "M_32_C",
    "M_32_A_B",
    "M_32_A_C",
    "M_32_B_C",
    "M_32_A_B_C",
)
IMPORTED_CELLS = {"0000", "1000", "0111", "1111"}
FUTURE_CELLS = (
    "0100", "0010", "0001",
    "0110", "0101", "0011",
    "1100", "1010", "1001",
    "1110", "1101", "1011",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def cell_id(bits: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def factorial_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for bits in product((0, 1), repeat=4):
        cid = cell_id(bits)
        delete: list[str] = []
        if bits[0]:
            delete.append("H32")
        for factor, bit in zip(("A", "B", "C"), bits[1:]):
            if bit:
                delete.extend(f"H{h}" for h in BLOCKS[factor])
        cells.append(
            {
                "cell": cid,
                "bits": list(bits),
                "factor_values": dict(zip(FACTORS, bits)),
                "delete_horizons": delete,
                "delete_set_size": len(delete),
                "scientific_endpoint_status": "NOT_EXECUTED_BY_DESIGN_REVIEW",
            }
        )
    return cells


def _subset_from_bits(bits: str) -> frozenset[str]:
    return frozenset(factor for factor, bit in zip(FACTORS, bits) if bit == "1")


def _bits_from_subset(subset: frozenset[str]) -> str:
    return "".join("1" if factor in subset else "0" for factor in FACTORS)


def mobius_interactions(endpoints: Mapping[str, float]) -> dict[frozenset[str], float]:
    """Return M(T) under the preregistered E(∅)=0 convention."""
    expected = {cell_id(bits) for bits in product((0, 1), repeat=4)}
    if set(endpoints) != expected:
        raise ValueError("endpoint registry must contain exactly all 16 factorial cells")
    effects = {frozenset(): 0.0}
    effects.update({_subset_from_bits(cid): float(value) for cid, value in endpoints.items()})
    interactions: dict[frozenset[str], float] = {}
    for size in range(1, 5):
        for subset_tuple in combinations(FACTORS, size):
            subset = frozenset(subset_tuple)
            value = effects[subset]
            for proper_size in range(1, size):
                for proper_tuple in combinations(subset_tuple, proper_size):
                    value -= interactions[frozenset(proper_tuple)]
            interactions[subset] = value
    return interactions


def seven_h32_components(interactions: Mapping[frozenset[str], float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in COMPONENTS:
        factor_names = ["X32" if token == "32" else token for token in name.removeprefix("M_").split("_")]
        result[name] = float(interactions[frozenset(factor_names)])
    return result


def classify_components(components: Mapping[str, float], higher_order_remainder: float) -> dict[str, Any]:
    material = [name for name, value in components.items() if abs(float(value)) >= H1_MATERIALITY_THRESHOLD]
    if len(material) == 0 and abs(higher_order_remainder) >= H1_MATERIALITY_THRESHOLD:
        label = "H32_BY_DISTRIBUTED_SUBTHRESHOLD_BLOCK_HIGHER_ORDER"
        dominant_block = None
    elif len(material) == 1 and material[0] in {"M_32_A", "M_32_B", "M_32_C"}:
        label = "H32_BY_SINGLE_BLOCK_INTERNAL_HIGHER_ORDER_LOCALIZED"
        dominant_block = material[0][-1]
    elif len(material) == 1 and material[0] in {"M_32_A_B", "M_32_A_C", "M_32_B_C"}:
        label = "H32_BY_SINGLE_CROSS_BLOCK_THIRD_ORDER_LOCALIZED"
        dominant_block = material[0].removeprefix("M_32_")
    elif len(material) == 1 and material[0] == "M_32_A_B_C":
        label = "H32_BY_FOURTH_ORDER_THREE_BLOCK_INTERACTION_LOCALIZED"
        dominant_block = None
    elif len(material) >= 2:
        label = "H32_BY_DISTRIBUTED_MULTI_COMPONENT_HIGHER_ORDER"
        dominant_block = None
    else:
        label = "BLOCK_FACTORIAL_DECOMPOSITION_INVALID"
        dominant_block = None
    ranking = sorted(material, key=lambda name: abs(float(components[name])), reverse=True)
    return {"classification": label, "material_components": ranking, "dominant_block": dominant_block}


def static_validation() -> dict[str, Any]:
    cells = factorial_cells()
    ids = [cell["cell"] for cell in cells]
    expected_ids = [cell_id(bits) for bits in product((0, 1), repeat=4)]
    if ids != expected_ids or len(set(ids)) != 16:
        raise AssertionError("factorial registry is not the complete unique 2^4 design")
    if set(ids) - IMPORTED_CELLS - set(FUTURE_CELLS) or IMPORTED_CELLS & set(FUTURE_CELLS):
        raise AssertionError("import/future cell partition is invalid")
    expected_delete = {
        "0000": set(),
        "1000": {"H32"},
        "0111": {f"H{h}" for h in HORIZONS},
        "1111": {"H32", *(f"H{h}" for h in HORIZONS)},
    }
    for cell in cells:
        if cell["cell"] in expected_delete and set(cell["delete_horizons"]) != expected_delete[cell["cell"]]:
            raise AssertionError(f"wrong delete semantics for {cell['cell']}")
    covered = set(HORIZONS)
    if set(BLOCKS["A"]) | set(BLOCKS["B"]) | set(BLOCKS["C"]) != covered:
        raise AssertionError("block coverage is incomplete")
    if any(set(BLOCKS[left]) & set(BLOCKS[right]) for left, right in combinations(BLOCKS, 2)):
        raise AssertionError("block overlap is non-empty")

    synthetic = {}
    for bits in product((0, 1), repeat=4):
        x32, a, b, c = bits
        synthetic[cell_id(bits)] = (
            0.37 * x32 + 0.11 * a - 0.23 * b + 0.19 * c
            + 0.41 * x32 * a - 0.17 * x32 * b + 0.29 * x32 * a * b
            - 0.13 * x32 * a * c + 0.07 * x32 * b * c + 0.53 * x32 * a * b * c
            + 0.05 * a * b - 0.09 * a * c + 0.15 * b * c
        )
    interactions = mobius_interactions(synthetic)
    components = seven_h32_components(interactions)
    i32_rest = synthetic["1111"] - synthetic["1000"] - synthetic["0111"]
    component_sum = sum(components.values())
    if not math.isclose(i32_rest, component_sum, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("synthetic H32 block reconstruction failed")
    pair_terms = {h: (0.003 * h if h % 2 else -0.002 * h) for h in HORIZONS}
    p_a = sum(pair_terms[h] for h in BLOCKS["A"])
    p_b = sum(pair_terms[h] for h in BLOCKS["B"])
    p_c = sum(pair_terms[h] for h in BLOCKS["C"])
    remainder = i32_rest - sum(pair_terms.values())
    q_a = components["M_32_A"] - p_a
    q_b = components["M_32_B"] - p_b
    q_c = components["M_32_C"] - p_c
    reconstructed_remainder = q_a + q_b + q_c + components["M_32_A_B"] + components["M_32_A_C"] + components["M_32_B_C"] + components["M_32_A_B_C"]
    if not math.isclose(remainder, reconstructed_remainder, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("synthetic higher-order remainder reconstruction failed")
    return {
        "status": "PASS",
        "scientific_updates_executed": False,
        "factor_registry": {"unique_cells": len(set(ids)), "cell_count": len(ids), "all_binary_vectors_present": True},
        "set_semantics": {"block_disjoint": True, "complete_H2_H31_coverage": True, "A_size": 16, "B_size": 9, "C_size": 5},
        "synthetic_mobius": {"status": "PASS", "h32_rest_reconstruction_residual": i32_rest - component_sum, "remainder_reconstruction_residual": remainder - reconstructed_remainder},
    }


def _load_design(design_dir: Path) -> dict[str, Any]:
    design = read_json(design_dir / "experiment_design.json")
    if design.get("design_review_only") is not True or design.get("no_tuning") is not True:
        raise RuntimeError("design review contract is not sealed")
    if design.get("expected_new_branch_count") != 12:
        raise RuntimeError("future branch count is not 12")
    return design


def _authorized(authorization: Path) -> bool:
    text = authorization.read_text(encoding="utf-8")
    required = ("D20_EXECUTION_AUTHORIZED: YES", "UPDATE_393_AUTHORIZED: NO", "FROZEN20_AUTHORIZED: NO")
    return all(marker in text for marker in required)


def _scientific_branch_update(
    d17: Any,
    branch: str,
    deleted_horizons: Sequence[int],
    model: Any,
    optimizer: Any,
    state: Mapping[str, Any],
    zero_index: int,
    rng_before: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """D17 branch_update with only the deletion-set resolver generalized.

    The forward, loss, diagnostics, clipping, AdamW, validation, RNG, and
    finite-state operations are the authenticated D17 primitive.  The sole
    D20 extension is subtracting the exact weighted terms in a frozen block
    deletion set before backward, with no renormalization.
    """
    import torch
    import torch.nn.functional as F

    update = zero_index + 1
    if update not in d17.UPDATES:
        raise RuntimeError(f"UNAUTHORIZED_UPDATE:{update}")
    d17.base.restore_rng(rng_before)
    pre = state["pre"]
    authority = state["authority"]
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(d17.ALL_NAMES):
        raise RuntimeError(f"BRANCH_PARAMETER_ORDER_MISMATCH:{branch}:{update}")
    batch_record = d17.d16.batch_identity(authority["schedule_manifest"], zero_index)
    batch = d17.base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    before_model = d17.d16.clone_parameters(model)
    before_optimizer = d17.d16.clone_named_optimizer_state(optimizer)
    before_model_hash = d17.canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
    before_steps = d17.d16.optimizer_steps(before_optimizer)
    optimizer.zero_grad(set_to_none=True)
    rollout = d17.base.causal_paired_rollout(
        model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"],
        batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=d17.ACTIVE_HORIZON,
        target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True,
    )
    if rollout.get("free_branch_reads_target_positions") is not False or rollout.get("teacher_reference_stop_gradient") is not True:
        raise RuntimeError(f"GRAPH_SEMANTICS_GATE_FAILED:{branch}:{update}")
    predictions = rollout["predictions"]
    targets = batch["target_positions"][:, : d17.ACTIVE_HORIZON, :]
    step_losses = F.smooth_l1_loss(predictions, targets, beta=d17.POSITION_BETA, reduction="none").mean(dim=(0, 2))
    weights = d17.rollout_horizon_weights(d17.ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
    weighted_terms = step_losses * weights
    full_rollout_loss = torch.sum(weighted_terms)
    h1_loss = step_losses[0]
    h32_term = weighted_terms[-1]
    hidden_loss = rollout["hidden_loss"]
    deleted = sorted(set(int(h) for h in deleted_horizons))
    if any(h < 2 or h > 32 for h in deleted):
        raise RuntimeError(f"INVALID_D20_DELETE_SET:{deleted}")
    rollout_loss = full_rollout_loss - sum((weighted_terms[h - 1] for h in deleted), start=weighted_terms[0] * 0.0)
    deletion_identity = "DELETED_weighted_rollout_" + ",".join(f"h{h}" for h in deleted) + "_BEFORE_BACKWARD"
    total_loss = rollout_loss + d17.LAMBDA_H1 * h1_loss + d17.LAMBDA_HIDDEN * hidden_loss
    if any(not bool(torch.isfinite(value)) for value in (full_rollout_loss, rollout_loss, h1_loss, hidden_loss, total_loss)):
        raise RuntimeError(f"NONFINITE_OBJECTIVE:{branch}:{update}")
    h1_gradients, _ = d17.d16.diagnostic_gradients(h1_loss, named)
    h32_gradients, _ = d17.d16.diagnostic_gradients(h32_term, named)
    rng_after_diagnostics = d17.base.capture_rng()
    if d17.base.rng_digest(rng_after_diagnostics) != d17.base.rng_digest(rng_before):
        raise RuntimeError(f"DIAGNOSTIC_RNG_CONTAMINATION:{branch}:{update}")
    total_loss.backward()
    if not d17.base.gradients_are_finite(model):
        raise RuntimeError(f"NONFINITE_RAW_GRADIENT:{branch}:{update}")
    raw_gradients, none_flags = d17.d16.capture_gradients(named)
    raw_norm = d17.d16.norm(raw_gradients.values())
    returned_preclip = d17.finite(torch.nn.utils.clip_grad_norm_(model.parameters(), d17.CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    if not math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5):
        raise RuntimeError(f"CLIP_RETURNED_NORM_MISMATCH:{branch}:{update}")
    postclip_gradients, _ = d17.d16.capture_gradients(named)
    postclip_norm = d17.d16.norm(postclip_gradients.values())
    clip_coefficient = 1.0 if returned_preclip <= d17.CLIP_MAX_NORM else min(1.0, d17.CLIP_MAX_NORM / (returned_preclip + 1.0e-6))
    residual = d17.d16.norm(postclip_gradients[name] - raw_gradients[name] * clip_coefficient for name in d17.ALL_NAMES)
    if residual > 1.0e-7 + 2.0e-5 * max(postclip_norm, 1.0):
        raise RuntimeError(f"POSTCLIP_RECONSTRUCTION_FAILED:{branch}:{update}")
    rng_before_optimizer = d17.base.capture_rng()
    optimizer.step()
    after_model = d17.d16.clone_parameters(model)
    after_optimizer = d17.d16.clone_named_optimizer_state(optimizer)
    after_model_hash = d17.canonical_model_state_sha256(model.state_dict())
    after_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
    if not d17.d16.state_finite(model, optimizer):
        raise RuntimeError(f"NONFINITE_STATE_AFTER_UPDATE:{branch}:{update}")
    rng_after_training = d17.base.capture_rng()
    if d17.base.rng_digest(rng_before_optimizer) != d17.base.rng_digest(rng_after_training):
        raise RuntimeError(f"OPTIMIZER_RNG_CONSUMPTION_UNMATCHED:{branch}:{update}")
    adamw_metrics, adamw_tensors = d17.d7.adamw_decomposition(model, optimizer, before_model, postclip_gradients, after_model, state["table"])
    validation, validation_meta = d17.d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    rng_after_validation = d17.base.capture_rng()
    if validation_meta["rng_unchanged_or_restored"] not in ("YES", "RESTORED"):
        raise RuntimeError(f"VALIDATION_RNG_GATE_FAILED:{branch}:{update}")
    after_steps = d17.d16.optimizer_steps(after_optimizer)
    parameter_delta = {name: after_model[name] - before_model[name] for name in d17.ALL_NAMES}
    record = {
        "branch": branch, "update": update, "schedule_index": zero_index, "batch_identity": batch_record,
        "batch_window_sha256": batch_record["batch_window_sha256"], "model_state_hash_before": before_model_hash,
        "model_state_hash_after": after_model_hash, "optimizer_state_hash_before": before_optimizer_hash,
        "optimizer_state_hash_after": after_optimizer_hash, "rng_before_digest": d17.base.rng_digest(rng_before),
        "rng_after_diagnostics_digest": d17.base.rng_digest(rng_after_diagnostics),
        "rng_after_training_digest": d17.base.rng_digest(rng_after_training),
        "rng_after_validation_digest": d17.base.rng_digest(rng_after_validation),
        "rollout_loss_full": d17.finite(full_rollout_loss), "rollout_loss_used_for_backward": d17.finite(rollout_loss),
        "weighted_rollout_h32_term": d17.finite(h32_term), "explicit_h1_loss": d17.finite(h1_loss),
        "hidden_loss_unweighted": d17.finite(hidden_loss), "weighted_hidden_loss": d17.finite(d17.LAMBDA_HIDDEN * hidden_loss),
        "total_loss": d17.finite(total_loss), "deleted_horizons": deleted,
        "objective_deletion_identity": deletion_identity, "objective_renormalization": "NO",
        "raw_total_gradient_norm": raw_norm, "returned_pre_clipping_total_norm": returned_preclip,
        "clipping_activated": "YES" if returned_preclip > d17.CLIP_MAX_NORM else "NO",
        "exact_clip_coefficient": clip_coefficient, "post_clipping_gradient_norm": postclip_norm,
        "postclip_reconstruction_residual_norm": residual,
        "raw_gradient_digest": d17.d7.d6.named_tensor_digest(raw_gradients),
        "post_clipping_gradient_digest": d17.d7.d6.named_tensor_digest(postclip_gradients),
        "gradient_none_pattern": none_flags,
        "gradient_none_pattern_hash": hashlib.sha256(json.dumps(none_flags, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "h1_gradient_norm": d17.d16.norm(h1_gradients.values()), "h32_gradient_norm": d17.d16.norm(h32_gradients.values()),
        "weighted_h32_gradient_norm": d17.d16.norm(h32_gradients[name] * float(weights[-1]) for name in d17.ALL_NAMES),
        "h1_gradient_norm_by_block": d17.d16.block_norms(h1_gradients), "h32_gradient_norm_by_block": d17.d16.block_norms(h32_gradients),
        "raw_gradient_norm_by_block": d17.d16.block_norms(raw_gradients), "postclip_gradient_norm_by_block": d17.d16.block_norms(postclip_gradients),
        "parameter_update_norm": d17.d16.norm(parameter_delta.values()),
        "parameter_update_norm_by_block": d17.d16.block_norms(parameter_delta),
        "parameter_update_digest": d17.d7.d6.named_tensor_digest(parameter_delta),
        "loss_driven_adamw_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["adaptive_update_norm"],
        "decoupled_weight_decay_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["decay_update_norm"],
        "adamw_decomposition_residual_norm": adamw_metrics["aggregate"]["whole_model"]["decomposition_residual_norm"],
        "adamw_decomposition_status": adamw_metrics["aggregate"]["whole_model"]["decomposition_pass_fail"],
        "step_counter_before": before_steps, "step_counter_after": after_steps, "exp_avg_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_norm"],
        "exp_avg_sq_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_sq_norm"], "adamw": adamw_metrics,
        "validation": validation, "per_update_H1": validation["1"], "per_update_H32": validation["32"],
        "finite_status": "FINITE", "branch_rng_contract_status": "PASS",
    }
    capture = {"branch": branch, "update": update, "model_before": before_model, "model_after": after_model, "optimizer_before": before_optimizer, "optimizer_after": after_optimizer, "raw_gradients": raw_gradients, "postclip_gradients": postclip_gradients, "parameter_delta": parameter_delta, "h1_gradients_diagnostic": h1_gradients, "h32_gradients_diagnostic": h32_gradients, "adamw_decomposition_tensors": adamw_tensors}
    return record, capture, copy.deepcopy(rng_after_validation)


def execute_scientific(design_dir: Path, authorization: Path, output: Path) -> int:
    if not _authorized(authorization):
        raise RuntimeError("D20_EXECUTION_AUTHORIZATION_MARKER_MISSING")
    design = _load_design(design_dir)
    if output.exists():
        raise RuntimeError(f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True)
    sys.path.insert(0, str(ROOT))
    from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17

    d17.verify_design_and_inputs()
    state = d17.preflight_runtime()
    d17_dir = Path(design["authoritative_d17_execution_path"])
    branch_manifest = read_json(design_dir / "branch_manifest.json")
    authorized_cells = list(design["future_branch_cells"])
    future_specs = [
        spec for spec in branch_manifest["branches"]
        if spec.get("cell") in authorized_cells
        and spec.get("endpoint_status") == "NOT_EXECUTED_BY_DESIGN_REVIEW"
        and spec.get("source_stage") == "D20"
    ]
    if [spec["cell"] for spec in future_specs] != authorized_cells:
        raise RuntimeError("D20_FUTURE_BRANCH_SPEC_RESOLUTION_FAILED")
    if len(future_specs) != design["expected_new_branch_count"]:
        raise RuntimeError("D20_FUTURE_BRANCH_COUNT_MISMATCH")
    control_rows = [json.loads(line) for line in (d17_dir / "per_update_instrumentation.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line).get("branch") == "CONTROL"]
    control_by_update = {int(row["update"]): row for row in control_rows}
    if set(control_by_update) != set(UPDATES):
        raise RuntimeError("AUTHENTICATED_D17_CONTROL_REFERENCE_INCOMPLETE")

    all_rows: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for spec in future_specs:
        model, optimizer = d17.d16.new_branch(state["bundle"])
        start_model_hash = d17.canonical_model_state_sha256(model.state_dict())
        start_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
        if start_model_hash != design["certified_start"]["model_semantic_hash"] or start_optimizer_hash != design["certified_start"]["optimizer_semantic_hash"]:
            raise RuntimeError(f"BRANCH_START_STATE_IDENTITY_MISMATCH:{spec['cell']}")
        rng = copy.deepcopy(state["bundle"]["rng_state"])
        branch_rows: list[dict[str, Any]] = []
        for zero_index in range(384, 392):
            row, _, rng = _scientific_branch_update(d17, f"D20_{spec['cell']}", [int(h.removeprefix("H")) for h in spec["delete_horizons"] if h != "H32" or True], model, optimizer, state, zero_index, rng)
            control = control_by_update[row["update"]]
            for field in ("batch_window_sha256", "rng_before_digest", "rng_after_diagnostics_digest", "rng_after_training_digest", "rng_after_validation_digest"):
                if row[field] != control[field]:
                    raise RuntimeError(f"CONTROL_MATCH_FAILED:{spec['cell']}:{row['update']}:{field}")
            branch_rows.append(row)
            all_rows.append(row)
        endpoint = branch_rows[-1]
        e_h1 = float(design["control_h1"]) - float(endpoint["per_update_H1"])
        endpoints.append({"cell": spec["cell"], "branch": f"D20_{spec['cell']}", "H1": endpoint["per_update_H1"], "H32": endpoint["per_update_H32"], "E": e_h1, "H1_material": e_h1 >= H1_MATERIALITY_THRESHOLD, "H32_preserved": endpoint["per_update_H32"] <= H32_PRESERVATION_THRESHOLD})
        manifests.append({"cell": spec["cell"], "branch": f"D20_{spec['cell']}", "start_model_hash": start_model_hash, "start_optimizer_hash": start_optimizer_hash, "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO"})
    (output / "per_update_instrumentation.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) for row in all_rows) + "\n", encoding="utf-8", newline="\n")
    write_json(output / "endpoint_results.json", {"control": {"H1": design["control_h1"], "H32": design["control_h32"], "source": "AUTHENTICATED_D17_REUSE"}, "new_branch_results": endpoints, "endpoint_update": 392, "updates": list(UPDATES), "update_393_executed": "NO"})
    write_json(output / "branch_execution_manifest.json", {"scientific_branches_authorized": authorized_cells, "scientific_branches_executed": authorized_cells, "branches": manifests, "start_update": 384, "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO"})
    write_json(output / "protocol_integrity.json", {"status": "PASSED", "first_blocker": "none", "d20_execution_authorized": "YES", "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO", "hyperparameter_tuning": "NO"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--execute-scientific", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    design_dir = args.design_dir.resolve()
    try:
        if args.execute_scientific:
            if args.authorization is None or args.output is None:
                raise RuntimeError("--execute-scientific requires --authorization and --output")
            return execute_scientific(design_dir, args.authorization.resolve(), args.output.resolve())
        result = static_validation()
        _load_design(design_dir)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}", "scientific_updates_executed": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
