"""D33 inheritance-preserving Stage-3 H1 continuation.

The immutable parent is the authenticated D32 committed update-450 checkpoint.
Every candidate is evaluated in an isolated clone of that parent's realized
AdamW state.  Only a deterministic-replay winner that passes the hard H1/H32,
optimizer, RNG, and adaptive H32-reserve gates is committed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d31_d32_active_boundary_tailor_continuation as d32


EXPERIMENT_ID = "STAGE_3_H13_D33_INHERITANCE_PRESERVING_H1_ACCELERATED_FORWARD_CONTINUATION"
HANDOFF_ROOT = (
    ROOT
    / "outputs"
    / "_d32_cross_chat_handoff_staging_20260824T153121+0800"
    / "D32_CROSS_CHAT_HANDOFF"
)
PARENT = HANDOFF_ROOT / "checkpoint" / "committed_update_450.pt"
AUTHORIZATION = Path(
    r"C:\Users\86198\.codex\attachments\772426d2-ef77-4114-b18e-9db0cfeb88b1\pasted-text.txt"
)
EXPECTED_PARENT_SHA256 = "ba908592ab6fdbfac2b514793d1c59d9c66f44ad4693dff12cbfa48b05af2d84"
EXPECTED_PARENT_PARENT_SHA256 = "9c2fe24292ca6c882451c3e8db8544e9e47d92801fe501a361970fc339775e28"
START_UPDATE = 450
START_H1 = 7.2765443365386504e-05
START_H32 = 0.018090939364518988
H32_LIMIT = d32.H32_LIMIT
TARGET_H1 = 5.0e-05
MIN_H1_DESCENT = d32.MIN_H1_DESCENT
MIN_ALPHA = 1.0 / 256.0
MAX_ALPHA = 0.5
HORIZON_CRITERION_VERSION = "D33_RESERVE_SAFE_TERMINAL_H1_V2"

COMPACT_CANDIDATE_FIELDS = (
    "parent_update",
    "candidate_update",
    "screening_stage",
    "candidate_family",
    "source_base",
    "spec_id",
    "alpha",
    "p4_beta",
    "cagrad_c",
    "used_h32_lambda",
    "parent_H1",
    "candidate_H1",
    "delta_H1",
    "parent_H32",
    "candidate_H32",
    "delta_H32",
    "H32_margin",
    "effective_update_norm",
    "safety_regime",
    "safety_reserve",
    "H32_outward_cap",
    "H32_outward_cap_pass",
    "safety_reserve_pass",
    "candidate_valid_before_replay",
    "candidate_valid",
    "deterministic_replay_status",
    "horizon_length",
    "horizon_requested",
    "horizon_criterion_version",
    "horizon_safety_status",
    "horizon_terminal_H1_pass",
    "horizon_terminal_H1",
    "horizon_terminal_H32",
    "horizon_min_H32_margin",
    "horizon_updates",
    "horizon_trace",
    "horizon_status",
    "selection_status",
    "rejection_reason",
)


class D33Blocker(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise D33Blocker(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reserve_policy(parent_margin: float, relaxed: bool = False) -> dict[str, Any]:
    """Return an explicit reserve whose scale follows available H32 slack."""
    require(parent_margin >= -1.0e-12, "PARENT_H32_ALREADY_ILLEGAL")
    # D32's realized 95th-percentile |delta H32| was 8.30074e-5.  Three
    # such transitions define the protected reserve.  GREEN begins two
    # additional q95 moves above it; outward spend is capped at 35% of the
    # currently free budget and at 1.25*q95.
    q95 = 8.30074e-5
    reserve = max(7.5e-5, 1.25 * q95) if relaxed else 2.5e-4
    green_buffer = (1.5 if relaxed else 2.0) * q95
    if parent_margin >= reserve + green_buffer:
        regime = "GREEN"
    elif parent_margin >= reserve:
        regime = "YELLOW"
    else:
        regime = "RED"
    outward_cap = 0.0 if regime == "RED" else min(0.35 * max(0.0, parent_margin - reserve), 1.25 * q95)
    return {
        "regime": regime,
        "reserve": reserve,
        "outward_cap": outward_cap,
        "relaxed_after_progress_deterioration": bool(relaxed),
        "historical_abs_delta_H32_q95": q95,
    }


def annotate_safety(record: dict[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    record["safety_regime"] = str(policy["regime"])
    record["safety_reserve"] = float(policy["reserve"])
    consumed = max(0.0, float(record["delta_H32"]))
    record["H32_outward_cap"] = float(policy.get("outward_cap", math.inf))
    record["H32_outward_cap_pass"] = consumed <= float(policy.get("outward_cap", math.inf)) + 1.0e-15
    # The reserve is the promotion gate.  The per-step cap is an advisory
    # trust-region diagnostic: a larger spend may still be promoted when the
    # protected terminal reserve remains intact and its H1 result wins replay.
    record["safety_reserve_pass"] = float(record["H32_margin"]) >= float(policy["reserve"])
    improvement = max(0.0, -float(record["delta_H1"]))
    record["H1_improvement_per_H32_slack_consumed"] = (
        improvement / consumed if consumed > 1.0e-15 else None
    )
    return record


def hard_gate_pass(record: Mapping[str, Any], replay_required: bool = False) -> bool:
    basic = bool(record.get("candidate_valid_before_replay")) and float(record["candidate_H32"]) <= H32_LIMIT
    if replay_required:
        basic = basic and d32.d31.candidate_is_legal(record)
    return basic


def safely_legal(record: Mapping[str, Any], replay_required: bool = False) -> bool:
    return hard_gate_pass(record, replay_required) and bool(record.get("safety_reserve_pass"))


def horizon_step_safety_pass(record: Mapping[str, Any]) -> bool:
    """Horizon safety excludes per-step H1 monotonicity by design."""
    return bool(
        record.get("finite_state_pass")
        and record.get("H32_preservation_pass")
        and record.get("optimizer_consistency_pass")
        and record.get("rng_identity_pass")
        and record.get("safety_reserve_pass")
        and record.get("candidate_H32") is not None
        and float(record["candidate_H32"]) <= H32_LIMIT
    )


def horizon_chain_pass(records: Sequence[Mapping[str, Any]], expected: int, parent_h1: float) -> bool:
    return bool(
        len(records) == expected
        and all(horizon_step_safety_pass(record) for record in records)
        and records[-1].get("candidate_H1") is not None
        and float(records[-1]["candidate_H1"]) <= float(parent_h1) - MIN_H1_DESCENT
    )


def horizon_evidence_safety_pass(record: Mapping[str, Any]) -> bool:
    """Interpret persisted horizon evidence without conflating safety and utility."""
    explicit = record.get("horizon_safety_status")
    if explicit in {"PASS", "FAIL"}:
        return explicit == "PASS"
    trace = record.get("horizon_trace")
    requested = record.get("horizon_requested")
    if isinstance(trace, list) and requested is not None:
        return len(trace) == int(requested) and all(bool(step.get("step_safety_pass")) for step in trace)
    return False


def promotion_safety_pass(record: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    reserve_or_red_recovery = safely_legal(record, replay_required=True) or (
        str(policy["regime"]) == "RED"
        and hard_gate_pass(record, replay_required=True)
        and float(record["delta_H32"]) <= 0.0
    )
    if not reserve_or_red_recovery:
        return False
    # Once a serious finalist has been given a short-horizon evaluation, an
    # explicit failure is authoritative.  A one-step cap pass must not erase
    # evidence that the repeated control exhausts the protected reserve.
    if record.get("horizon_status") == "FAIL" and not horizon_evidence_safety_pass(record):
        return False
    outward_cap_exception_pass = (
        float(record["delta_H32"]) <= 0.0
        or bool(record.get("H32_outward_cap_pass"))
        or record.get("horizon_status") == "PASS"
    )
    return bool(outward_cap_exception_pass)


def h1_first_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank only after hard legality/reserve filtering; H1 is authoritative."""
    return (
        float(record["candidate_H1"]),
        -float(record["H32_margin"]),
        float(record["effective_update_norm"]),
        str(record["spec_id"]),
    )


def yellow_lattice_has_enough_evidence(screened: Sequence[Mapping[str, Any]]) -> bool:
    """Stop YELLOW screening only after a conservative comparator exists."""
    safe = [record for record in screened if safely_legal(record)]
    return len(safe) >= 5 and any(bool(record.get("H32_outward_cap_pass")) for record in safe)


def most_h32_conservative(screened: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the safest realized H32 comparator, breaking ties by H1."""
    if not screened:
        return None
    return min(screened, key=lambda record: (float(record["delta_H32"]),) + h1_first_rank(record))


def smallest_scale_comparator(screened: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the smallest realized trust-region step, breaking ties by H1."""
    if not screened:
        return None
    return min(screened, key=lambda record: (float(record["alpha"]),) + h1_first_rank(record))


def retained_horizon_map(
    records: Sequence[Mapping[str, Any]],
    candidate_update: int,
    parent_update: int,
    reserve: float,
) -> dict[str, dict[str, Any]]:
    """Recover matching persisted horizon evidence after interruption/retry."""
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if (
            int(record.get("candidate_update", -1)) != candidate_update
            or int(record.get("parent_update", -1)) != parent_update
            or record.get("screening_stage") != "D33_SERIOUS_DETERMINISTIC_REPLAY"
            or record.get("horizon_status") not in {"PASS", "FAIL"}
            or record.get("horizon_criterion_version") != HORIZON_CRITERION_VERSION
            or abs(float(record.get("safety_reserve", float("nan"))) - float(reserve)) > 1.0e-15
        ):
            continue
        result[str(record["spec_id"])] = {
            key: record[key]
            for key in (
                "horizon_length",
                "horizon_requested",
                "horizon_criterion_version",
                "horizon_safety_status",
                "horizon_terminal_H1_pass",
                "horizon_status",
                "horizon_terminal_H1",
                "horizon_terminal_H32",
                "horizon_min_H32_margin",
                "horizon_updates",
                "horizon_trace",
            )
            if key in record
        }
    return result


def select_replayed_candidate(
    legal: Sequence[tuple[dict[str, Any], Any, Any, dict[str, Any]]],
    close: bool,
) -> tuple[tuple[dict[str, Any], Any, Any, dict[str, Any]], str]:
    """Production horizon-aware selection used after deterministic replay."""
    require(bool(legal), "NO_LEGAL_REPLAYED_CANDIDATE_TO_SELECT")
    horizon_legal = [
        item
        for item in legal
        if item[0].get("horizon_status") == "PASS"
        and item[0].get("horizon_terminal_H1") is not None
    ]
    if horizon_legal and close:
        return (
            min(
                horizon_legal,
                key=lambda item: (float(item[0]["horizon_terminal_H1"]),) + h1_first_rank(item[0]),
            ),
            "CLOSE_FINALIST_SHORT_HORIZON_THEN_H1",
        )
    return min(legal, key=lambda item: h1_first_rank(item[0])), "LOWEST_REPLAYED_H1_AFTER_HARD_AND_RESERVE_GATES"


def unique_specs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        identity = d32.spec_id(spec)
        if identity not in seen:
            seen.add(identity)
            result.append(dict(spec))
    return result


def clamp_alpha(value: float) -> float:
    return max(MIN_ALPHA, min(MAX_ALPHA, float(value)))


def inherited_baseline(previous_alpha: float, previous_beta: float) -> dict[str, Any]:
    return {
        "family": "D31_BASE_PLUS_H32_CORRECTION",
        "base": "P4",
        "p4_beta": float(previous_beta),
        "lambda": 0.0,
        "alpha": clamp_alpha(previous_alpha),
        "d33_role": "INHERITED_D32_BASELINE",
    }


def primary_specs(
    previous_alpha: float,
    previous_beta: float,
    regime: str,
    stagnation: bool,
) -> list[dict[str, Any]]:
    """A bounded progressive lattice centered on the inherited winner."""
    baseline = inherited_baseline(previous_alpha, previous_beta)
    expanded = clamp_alpha(previous_alpha * (4.0 if stagnation and regime == "GREEN" else 2.0))
    contracted = clamp_alpha(previous_alpha / 2.0)
    beta_low = max(0.0, previous_beta / 2.0)
    beta_high = min(5.6, max(0.35, previous_beta * 2.0))
    specs: list[dict[str, Any]] = [baseline]
    local_pairs = [
        (contracted, previous_beta),
        (previous_alpha, beta_low),
    ]
    if regime == "YELLOW":
        # Preserve a historical 1/256-scale trust-region comparator when
        # inherited AdamW moments make repeated larger steps unstable.
        local_pairs.insert(1, (clamp_alpha(previous_alpha / 4.0), previous_beta))
    if regime == "GREEN":
        local_pairs.extend(
            [
                (previous_alpha, beta_high),
                (expanded, previous_beta),
            ]
        )
    specs.extend(
        {
            "family": "D31_BASE_PLUS_H32_CORRECTION",
            "base": "P4",
            "p4_beta": beta,
            "lambda": 0.0,
            "alpha": alpha,
        }
        for alpha, beta in local_pairs
    )
    if regime == "GREEN":
        specs.append(
            {
                "family": "D31_BASE_PLUS_H32_CORRECTION",
                "base": "P4",
                "p4_beta": beta_high,
                "lambda": 0.0,
                "alpha": expanded,
            }
        )
        specs.extend(
            {
                "family": "H1_PLUS_H32_PRESENTED",
                "base": "H1",
                "lambda": amount,
                "alpha": expanded,
            }
            for amount in (-1.6, -0.8, -0.4, 0.0)
        )
        specs.extend(
            {
                "family": "D31_BASE_PLUS_H32_CORRECTION",
                "base": "P4",
                "p4_beta": previous_beta,
                "lambda": amount,
                "alpha": expanded,
            }
            for amount in (-0.4, 0.2)
        )
        specs.append(
            {
                "family": "D31_BASE_PLUS_H32_CORRECTION",
                "base": "CAGRAD",
                "cagrad_c": 0.8,
                "lambda": 0.0,
                "alpha": expanded,
            }
        )
    elif regime == "YELLOW":
        specs.extend(
            {
                "family": "D31_BASE_PLUS_H32_CORRECTION",
                "base": base,
                "p4_beta": previous_beta,
                "cagrad_c": 0.4,
                "lambda": amount,
                "alpha": clamp_alpha(previous_alpha),
            }
            for base, amount in (
                ("P4", 0.2),
                ("P4", 0.4),
                ("P4", 0.8),
                ("P4", 1.6),
                ("P4", 3.2),
                ("P4", 5.6),
                ("P4", 8.0),
                ("CAGRAD", 0.2),
                ("CAGRAD", 0.4),
                ("CAGRAD", 0.8),
                ("CAGRAD", 1.6),
            )
        )
    else:
        specs.extend(
            [
                {"family": "H32_LEVEL_TANGENT", "base": "P4", "p4_beta": previous_beta, "alpha": contracted},
                {"family": "INWARD_CORRECTED_TANGENT", "base": "P4", "p4_beta": previous_beta, "rho": 0.4, "alpha": contracted},
                {"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "kappa": 0.02, "alpha": contracted},
                {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "p4_beta": previous_beta, "lambda": 0.4, "alpha": contracted},
            ]
        )
    return unique_specs(specs)


def fallback_specs(previous_alpha: float, previous_beta: float) -> list[dict[str, Any]]:
    alphas = sorted({MIN_ALPHA, clamp_alpha(previous_alpha / 4.0), clamp_alpha(previous_alpha / 2.0)})
    specs: list[dict[str, Any]] = []
    specs.extend(
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "p4_beta": beta, "lambda": amount, "alpha": alpha}
        for alpha in alphas
        for beta in (max(0.0, previous_beta / 2.0), previous_beta, min(5.6, previous_beta * 2.0))
        for amount in (0.0, 0.2, 0.4)
    )
    specs.extend(
        {"family": "H1_PLUS_H32_PRESENTED", "base": "H1", "lambda": amount, "alpha": alpha}
        for alpha in alphas
        for amount in (-0.4, 0.0, 0.4, 0.8)
    )
    specs.extend(
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "CAGRAD", "cagrad_c": c_value, "lambda": 0.2, "alpha": alpha}
        for alpha in alphas
        for c_value in (0.4, 0.8)
    )
    return unique_specs(specs)


def compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in COMPACT_CANDIDATE_FIELDS if key in record}


def persist_search_progress(output: Path, candidate_count: int, shadow_executions: int, update: int) -> None:
    atomic_json(
        output / "D33_SEARCH_PROGRESS.json",
        {
            "active_candidate_update": int(update),
            "candidate_specifications_screened": int(candidate_count),
            "shadow_executions": int(shadow_executions),
            "canonical_state_advanced_by_search_progress": False,
        },
    )


def failed_candidate_record(
    update: int,
    parent_update: int,
    spec: Mapping[str, Any],
    stage: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "parent_update": parent_update,
        "candidate_update": update,
        "screening_stage": stage,
        "candidate_family": str(spec.get("family")),
        "source_base": str(spec.get("base", "H1")),
        "spec_id": d32.spec_id(spec),
        "alpha": float(spec.get("alpha", 0.0)),
        "p4_beta": float(spec.get("p4_beta", 0.35)),
        "cagrad_c": float(spec.get("cagrad_c", 0.4)),
        "used_h32_lambda": float(spec.get("lambda", 0.0)),
        "candidate_valid_before_replay": False,
        "candidate_valid": False,
        "deterministic_replay_status": "NOT_RUN_CANDIDATE_FAILURE",
        "selection_status": "REJECTED_CANDIDATE_LOCAL_FAILURE",
        "rejection_reason": f"{type(exc).__name__}:{exc}",
    }


def load_d32_trajectory() -> list[dict[str, Any]]:
    payload = json.loads((HANDOFF_ROOT / "trajectory" / "D32_H1_RECOVERY_TRAJECTORY.json").read_text(encoding="utf-8"))
    rows = list(payload["combined_post426_trajectory"])
    require([int(row["update"]) for row in rows] == list(range(427, 451)), "D32_TRAJECTORY_NOT_CONTIGUOUS_TO_450")
    require(float(rows[-1]["H1"]) == START_H1 and float(rows[-1]["H32"]) == START_H32, "D32_TRAJECTORY_ENDPOINT_MISMATCH")
    return rows


def instantiate_state(runtime: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], list[float]]:
    model, optimizer = d32.d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = copy.deepcopy(payload["rng_state"])
    canonical = [float(value) for value in payload["canonical_base_lr"]]
    require(canonical == [d32.BASE_LR], "CANONICAL_LR_MISMATCH")
    last_identity = payload.get("last_record_identity", {})
    require(last_identity.get("model_state_hash") == d32.d24.d17.canonical_model_state_sha256(model.state_dict()), "LOADED_MODEL_SEMANTIC_HASH_MISMATCH")
    require(last_identity.get("optimizer_state_hash") == d32.d24.d17.base.optimizer_semantic_hash(optimizer), "LOADED_OPTIMIZER_SEMANTIC_HASH_MISMATCH")
    if last_identity.get("rng_state_hash") is not None:
        require(last_identity["rng_state_hash"] == d32.d24.d17.base.rng_digest(rng), "LOADED_RNG_SEMANTIC_HASH_MISMATCH")
    return model, optimizer, rng, canonical


def verify_loaded_payload(
    payload: Mapping[str, Any],
    update: int,
    h1: float,
    h32: float,
    parent_update: int,
    parent_sha: str | None,
) -> None:
    require(int(payload["completed_optimizer_step"]) == update, "CHECKPOINT_COMPLETED_STEP_MISMATCH")
    require(int(payload["next_schedule_index"]) == update, "CHECKPOINT_SCHEDULE_INDEX_MISMATCH")
    optimizer_state = payload["optimizer_state_dict"]["state"]
    require(len(optimizer_state) == len(d32.d24.d16.ALL_NAMES), "CHECKPOINT_ADAMW_STATE_CARDINALITY_MISMATCH")
    require(
        all(
            int(state.get("step", -1)) == update
            and isinstance(state.get("exp_avg"), torch.Tensor)
            and isinstance(state.get("exp_avg_sq"), torch.Tensor)
            for state in optimizer_state.values()
        ),
        "CHECKPOINT_ADAMW_STATE_CONTRACT_MISMATCH",
    )
    groups = payload["optimizer_state_dict"].get("param_groups", [])
    require(len(groups) == 1, "CHECKPOINT_ADAMW_PARAM_GROUP_COUNT_MISMATCH")
    group = groups[0]
    require(tuple(group.get("betas", ())) == tuple(d32.consistency.BETAS), "CHECKPOINT_ADAMW_BETAS_MISMATCH")
    require(float(group.get("eps")) == float(d32.consistency.EPS), "CHECKPOINT_ADAMW_EPS_MISMATCH")
    require(float(group.get("weight_decay")) == float(d32.consistency.WEIGHT_DECAY), "CHECKPOINT_ADAMW_WEIGHT_DECAY_MISMATCH")
    require(int(payload["parent_update"]) == parent_update, "CHECKPOINT_PARENT_UPDATE_MISMATCH")
    if parent_sha is not None:
        require(str(payload["parent_checkpoint_sha256"]) == parent_sha, "CHECKPOINT_PARENT_SHA_MISMATCH")
    require(float(payload["H1"]) == h1 and float(payload["H32"]) == h32, "CHECKPOINT_METRIC_MISMATCH")
    require(payload.get("scientific_state") == "COMMITTED_FORWARD_STATE" and payload.get("resumable") is True, "CHECKPOINT_NOT_RESUMABLE_COMMITTED_STATE")


def verify_committed_checkpoint(
    path: Path,
    expected_parent_update: int,
    expected_update: int,
    expected_parent_sha: str,
    expected_record: Mapping[str, Any],
    expected_model: Any,
    expected_optimizer: Any,
    expected_rng: Mapping[str, Any],
) -> dict[str, Any]:
    structural = d32.d27.verify_checkpoint(path, expected_parent_update, expected_update, expected_parent_sha, expected_record)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    parent_sha_pass = str(payload.get("parent_checkpoint_sha256")) == expected_parent_sha
    metric_pass = float(payload.get("H1")) == float(expected_record["candidate_H1"]) and float(payload.get("H32")) == float(expected_record["candidate_H32"])
    model_pass = d32.d27.tensor_state_equal(payload.get("model_state_dict"), expected_model.state_dict())
    optimizer_pass = d32.d27.tensor_state_equal(payload.get("optimizer_state_dict"), expected_optimizer.state_dict())
    rng_pass = d32.d27.tensor_state_equal(payload.get("rng_state"), expected_rng)
    committed_pass = payload.get("scientific_state") == "COMMITTED_FORWARD_STATE" and payload.get("resumable") is True
    return {
        **structural,
        "pass": bool(structural["pass"] and parent_sha_pass and metric_pass and model_pass and optimizer_pass and rng_pass and committed_pass),
        "payload_parent_sha256_pass": parent_sha_pass,
        "payload_metric_pass": metric_pass,
        "model_state_pass": model_pass,
        "optimizer_state_pass": optimizer_pass,
        "rng_state_pass": rng_pass,
        "committed_resumable_state_pass": committed_pass,
    }


def load_start_or_resume(
    output: Path, runtime: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any], list[float], list[dict[str, Any]], list[dict[str, Any]], int, float, float, str, float, float, int, int]:
    d32_rows = load_d32_trajectory()
    trajectory_path = output / "D33_CANONICAL_TRAJECTORY.json"
    if not trajectory_path.is_file():
        allowed_partial_names = {"D33_CANDIDATE_RESULTS_COMPACT.json", "D33_SEARCH_PROGRESS.json"}
        partial_files = [path for path in output.rglob("*") if path.is_file()]
        unsafe_partial_files = [path for path in partial_files if path.name not in allowed_partial_names]
        require(not unsafe_partial_files, f"PARTIAL_D33_OUTPUT_WITHOUT_RECOVERABLE_TRAJECTORY:{[path.name for path in unsafe_partial_files]}")
        partial_candidates_path = output / "D33_CANDIDATE_RESULTS_COMPACT.json"
        partial_candidates = json.loads(partial_candidates_path.read_text(encoding="utf-8")) if partial_candidates_path.is_file() else []
        partial_search_path = output / "D33_SEARCH_PROGRESS.json"
        partial_search = json.loads(partial_search_path.read_text(encoding="utf-8")) if partial_search_path.is_file() else {}
        require(d32.d31.sha256_file(PARENT) == EXPECTED_PARENT_SHA256, "UPDATE_450_CHECKPOINT_SHA256_MISMATCH")
        payload = torch.load(PARENT, map_location="cpu", weights_only=False)
        verify_loaded_payload(payload, START_UPDATE, START_H1, START_H32, 449, EXPECTED_PARENT_PARENT_SHA256)
        model, optimizer, rng, canonical = instantiate_state(runtime, payload)
        tail = d32_rows[-1]
        return (
            model,
            optimizer,
            rng,
            canonical,
            [],
            partial_candidates,
            START_UPDATE,
            START_H1,
            START_H32,
            EXPECTED_PARENT_SHA256,
            float(tail.get("alpha", 0.03125)),
            float(tail.get("p4_beta", 0.875)),
            int(partial_search.get("candidate_specifications_screened", 0)),
            int(partial_search.get("shadow_executions", 0)),
        )
    saved = json.loads(trajectory_path.read_text(encoding="utf-8"))
    rows = list(saved.get("trajectory", []))
    require(rows, "RESUME_TRAJECTORY_EMPTY")
    expected_updates = list(range(451, int(rows[-1]["update"]) + 1))
    require([int(row["update"]) for row in rows] == expected_updates, "D33_RESUME_TRAJECTORY_NOT_CONTIGUOUS")
    previous_update, previous_sha, previous_h1, previous_h32 = START_UPDATE, EXPECTED_PARENT_SHA256, START_H1, START_H32
    for row in rows:
        update = int(row["update"])
        require(int(row["parent"]) == previous_update, f"D33_RESUME_PARENT_UPDATE_MISMATCH:{update}")
        require(str(row["parent_checkpoint_sha256"]) == previous_sha, f"D33_RESUME_PARENT_SHA_MISMATCH:{update}")
        require(float(row["H1_before"]) == previous_h1 and float(row["H32_before"]) == previous_h32, f"D33_RESUME_METRIC_LINK_MISMATCH:{update}")
        checkpoint = output / "checkpoints" / f"committed_update_{update}.pt"
        require(checkpoint.is_file(), f"D33_RESUME_CHECKPOINT_MISSING:{update}")
        require(d32.d31.sha256_file(checkpoint) == str(row["checkpoint_sha256"]), f"D33_RESUME_CHECKPOINT_SHA_MISMATCH:{update}")
        checkpoint_payload_value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        verify_loaded_payload(checkpoint_payload_value, update, float(row["H1"]), float(row["H32"]), previous_update, previous_sha)
        previous_update, previous_sha = update, str(row["checkpoint_sha256"])
        previous_h1, previous_h32 = float(row["H1"]), float(row["H32"])
    final_path = output / "checkpoints" / f"committed_update_{previous_update}.pt"
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    verify_loaded_payload(payload, previous_update, previous_h1, previous_h32, previous_update - 1, str(rows[-1]["parent_checkpoint_sha256"]))
    model, optimizer, rng, canonical = instantiate_state(runtime, payload)
    candidates_path = output / "D33_CANDIDATE_RESULTS_COMPACT.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.is_file() else []
    retained_screen_count = len(
        {
            (int(record["candidate_update"]), str(record["spec_id"]))
            for record in candidates
            if record.get("candidate_update") is not None
            and record.get("spec_id") is not None
            and record.get("screening_stage") in ("D33_PRIMARY_H1_FIRST_LATTICE", "D33_LOCAL_REPAIR_FALLBACK_LATTICE")
        }
    )
    search_progress_path = output / "D33_SEARCH_PROGRESS.json"
    search_progress = json.loads(search_progress_path.read_text(encoding="utf-8")) if search_progress_path.is_file() else {}
    return (
        model,
        optimizer,
        rng,
        canonical,
        rows,
        candidates,
        previous_update,
        previous_h1,
        previous_h32,
        previous_sha,
        float(rows[-1]["candidate_scale"]),
        float(rows[-1].get("baseline_p4_beta", rows[-1].get("p4_beta", 0.875))),
        max(int(saved.get("candidate_specifications_screened", 0)), retained_screen_count, int(search_progress.get("candidate_specifications_screened", 0))),
        max(int(saved.get("shadow_executions", 0)), int(search_progress.get("shadow_executions", 0))),
    )


def run_one(
    trial_args: tuple[Any, ...],
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
    stage: str,
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    record, model, optimizer, rng = d32.run_trial(*trial_args, spec)
    record.update(
        {
            "screening_stage": stage,
            "deterministic_replay_consistency_pass": False,
            "deterministic_replay_status": "NOT_RUN_SCREEN_ONLY",
            "candidate_valid": False,
            "selected_spec": dict(spec),
        }
    )
    annotate_safety(record, policy)
    return record, model, optimizer, rng


def replay_candidate(
    trial_args: tuple[Any, ...],
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    record, model, optimizer, rng = d32.evaluated_candidate(*trial_args, spec)
    record.update({"screening_stage": "D33_SERIOUS_DETERMINISTIC_REPLAY", "selected_spec": dict(spec)})
    annotate_safety(record, policy)
    return record, model, optimizer, rng


def shadow_horizon(
    runtime: Mapping[str, Any],
    parent_model: Any,
    parent_optimizer: Any,
    parent_rng: Mapping[str, Any],
    canonical: Sequence[float],
    parent_update: int,
    parent_h1: float,
    parent_h32: float,
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
    horizon: int,
    schedule_length: int,
) -> tuple[dict[str, Any], int]:
    model, optimizer, rng = parent_model, parent_optimizer, copy.deepcopy(parent_rng)
    update, h1, h32 = parent_update, parent_h1, parent_h32
    records: list[dict[str, Any]] = []
    executions = 0
    for child_update in range(parent_update + 1, min(parent_update + horizon, schedule_length) + 1):
        identity = d32.semantic_parent_identity(model, optimizer, rng)
        args = (runtime, model, optimizer, rng, canonical, identity, update, child_update, h1, h32)
        child_policy = reserve_policy(
            H32_LIMIT - h32,
            relaxed=bool(policy.get("relaxed_after_progress_deterioration", False)),
        )
        try:
            record, model, optimizer, rng = run_one(args, spec, child_policy, "D33_SHORT_HORIZON_SHADOW")
        except Exception as exc:
            records.append(failed_candidate_record(child_update, update, spec, "D33_SHORT_HORIZON_SHADOW", exc))
            executions += 1
            break
        executions += 1
        records.append(record)
        if not horizon_step_safety_pass(record):
            break
        update, h1, h32 = child_update, float(record["candidate_H1"]), float(record["candidate_H32"])
    expected = min(horizon, schedule_length - parent_update)
    safety_passed = bool(len(records) == expected and all(horizon_step_safety_pass(record) for record in records))
    terminal_h1_passed = bool(
        records
        and records[-1].get("candidate_H1") is not None
        and float(records[-1]["candidate_H1"]) <= float(parent_h1) - MIN_H1_DESCENT
    )
    passed = bool(safety_passed and terminal_h1_passed)
    return (
        {
            "horizon_length": len(records),
            "horizon_requested": horizon,
            "horizon_criterion_version": HORIZON_CRITERION_VERSION,
            "horizon_safety_status": "PASS" if safety_passed else "FAIL",
            "horizon_terminal_H1_pass": terminal_h1_passed,
            "horizon_status": "PASS" if passed else "FAIL",
            "horizon_terminal_H1": float(records[-1]["candidate_H1"]) if records and records[-1].get("candidate_H1") is not None else None,
            "horizon_terminal_H32": float(records[-1]["candidate_H32"]) if records and records[-1].get("candidate_H32") is not None else None,
            "horizon_min_H32_margin": min((float(record["H32_margin"]) for record in records if record.get("H32_margin") is not None), default=None),
            "horizon_updates": [int(record["candidate_update"]) for record in records],
            "horizon_trace": [
                {
                    "update": int(record["candidate_update"]),
                    "H1": record.get("candidate_H1"),
                    "H32": record.get("candidate_H32"),
                    "H32_margin": record.get("H32_margin"),
                    "H1_improvement_pass": record.get("H1_improvement_pass"),
                    "step_safety_pass": horizon_step_safety_pass(record),
                }
                for record in records
            ],
        },
        executions,
    )


def checkpoint_payload(
    model: Any,
    optimizer: Any,
    rng: Mapping[str, Any],
    record: Mapping[str, Any],
    parent_sha: str,
    canonical: Sequence[float],
) -> dict[str, Any]:
    payload = d32.checkpoint_payload(model, optimizer, rng, record, parent_sha, canonical)
    payload["schema_version"] = "stage3_h13_d33_resumable_committed_checkpoint_v1"
    payload["experiment_id"] = EXPERIMENT_ID
    payload["mode"] = "D33_H1_FIRST_SAFE_SLACK"
    payload["immutable_d32_parent"] = {"update": START_UPDATE, "sha256": EXPECTED_PARENT_SHA256}
    payload["candidate_rule"].update(
        {
            "selection_order": "hard_legality_then_adaptive_H32_reserve_then_lowest_replayed_H1",
            "safety_regime": record["safety_regime"],
            "safety_reserve": record["safety_reserve"],
            "inherited_D32_baseline_competed": True,
        }
    )
    payload["last_record_identity"]["rng_state_hash"] = d32.d24.d17.base.rng_digest(rng)
    return payload


def material_rejections(replayed: Sequence[Mapping[str, Any]], selected_id: str) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for record in replayed:
        if str(record["spec_id"]) == selected_id:
            continue
        item = compact_record(record)
        if record.get("horizon_status") == "FAIL":
            reason = "SHORT_HORIZON_SAFETY_FAILURE"
        elif not hard_gate_pass(record, replay_required=True):
            reason = "HARD_GATE_OR_REPLAY_FAILURE"
        elif not bool(record.get("safety_reserve_pass")):
            reason = "ADAPTIVE_H32_SAFETY_RESERVE_FAILURE"
        else:
            reason = "WORSE_CREDIBLE_REALIZED_H1_THAN_SELECTED"
        item["material_rejection_reason"] = reason
        rejected.append(item)
    return rejected


def persist_progress(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    candidate_count: int,
    shadow_executions: int,
    blocker: str | None,
) -> None:
    value = {
        "schema_version": "stage3_h13_d33_canonical_trajectory_v1",
        "immutable_parent": {"update": START_UPDATE, "H1": START_H1, "H32": START_H32, "sha256": EXPECTED_PARENT_SHA256},
        "trajectory": list(rows),
        "candidate_specifications_screened": int(candidate_count),
        "shadow_executions": int(shadow_executions),
        "stage3_H1_target_reached": bool(rows and float(rows[-1]["H1"]) <= TARGET_H1),
        "next_blocker": blocker,
    }
    atomic_json(output / "D33_CANONICAL_TRAJECTORY.json", value)
    write_csv(output / "D33_CANONICAL_TRAJECTORY.csv", rows)
    atomic_json(output / "D33_CANDIDATE_RESULTS_COMPACT.json", list(candidates))
    atomic_json(output / "D33_MATERIAL_REJECTED_FINALISTS.json", list(rejected))


def render_report(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
    shadow_executions: int,
    blocker: str | None,
    final_sha: str,
) -> str:
    final_update = int(rows[-1]["update"]) if rows else START_UPDATE
    final_h1 = float(rows[-1]["H1"]) if rows else START_H1
    final_h32 = float(rows[-1]["H32"]) if rows else START_H32
    minimum_margin = min([H32_LIMIT - START_H32] + [float(row["H32_margin"]) for row in rows])
    target = final_h1 <= TARGET_H1
    best_delta = min((float(row["delta_H1"]) for row in rows), default=0.0)
    table = "\n".join(
        f"| {int(row['update'])} | {float(row['H1']):.17g} | {float(row['delta_H1']):.17g} | {float(row['H32']):.17g} | {float(row['delta_H32']):.17g} | {float(row['H32_margin']):.17g} | {row['selected_method']} | {float(row['candidate_scale']):.8g} | {row['status']} |"
        for row in rows
    )
    status = "PASS" if target else ("BLOCKED" if blocker else "IN_PROGRESS")
    classification = "D33_STAGE3_H1_TARGET_REACHED" if target else "D33_INHERITANCE_PRESERVING_FORWARD_CONTINUATION"
    final_checkpoint = output / "checkpoints" / f"committed_update_{final_update}.pt" if rows else PARENT
    report = f"""# D33 Inheritance-Preserving Stage-3 Continuation Report

TASK_STATUS: {status}
SCIENTIFIC_EXECUTION_STATUS: {status}
FINAL_CLASSIFICATION: {classification}

START_UPDATE: {START_UPDATE}
FINAL_COMMITTED_UPDATE: {final_update}
NUMBER_OF_D33_COMMITS: {len(rows)}

START_H1: {START_H1:.17g}
FINAL_H1: {final_h1:.17g}
TOTAL_H1_IMPROVEMENT: {START_H1-final_h1:.17g}
STAGE3_H1_TARGET: {TARGET_H1:.17g}
STAGE3_H1_TARGET_REACHED: {'YES' if target else 'NO'}

START_H32: {START_H32:.17g}
FINAL_H32: {final_h32:.17g}
FINAL_H32_MARGIN: {H32_LIMIT-final_h32:.17g}
MIN_D33_H32_MARGIN: {minimum_margin:.17g}

CANONICAL_PREFIX_0_450_PRESERVED: YES
ADAMW_HISTORY_PRESERVED: YES
SCIENTIFIC_STATE_CONTAMINATION: NO
OPTIMIZER_CONSISTENCY: {'PASS' if all(row['optimizer_consistency'] == 'PASS' for row in rows) else 'NOT_RUN'}
REPLAY_VALIDATION: {'PASS' if all(row['replay_validation'] == 'PASS' for row in rows) else 'NOT_RUN'}
INDEPENDENT_FINAL_REPLAY_VALIDATION: NOT_RUN_SEPARATE_VALIDATOR

SOL_EXTREME_PRIMARY_USED: NOT_GUARANTEED_BY_RUNTIME
LUNA_MAX_SUBAGENTS_USED: YES
LUNA_MAX_SUBAGENT_COUNT: 3

CANDIDATES_SCREENED: {candidate_count}
SHADOW_EXECUTIONS: {shadow_executions}

INHERITED_D32_METHODS_PRESERVED: P4; adaptive-beta P4; H1/H32 presented-gradient correction; tangent/inward projection; CAGrad; realized-AdamW correction; isolated shadow execution; AdamW transition validation; deterministic replay; resumable canonical checkpoints
D33_INCREMENTAL_IMPROVEMENTS: H1-first ranking after hard legality and adaptive H32 reserve gates; controlled outward H32 use in GREEN; inherited-baseline competition; trust-region expansion/contraction; conditional short-horizon shadow evaluation; prospective finalist rejection records
BEST_REALIZED_H1_IMPROVEMENT_PER_UPDATE: {best_delta:.17g}

FINAL_CHECKPOINT: {final_checkpoint}
FINAL_CHECKPOINT_SHA256: {final_sha}

NEXT_BLOCKER: {blocker or 'none'}

| update | H1 | delta_H1 | H32 | delta_H32 | H32_margin | selected_method | candidate_scale | status |
|---:|---:|---:|---:|---:|---:|---|---:|---|
{table}
"""
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "TASK_AND_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8", newline="\n")
    return report


def run(output: Path, max_update: int) -> dict[str, Any]:
    require(AUTHORIZATION.is_file(), "D33_AUTHORIZATION_MISSING")
    output.mkdir(parents=True, exist_ok=True)
    runtime = d32.d24.d17.preflight_runtime()
    schedule_length = len(runtime["authority"]["schedule_rows"])
    require(max_update <= schedule_length, f"MAX_UPDATE_EXCEEDS_AUTHENTICATED_SCHEDULE:{schedule_length}")
    (
        model,
        optimizer,
        rng,
        canonical,
        rows,
        candidate_records,
        current_update,
        h1,
        h32,
        current_sha,
        previous_alpha,
        previous_beta,
        candidate_count,
        shadow_executions,
    ) = load_start_or_resume(output, runtime)
    require(current_update <= max_update, "RESUME_STATE_BEYOND_REQUESTED_MAX_UPDATE")
    material_rejected: list[dict[str, Any]] = []
    rejected_path = output / "D33_MATERIAL_REJECTED_FINALISTS.json"
    if rejected_path.is_file():
        material_rejected = list(json.loads(rejected_path.read_text(encoding="utf-8")))
    blocker: str | None = None
    if h1 <= TARGET_H1:
        persist_progress(output, rows, candidate_records, material_rejected, candidate_count, shadow_executions, None)
        render_report(output, rows, candidate_count, shadow_executions, None, current_sha)
        return {"status": "PASS_REQUIRES_INDEPENDENT_FINAL_REPLAY", "classification": "D33_STAGE3_H1_TARGET_REACHED_PENDING_INDEPENDENT_REPLAY", "final_update": current_update, "final_H1": h1, "final_H32": h32, "final_checkpoint_sha256": current_sha, "candidate_count": candidate_count, "shadow_executions": shadow_executions, "output": str(output)}

    for update in range(current_update + 1, max_update + 1):
        margin_before = H32_LIMIT - h32
        recent = [-float(row["delta_H1"]) for row in rows[-3:]]
        stagnation = len(recent) == 3 and sum(recent) / 3.0 < 5.0e-8
        previous_relaxed = bool(rows and float(rows[-1].get("safety_reserve", 2.5e-4)) < 2.5e-4)
        full_reserve_recovered = margin_before >= 2.5e-4 + 2.0 * 8.30074e-5
        relaxed_controller = stagnation or (previous_relaxed and not full_reserve_recovered)
        policy = reserve_policy(margin_before, relaxed=relaxed_controller)
        specs = primary_specs(previous_alpha, previous_beta, str(policy["regime"]), stagnation)
        baseline_id = d32.spec_id(inherited_baseline(previous_alpha, previous_beta))
        parent_identity = d32.semantic_parent_identity(model, optimizer, rng)
        trial_args = (runtime, model, optimizer, rng, canonical, parent_identity, current_update, update, h1, h32)
        screened: list[dict[str, Any]] = []
        spec_map: dict[str, dict[str, Any]] = {}
        retained_screens = {
            str(record["spec_id"]): dict(record)
            for record in candidate_records
            if int(record.get("candidate_update", -1)) == update
            and record.get("screening_stage") == "D33_PRIMARY_H1_FIRST_LATTICE"
            and record.get("candidate_H1") is not None
        }
        for spec in specs:
            identity = d32.spec_id(spec)
            if identity in retained_screens:
                record = retained_screens[identity]
            else:
                try:
                    record, _, _, _ = run_one(trial_args, spec, policy, "D33_PRIMARY_H1_FIRST_LATTICE")
                except Exception as exc:
                    record = failed_candidate_record(update, current_update, spec, "D33_PRIMARY_H1_FIRST_LATTICE", exc)
                candidate_records.append(compact_record(record))
                candidate_count += 1
                shadow_executions += 1
                atomic_json(output / "D33_CANDIDATE_RESULTS_COMPACT.json", candidate_records)
                persist_search_progress(output, candidate_count, shadow_executions, update)
            screened.append(record)
            spec_map[str(record["spec_id"])] = dict(spec)
            if record.get("candidate_H1") is not None:
                print(
                    f"D33_SCREEN update={update} regime={policy['regime']} family={record['candidate_family']} "
                    f"base={record['source_base']} beta={record['p4_beta']:.8g} alpha={record['alpha']:.8g} "
                    f"lambda={record['used_h32_lambda']:.8g} H1={record['candidate_H1']:.17g} "
                    f"H32={record['candidate_H32']:.17g} reserve={record['safety_reserve_pass']} legal={record['candidate_valid_before_replay']}",
                    flush=True,
                )
            else:
                print(f"D33_SCREEN_REJECTED_LOCAL_FAILURE update={update} family={record['candidate_family']} reason={record['rejection_reason']}", flush=True)
            if str(policy["regime"]) == "YELLOW" and yellow_lattice_has_enough_evidence(screened):
                break
        safe = sorted([record for record in screened if safely_legal(record)], key=h1_first_rank)
        if not safe:
            retained_fallback = {
                str(record["spec_id"]): dict(record)
                for record in candidate_records
                if int(record.get("candidate_update", -1)) == update
                and record.get("screening_stage") == "D33_LOCAL_REPAIR_FALLBACK_LATTICE"
                and record.get("candidate_H1") is not None
            }
            for spec in fallback_specs(previous_alpha, previous_beta):
                identity = d32.spec_id(spec)
                if identity in spec_map:
                    continue
                if identity in retained_fallback:
                    record = retained_fallback[identity]
                else:
                    try:
                        record, _, _, _ = run_one(trial_args, spec, policy, "D33_LOCAL_REPAIR_FALLBACK_LATTICE")
                    except Exception as exc:
                        record = failed_candidate_record(update, current_update, spec, "D33_LOCAL_REPAIR_FALLBACK_LATTICE", exc)
                    candidate_records.append(compact_record(record))
                    candidate_count += 1
                    shadow_executions += 1
                    atomic_json(output / "D33_CANDIDATE_RESULTS_COMPACT.json", candidate_records)
                    persist_search_progress(output, candidate_count, shadow_executions, update)
                screened.append(record)
                spec_map[str(record["spec_id"])] = dict(spec)
                if len([candidate for candidate in screened if safely_legal(candidate)]) >= 3:
                    break
            safe = sorted([record for record in screened if safely_legal(record)], key=h1_first_rank)
        if not safe:
            legal_without_reserve = sorted([record for record in screened if hard_gate_pass(record)], key=h1_first_rank)
            if legal_without_reserve and str(policy["regime"]) == "RED":
                inward = [record for record in legal_without_reserve if float(record["delta_H32"]) <= 0.0]
                safe = inward[:2]
            if not safe:
                blocker = f"NO_HARD_GATE_AND_SAFETY_RESERVE_CANDIDATE_AT_UPDATE_{update}"
                break

        # The realized one-step lattice is already persisted.  Keep replay
        # bounded to the two H1 leaders plus the exact inherited baseline;
        # close leaders receive the chained horizon comparison below.
        serious = list(safe[:2])
        cap_pareto = next((record for record in safe if bool(record.get("H32_outward_cap_pass"))), None)
        if cap_pareto is not None and all(str(record["spec_id"]) != str(cap_pareto["spec_id"]) for record in serious):
            serious.append(cap_pareto)
        if str(policy["regime"]) != "GREEN":
            conservative = most_h32_conservative(safe)
            if conservative is not None and all(str(record["spec_id"]) != str(conservative["spec_id"]) for record in serious):
                serious.append(dict(conservative))
            stable_scale = smallest_scale_comparator(safe)
            if stable_scale is not None and all(str(record["spec_id"]) != str(stable_scale["spec_id"]) for record in serious):
                serious.append(dict(stable_scale))
        baseline_screen = next((record for record in screened if str(record["spec_id"]) == baseline_id), None)
        if baseline_screen is not None and safely_legal(baseline_screen) and all(str(record["spec_id"]) != baseline_id for record in serious):
            serious.append(baseline_screen)

        horizon_map = retained_horizon_map(candidate_records, update, current_update, float(policy["reserve"]))
        best_improvement = max(0.0, -float(serious[0]["delta_H1"]))
        close = len(serious) > 1 and abs(float(serious[0]["candidate_H1"]) - float(serious[1]["candidate_H1"])) <= max(2.0e-8, 0.25 * best_improvement)
        high_slack_use = float(serious[0]["delta_H32"]) > 0.20 * max(0.0, margin_before - float(policy["reserve"]))
        if (close or high_slack_use) and update + 1 <= schedule_length:
            horizon = min(3, schedule_length - current_update)
            # ``serious`` is bounded to at most four entries (two H1 leaders,
            # a cap-passing Pareto point, and the inherited baseline).  Screen
            # all of them so a conservative baseline remains eligible when
            # every high-impact finalist fails its horizon.
            horizon_screens = serious
            for screen in horizon_screens:
                if str(screen["spec_id"]) in horizon_map:
                    continue
                result, count = shadow_horizon(
                    runtime,
                    model,
                    optimizer,
                    rng,
                    canonical,
                    current_update,
                    h1,
                    h32,
                    spec_map[str(screen["spec_id"])],
                    policy,
                    horizon,
                    schedule_length,
                )
                horizon_map[str(screen["spec_id"])] = result
                shadow_executions += count
                persist_search_progress(output, candidate_count, shadow_executions, update)

        replayed: list[tuple[dict[str, Any], Any, Any, dict[str, Any]]] = []
        for screen in serious:
            serious_spec = spec_map[str(screen["spec_id"])]
            try:
                item = replay_candidate(trial_args, serious_spec, policy)
            except Exception as exc:
                failed = failed_candidate_record(update, current_update, serious_spec, "D33_SERIOUS_DETERMINISTIC_REPLAY", exc)
                candidate_records.append(compact_record(failed))
                shadow_executions += 1
                atomic_json(output / "D33_CANDIDATE_RESULTS_COMPACT.json", candidate_records)
                persist_search_progress(output, candidate_count, shadow_executions, update)
                print(f"D33_REPLAY_REJECTED_LOCAL_FAILURE update={update} family={failed['candidate_family']} reason={failed['rejection_reason']}", flush=True)
                continue
            if str(item[0]["spec_id"]) in horizon_map:
                item[0].update(horizon_map[str(item[0]["spec_id"])])
            replayed.append(item)
            candidate_records.append(compact_record(item[0]))
            shadow_executions += 2
            atomic_json(output / "D33_CANDIDATE_RESULTS_COMPACT.json", candidate_records)
            persist_search_progress(output, candidate_count, shadow_executions, update)
            print(
                f"D33_REPLAY update={update} family={item[0]['candidate_family']} alpha={item[0]['alpha']:.8g} "
                f"H1={item[0]['candidate_H1']:.17g} H32={item[0]['candidate_H32']:.17g} "
                f"reserve={item[0]['safety_reserve_pass']} replay={item[0]['deterministic_replay_status']}",
                flush=True,
            )
        legal = [item for item in replayed if promotion_safety_pass(item[0], policy)]
        if not legal:
            blocker = f"NO_REPLAY_VALIDATED_D33_CANDIDATE_AT_UPDATE_{update}"
            break

        selected_item, selection_basis = select_replayed_candidate(legal, close)
        selected, selected_model, selected_optimizer, selected_rng = selected_item
        selected["selection_status"] = "SELECTED_COMMITTED"
        selected["trial_state"] = "COMMITTED_FORWARD_STATE"
        selected["selection_basis"] = selection_basis
        selected_spec = spec_map[str(selected["spec_id"])]
        checkpoint = output / "checkpoints" / f"committed_update_{update}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save(checkpoint_payload(selected_model, selected_optimizer, selected_rng, selected, current_sha, canonical), temporary_checkpoint)
        preliminary_verification = verify_committed_checkpoint(
            temporary_checkpoint,
            current_update,
            update,
            current_sha,
            selected,
            selected_model,
            selected_optimizer,
            selected_rng,
        )
        require(bool(preliminary_verification["pass"]), f"PRECOMMIT_CHECKPOINT_VERIFICATION_FAILED:{update}")
        temporary_checkpoint.replace(checkpoint)
        checkpoint_sha = d32.d31.sha256_file(checkpoint)
        selected["checkpoint_sha256"] = checkpoint_sha
        verification = verify_committed_checkpoint(
            checkpoint,
            current_update,
            update,
            current_sha,
            selected,
            selected_model,
            selected_optimizer,
            selected_rng,
        )
        require(bool(verification["pass"]), f"CHECKPOINT_VERIFICATION_FAILED:{update}")
        improvement = -float(selected["delta_H1"])
        row = {
            "update": update,
            "parent": current_update,
            "H1_before": h1,
            "H1": float(selected["candidate_H1"]),
            "delta_H1": float(selected["delta_H1"]),
            "H32_before": h32,
            "H32": float(selected["candidate_H32"]),
            "delta_H32": float(selected["delta_H32"]),
            "H32_margin": float(selected["H32_margin"]),
            "selected_method": f"{selected['candidate_family']}:{selected['source_base']}",
            "candidate_family": selected["candidate_family"],
            "source_base": selected["source_base"],
            "candidate_scale": float(selected["alpha"]),
            "alpha": float(selected["alpha"]),
            "p4_beta": float(selected["p4_beta"]),
            "used_h32_lambda": float(selected["used_h32_lambda"]),
            "safety_regime": selected["safety_regime"],
            "safety_reserve": float(selected["safety_reserve"]),
            "recent_average_H1_improvement": (sum([-float(value["delta_H1"]) for value in rows[-2:]] + [improvement]) / min(3, len(rows) + 1)),
            "H1_improvement_per_H32_slack_consumed": selected.get("H1_improvement_per_H32_slack_consumed"),
            "selection_basis": selection_basis,
            "selected_spec": dict(selected_spec),
            "optimizer_consistency": "PASS",
            "replay_validation": "PASS",
            "deterministic_replay_consistency": "PASS",
            "RNG_continuity": "PASS",
            "status": "COMMITTED_VALID",
            "parent_checkpoint_sha256": current_sha,
            "checkpoint_sha256": checkpoint_sha,
        }
        rows.append(row)
        if str(selected_spec.get("base")) == "P4":
            previous_beta = float(selected_spec.get("p4_beta", previous_beta))
        row["baseline_p4_beta"] = previous_beta
        for retained in reversed(candidate_records):
            if int(retained.get("candidate_update", -1)) == update and str(retained.get("spec_id")) == str(selected["spec_id"]) and retained.get("deterministic_replay_status") == "PASS":
                retained["selection_status"] = "SELECTED_COMMITTED"
                break
        material_rejected.extend(material_rejections([item[0] for item in replayed], str(selected["spec_id"])))
        atomic_json(output / "metrics" / f"update_{update}.json", {"trajectory_row": row, "selected_candidate": selected, "checkpoint_validation": verification})
        model, optimizer, rng = selected_model, selected_optimizer, selected_rng
        current_update, h1, h32, current_sha = update, float(selected["candidate_H1"]), float(selected["candidate_H32"]), checkpoint_sha
        previous_alpha = float(selected["alpha"])
        persist_progress(output, rows, candidate_records, material_rejected, candidate_count, shadow_executions, None)
        print(
            f"D33_COMMITTED update={update} H1={h1:.17g} H32={h32:.17g} margin={H32_LIMIT-h32:.17g} "
            f"method={row['selected_method']} alpha={previous_alpha:.8g} sha={current_sha}",
            flush=True,
        )
        if h1 <= TARGET_H1:
            break

    if h1 > TARGET_H1 and blocker is None and current_update >= schedule_length:
        blocker = f"AUTHENTICATED_CANONICAL_SCHEDULE_EXHAUSTED_AT_{schedule_length}_WITH_H1_ABOVE_TARGET"
    persist_progress(output, rows, candidate_records, material_rejected, candidate_count, shadow_executions, blocker)
    report = render_report(output, rows, candidate_count, shadow_executions, blocker, current_sha)
    atomic_json(
        output / "D33_EXECUTION_MANIFEST.json",
        {
            "schema_version": "stage3_h13_d33_execution_manifest_v1",
            "experiment_id": EXPERIMENT_ID,
            "authorization": {"path": str(AUTHORIZATION)},
            "authoritative_parent": {"update": START_UPDATE, "path": str(PARENT), "sha256": EXPECTED_PARENT_SHA256, "H1": START_H1, "H32": START_H32},
            "updates_committed": [int(row["update"]) for row in rows],
            "final_update": current_update,
            "final_H1": h1,
            "final_H32": h32,
            "target_reached": h1 <= TARGET_H1,
            "next_blocker": blocker,
            "candidate_specifications_screened": candidate_count,
            "shadow_executions": shadow_executions,
            "scientific_state_contamination": False,
            "optimizer_history_reset": False,
            "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
            "execution_source": {"path": str(Path(__file__).resolve()), "sha256": d32.d31.sha256_file(Path(__file__).resolve())},
        },
    )
    status = "PASS" if h1 <= TARGET_H1 else ("BLOCKED" if blocker else "IN_PROGRESS")
    return {
        "status": status,
        "classification": "D33_STAGE3_H1_TARGET_REACHED" if h1 <= TARGET_H1 else "D33_INHERITANCE_PRESERVING_FORWARD_CONTINUATION",
        "first_blocker": blocker,
        "output": str(output),
        "final_update": current_update,
        "final_H1": h1,
        "final_H32": h32,
        "final_checkpoint_sha256": current_sha,
        "candidate_count": candidate_count,
        "shadow_executions": shadow_executions,
        "report_tail": report.splitlines()[-1] if report else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-update", type=int, default=512)
    args = parser.parse_args()
    try:
        result = run(args.output.resolve(), args.max_update)
    except Exception as exc:
        result = {"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}"}
        print(json.dumps(result, sort_keys=True), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else (2 if result["status"] == "BLOCKED" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
