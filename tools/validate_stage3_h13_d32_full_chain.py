"""Fresh-replay validator for a D32 chain split across continuation bundles."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d31_d32_active_boundary_tailor_continuation as d32


PREFIX_EXPECTED = {
    423: "9d5c51872209baafe69f01b739ac1608b6adebd466ea8d7bae123bc56ba5fccc",
    424: "7fa4aed180e90b432b2b734c5a77e1bd532e0910fd1c91729d81d03b81b706bc",
    425: "569e256834a4e6db68755511139d27900a7cde74228af4bbc5de3ba95e9db298",
    426: d32.EXPECTED_PARENT_SHA256,
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def prefix_paths() -> dict[int, Path]:
    return {
        423: d32.d31.D30_ROOT / "checkpoints" / "committed_update_423.pt",
        424: d32.D31_ROOT / "checkpoints" / "committed_update_424.pt",
        425: d32.D31_ROOT / "checkpoints" / "committed_update_425.pt",
        426: d32.PARENT,
    }


def load_branch(runtime: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    model, optimizer = d32.d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, optimizer, copy.deepcopy(payload["rng_state"])


def spec_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    rule = payload["candidate_rule"]
    return {
        "family": rule["family"],
        "base": rule["source_base"],
        "lambda": float(rule["h32_lambda"]),
        "rho": float(rule["rho"]),
        "kappa": float(rule["kappa"]),
        "p4_beta": float(rule.get("p4_beta", 0.35)),
        "cagrad_c": float(rule.get("cagrad_c", 0.4)),
        "alpha": float(rule["alpha"]),
        "target_norm": float(rule["target_presented_norm"]),
    }


def checkpoint_map(output_roots: Sequence[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for output_root in output_roots:
        for path in (output_root / "checkpoints").glob("committed_update_*.pt"):
            update = int(path.stem.rsplit("_", 1)[-1])
            require(update not in result, f"DUPLICATE_COMMITTED_UPDATE:{update}")
            result[update] = path.resolve()
    require(result, "NO_D32_CHECKPOINTS_FOUND")
    updates = sorted(result)
    require(updates[0] == 427, f"CHAIN_MUST_BEGIN_AT_427:{updates[0]}")
    require(updates == list(range(427, updates[-1] + 1)), f"NONCONTIGUOUS_CHAIN:{updates}")
    return result


def run(output_roots: Sequence[Path], report_root: Path, expected_final_update: int) -> dict[str, Any]:
    checkpoints = checkpoint_map(output_roots)
    require(max(checkpoints) == expected_final_update, f"FINAL_UPDATE_MISMATCH:{max(checkpoints)}!={expected_final_update}")
    prefix = []
    for update, path in prefix_paths().items():
        actual = d32.d31.sha256_file(path)
        passed = actual == PREFIX_EXPECTED[update]
        prefix.append({"update": update, "path": str(path), "expected_sha256": PREFIX_EXPECTED[update], "actual_sha256": actual, "pass": passed})
        require(passed, f"PREFIX_HASH_MISMATCH:{update}")

    runtime = d32.d24.d17.preflight_runtime()
    parent_path = d32.PARENT
    parent_sha = d32.EXPECTED_PARENT_SHA256
    transitions = []
    for update, child_path in sorted(checkpoints.items()):
        require(d32.d31.sha256_file(parent_path) == parent_sha, f"PARENT_FILE_SHA256_MISMATCH:{update - 1}")
        child_sha = d32.d31.sha256_file(child_path)
        parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        child_payload = torch.load(child_path, map_location="cpu", weights_only=False)
        parent_model, parent_optimizer, parent_rng = load_branch(runtime, parent_payload)
        child_model, child_optimizer, child_rng = load_branch(runtime, child_payload)
        parent_identity = d32.semantic_parent_identity(parent_model, parent_optimizer, parent_rng)
        replay, replay_model, replay_optimizer, replay_rng = d32.evaluated_candidate(
            runtime,
            parent_model,
            parent_optimizer,
            parent_rng,
            [float(value) for value in parent_payload["canonical_base_lr"]],
            parent_identity,
            int(parent_payload["completed_optimizer_step"]),
            update,
            float(parent_payload["H1"]),
            float(parent_payload["H32"]),
            spec_from_payload(child_payload),
        )
        child_model_hash = d32.d24.d17.canonical_model_state_sha256(child_model.state_dict())
        child_optimizer_hash = d32.d24.d17.base.optimizer_semantic_hash(child_optimizer)
        child_rng_hash = d32.d24.d17.base.rng_digest(child_rng)
        optimizer_steps = d32.d24.d16.optimizer_steps(d32.d24.d16.clone_named_optimizer_state(child_optimizer))
        replay_effective_lr = [float(value) for value in json.loads(str(replay["effective_lr"]))]
        digest_schema_compatible = "p4_beta" in child_payload["candidate_rule"] and "cagrad_c" in child_payload["candidate_rule"]
        checks = {
            "payload_parent_sha256_identity": str(child_payload["parent_checkpoint_sha256"]) == parent_sha,
            "payload_parent_update_identity": int(child_payload["parent_update"]) == update - 1,
            "completed_step_identity": int(child_payload["completed_optimizer_step"]) == update,
            "next_schedule_index_identity": int(child_payload["next_schedule_index"]) == update,
            "optimizer_step_counters_identity": all(int(value) == update for value in optimizer_steps.values()),
            "canonical_lr_continuity": [float(value) for value in child_payload["canonical_base_lr"]] == [float(value) for value in parent_payload["canonical_base_lr"]],
            "effective_lr_identity": [float(value) for value in child_payload["actual_effective_lr"]] == replay_effective_lr,
            "batch_position_completed_step_identity": int(child_payload["batch_position"]["completed_optimizer_step"]) == update,
            "batch_position_schedule_index_identity": int(child_payload["batch_position"]["next_zero_based_schedule_index"]) == update,
            "last_record_update_identity": int(child_payload["last_record_identity"]["update"]) == update,
            "batch_identity": str(child_payload["last_record_identity"]["batch_window_sha256"]) == str(replay["batch_identity"]),
            "fresh_replay_pass": bool(replay["candidate_valid"]) and bool(replay["deterministic_replay_consistency_pass"]),
            "fresh_replay_H1_identity": math.isclose(float(replay["candidate_H1"]), float(child_payload["H1"]), rel_tol=0.0, abs_tol=1.0e-15),
            "fresh_replay_H32_identity": math.isclose(float(replay["candidate_H32"]), float(child_payload["H32"]), rel_tol=0.0, abs_tol=1.0e-15),
            "fresh_replay_model_identity": d32.d24.d17.canonical_model_state_sha256(replay_model.state_dict()) == child_model_hash,
            "fresh_replay_optimizer_identity": d32.d24.d17.base.optimizer_semantic_hash(replay_optimizer) == child_optimizer_hash,
            "fresh_replay_rng_identity": d32.d24.d17.base.rng_digest(replay_rng) == child_rng_hash,
            "payload_model_identity": str(child_payload["last_record_identity"]["model_state_hash"]) == child_model_hash,
            "payload_optimizer_identity": str(child_payload["last_record_identity"]["optimizer_state_hash"]) == child_optimizer_hash,
            "payload_replay_declared_pass": bool(child_payload["replay_consistency"]["pass"]),
            "fresh_candidate_digest_identity": (not digest_schema_compatible) or str(child_payload["replay_consistency"]["candidate_record_digest"]) == str(replay["deterministic_replay_candidate_record_digest"]),
            "fresh_replay_digest_identity": (not digest_schema_compatible) or str(child_payload["replay_consistency"]["replay_record_digest"]) == str(replay["deterministic_replay_record_digest"]),
            "resumable": bool(child_payload["resumable"]),
            "scientific_state_identity": str(child_payload["scientific_state"]) == "COMMITTED_FORWARD_STATE",
            "H1_descent": float(child_payload["H1"]) < float(parent_payload["H1"]),
            "H32_legality": float(child_payload["H32"]) <= d32.H32_LIMIT,
            "finite_state": d32.d24.d16.state_finite(child_model, child_optimizer),
        }
        failed = [key for key, value in checks.items() if not bool(value)]
        digest_detail = {
            "stored_candidate": str(child_payload["replay_consistency"]["candidate_record_digest"]),
            "fresh_candidate": str(replay["deterministic_replay_candidate_record_digest"]),
            "stored_replay": str(child_payload["replay_consistency"]["replay_record_digest"]),
            "fresh_replay": str(replay["deterministic_replay_record_digest"]),
        }
        require(not failed, f"D32_TRANSITION_VALIDATION_FAILED:{update}:{failed}:{json.dumps(digest_detail, sort_keys=True)}")
        transitions.append({
            "update": update,
            "parent_checkpoint": str(parent_path),
            "checkpoint": str(child_path),
            "parent_sha256": parent_sha,
            "checkpoint_sha256": child_sha,
            "H1": float(child_payload["H1"]),
            "H32": float(child_payload["H32"]),
            "fresh_replay_candidate_digest": replay["deterministic_replay_candidate_record_digest"],
            "fresh_replay_record_digest": replay["deterministic_replay_record_digest"],
            "full_record_digest_comparison": "ENFORCED_PASS" if digest_schema_compatible else "NOT_APPLICABLE_SCHEMA_EVOLUTION_PRE_P4_CAGRAD_IDENTITY_FIELDS",
            "checks": checks,
            "pass": True,
        })
        parent_path, parent_sha = child_path, child_sha

    result = {
        "schema_version": "stage3_h13_d32_full_chain_validation_v1",
        "status": "PASS",
        "output_roots": [str(path.resolve()) for path in output_roots],
        "canonical_prefix_423_426_preserved": all(item["pass"] for item in prefix),
        "prefix_checks": prefix,
        "transitions": transitions,
        "final_update": max(checkpoints),
        "final_checkpoint_sha256": parent_sha,
        "scientific_state_contamination": False,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    d32.d31.write_json(report_root / "POST_RUN_FULL_CHAIN_VALIDATION.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", action="append", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--expected-final-update", type=int, required=True)
    args = parser.parse_args()
    try:
        result = run([path.resolve() for path in args.output_root], args.report_root.resolve(), args.expected_final_update)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 1
    print(json.dumps({"status": result["status"], "final_update": result["final_update"], "final_checkpoint_sha256": result["final_checkpoint_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
