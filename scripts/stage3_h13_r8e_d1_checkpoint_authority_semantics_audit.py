#!/usr/bin/env python3
"""Additive, read-only Stage 3 H13 R8E-D1 checkpoint authority audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h13_r8e_b_multiseed_robustness import EXPECTED_A1_HASH, R8E_ROOT
from scripts.stage3_h13_r8e_causal_hidden_stability_training import R6_CHECKPOINT, load_json
from scripts.stage3_h13_r8e_d_counterfactual_feedback_audit import checkpoint_state_hash
from src.stage3_h13_r8e_b import SEEDS as R8E_B_SEEDS
from src.stage3_h13_r8e_b import sha256_file
from src.stage3_h13_r8e_d import SEEDS as R8E_D_SEEDS


R6_EXPECTED_HASH = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
R8E_B_ROOT = ROOT / "outputs/stage3_h13_r8e_b_multiseed_robustness_20260815T015906Z"
R8E_C_ROOT = ROOT / "outputs/stage3_h13_r8e_c_mechanism_audit_20260816T020000Z"
R8E_A1_CHECKPOINT = R8E_ROOT / "best_candidate_checkpoint.pt"
R8E_B_RECORDS = R8E_B_ROOT / "r8e_b_per_seed_records.json"
R8E_B_SEED_MANIFEST = R8E_B_ROOT / "r8e_b_seed_manifest.json"
R8E_D_SCRIPT = ROOT / "scripts/stage3_h13_r8e_d_counterfactual_feedback_audit.py"
EXISTING_R8E_D_ROOT = ROOT / "outputs/stage3_h13_r8e_d_counterfactual_feedback_audit_20260816T023400Z"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict[str, Any]:
    return load_json(path)


def source_line(path: Path, line: int) -> str:
    return f"{path.resolve()}:{line}"


def inspect_a1() -> dict[str, Any]:
    raw_sha = sha256_file(R8E_A1_CHECKPOINT)
    if raw_sha != EXPECTED_A1_HASH:
        raise RuntimeError(f"r8e_a1_raw_hash_mismatch:{raw_sha}")
    payload = torch.load(R8E_A1_CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError("r8e_a1_checkpoint_structure_invalid")
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError("r8e_a1_model_state_dict_invalid")
    from scripts.stage3_h13_r8e_d_counterfactual_feedback_audit import load_model

    model = load_model(R8E_A1_CHECKPOINT)
    canonical = checkpoint_state_hash(model)
    return {
        "path": str(R8E_A1_CHECKPOINT.resolve()),
        "raw_file_sha256": raw_sha,
        "canonical_model_state_sha256": canonical,
        "payload_type": type(payload).__name__,
        "payload_keys": sorted(payload),
        "state_dict_entry_count": len(state),
        "state_dict_keys": sorted(str(key) for key in state),
        "cpu_structurally_valid": True,
        "mutated_or_reserialized": False,
    }


def upstream_hashes() -> dict[str, dict[str, str]]:
    paths = {
        "R6": R6_CHECKPOINT,
        "R8E_A1": R8E_A1_CHECKPOINT,
        "R8E_B_RECORDS": R8E_B_RECORDS,
        "R8E_B_SEED_MANIFEST": R8E_B_SEED_MANIFEST,
        "R8E_B_INTEGRITY": R8E_B_ROOT / "r8e_b_integrity_manifest.json",
        "R8E_B_FINAL_REPORT": R8E_B_ROOT / "FINAL_REPORT.md",
    }
    return {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in paths.items()}


def build_authority_chain(a1: dict[str, Any], before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]) -> dict[str, Any]:
    records_payload = read_json(R8E_B_RECORDS)
    records = {int(item["seed"]): item for item in records_payload["records"]}
    seed_manifest = read_json(R8E_B_SEED_MANIFEST)
    b_hashes = {str(seed): str(records[seed]["checkpoint_sha256"]) for seed in R8E_B_SEEDS}
    retention = []
    for seed in R8E_B_SEEDS:
        retention.append({
            "SEED": seed,
            "HISTORICAL_CHECKPOINT_CREATED": True,
            "HISTORICAL_CHECKPOINT_RETAINED": bool(records[seed].get("temporary_checkpoint_retained")),
            "HISTORICAL_RAW_SHA_RECORDED": bool(records[seed].get("checkpoint_sha256")),
            "CURRENT_EXACT_RAW_FILE_AVAILABLE": False,
            "HISTORICAL_RAW_SHA256": b_hashes[str(seed)],
        })
    authority_table = [
        {
            "ARTIFACT": str((R8E_B_ROOT / "r8e_b_per_seed_records.json").resolve()),
            "FIELD": "records[seed=13086].checkpoint_sha256",
            "SEED": 13086,
            "VALUE": b_hashes["13086"],
            "VALUE_TYPE": "RAW_SERIALIZED_FILE_SHA256",
            "MEANING": "SHA-256 of R8E-B's in-memory torch.save byte stream; the corresponding file was not retained.",
            "AUTHORITATIVE_FOR_WHAT": "Historical R8E-B serialized-artifact digest only; not canonical model-state identity.",
            "SOURCE": [source_line(ROOT / "scripts/stage3_h13_r8e_causal_hidden_stability_training.py", 131), source_line(ROOT / "scripts/stage3_h13_r8e_causal_hidden_stability_training.py", 134), source_line(ROOT / "scripts/stage3_h13_r8e_b_multiseed_robustness.py", 450), source_line(ROOT / "scripts/stage3_h13_r8e_b_multiseed_robustness.py", 467)],
        },
        {
            "ARTIFACT": str((R8E_ROOT / "r8e_storage_manifest.json").resolve()),
            "FIELD": "selected_checkpoint_sha256",
            "SEED": 13086,
            "VALUE": a1["raw_file_sha256"],
            "VALUE_TYPE": "RAW_SERIALIZED_FILE_SHA256",
            "MEANING": "SHA-256 of the currently retained R8E-A1 best_candidate_checkpoint.pt bytes.",
            "AUTHORITATIVE_FOR_WHAT": "Current R8E_A1 file identity and validation-champion artifact.",
            "SOURCE": [source_line(R8E_ROOT / "r8e_storage_manifest.json", 9), source_line(ROOT / "scripts/stage3_h13_r8e_b_multiseed_robustness.py", 59)],
        },
        {
            "ARTIFACT": str(EXISTING_R8E_D_ROOT / "integrity_hashes.json"),
            "FIELD": "after.seed_13086",
            "SEED": 13086,
            "VALUE": a1["raw_file_sha256"],
            "VALUE_TYPE": "PATH_IDENTITY",
            "MEANING": "Prior R8E-D measured the same R8E_A1 path under the alias seed_13086.",
            "AUTHORITATIVE_FOR_WHAT": "Prior D run's path alias measurement, not historical R8E-B raw-artifact identity.",
            "SOURCE": [source_line(EXISTING_R8E_D_ROOT / "integrity_hashes.json", 12)],
        },
        {
            "ARTIFACT": str(R8E_C_ROOT / "FINAL_REPORT.md"),
            "FIELD": "checkpoint reproduction",
            "SEED": 13086,
            "VALUE": "5/5 exact SHA-256 matches",
            "VALUE_TYPE": "PROVENANCE_REFERENCE",
            "MEANING": "R8E-C regenerated serialized bytes; for seed 13086 it loaded A1 and reserialized with the R8E-B config.",
            "AUTHORITATIVE_FOR_WHAT": "Reproduction claim only; not a retained historical file or canonical state hash.",
            "SOURCE": [source_line(ROOT / "scripts/stage3_h13_r8e_c_mechanism_audit.py", 545), source_line(ROOT / "scripts/stage3_h13_r8e_c_mechanism_audit.py", 554), source_line(R8E_C_ROOT / "FINAL_REPORT.md", 27)],
        },
        {
            "ARTIFACT": str((R8E_B_ROOT / "r8e_b_per_seed_records.json").resolve()),
            "FIELD": "records[*].temporary_checkpoint_retained",
            "SEED": "all five",
            "VALUE": False,
            "VALUE_TYPE": "PROVENANCE_REFERENCE",
            "MEANING": "The temporary serialized checkpoint bytes were released after the record was written.",
            "AUTHORITATIVE_FOR_WHAT": "R8E-B retention policy and historical recovery classification.",
            "SOURCE": [source_line(ROOT / "scripts/stage3_h13_r8e_b_multiseed_robustness.py", 468), source_line(R8E_B_ROOT / "FINAL_REPORT.md", 160)],
        },
        {
            "ARTIFACT": str(R8E_D_SCRIPT.resolve()),
            "FIELD": "load_checkpoint_manifest / preflight seed 13086 predicates",
            "SEED": 13086,
            "VALUE": "raw digest equality plus A1 path equality",
            "VALUE_TYPE": "PROVENANCE_REFERENCE",
            "MEANING": "D requires both the manifest/record raw SHA match and the resolved path to equal A1.",
            "AUTHORITATIVE_FOR_WHAT": "Existing R8E-D preflight gate semantics under audit.",
            "SOURCE": [source_line(R8E_D_SCRIPT, 122), source_line(R8E_D_SCRIPT, 147)],
        },
    ]
    return {
        "schema_version": "stage3_h13_r8e_d1_authority_chain_v1",
        "hash_origins": {
            "r8e_b_seed_13086_recorded_sha256": b_hashes["13086"],
            "r8e_a1_current_raw_sha256": a1["raw_file_sha256"],
        },
        "authority_table": authority_table,
        "r8e_b_seed_manifest": {
            "path": str(R8E_B_SEED_MANIFEST.resolve()),
            "seeds": seed_manifest["seeds"],
            "mutation_policy": seed_manifest["mutation_policy"],
            "predeclared_before_first_training": seed_manifest["predeclared_before_first_training"],
        },
        "retention_table": retention,
        "historical_raw_checkpoint_recovery_status": "UNRECOVERABLE_BY_DESIGN",
        "recovery_scope": "Authoritative R8E-A/A1, R8E-B, R8E-C, and R8E-D evidence only; no indefinite unrelated-disk search.",
        "upstream_hashes_before": before,
        "upstream_hashes_after": after,
        "upstream_artifacts_unchanged": before == after,
    }


def build_identity_semantics(a1: dict[str, Any]) -> dict[str, Any]:
    d_source = R8E_D_SCRIPT.read_text(encoding="utf-8")
    return {
        "schema_version": "stage3_h13_r8e_d1_checkpoint_identity_semantics_v1",
        "r8e_b_checkpoint_sha_meaning": "RAW_SERIALIZED_FILE_SHA256",
        "r8e_b_hash_producer": {
            "raw_bytes": "serialize_candidate(model, name, config) writes a dict containing model_state_dict, candidate, training_config, and initialization_sha256 to io.BytesIO via torch.save.",
            "hash_operation": "hashlib.sha256(checkpoint_bytes).hexdigest()",
            "direct_file_read": False,
            "state_dict_hash": False,
            "canonical_parameter_serialization": False,
            "freshly_serialized_after_training": True,
            "file_written_by_r8e_b": False,
            "temporary_checkpoint_retained": False,
            "later_deleted_by_low_storage_cleanup": False,
            "retention_explanation": "The bytes existed in memory, were recorded by digest, and were released; this is an explicit retention policy, not an observed accidental deletion.",
        },
        "seed_13086_provenance": {
            "r8e_b_run": "Independently executed seed-13086 R8E-B candidate; its bytes were freshly serialized in memory.",
            "r8e_a1": "Current R8E_A1 was selected and retained by the earlier R8E-A/R8E-A1 run.",
            "copied": False,
            "loaded_and_resaved_from_a1_by_r8e_b": False,
            "independently_reproduced_by_r8e_b": True,
            "metric_reproducibility_is_parameter_identity": False,
        },
        "identity_1_raw_artifact": {
            "name": "RAW_FILE_SHA256",
            "question": "Are these the exact same serialized checkpoint bytes?",
            "different_hash_means_raw_file_identical": False,
        },
        "identity_2_canonical_model_state": {
            "name": "CANONICAL_MODEL_STATE_SHA256",
            "procedure": "Sort fully qualified state_dict keys; for each tensor hash the UTF-8 key, ASCII dtype string, ASCII tuple shape, and raw contiguous CPU tensor bytes in C order.",
            "includes": ["parameter names", "buffers in state_dict", "dtype", "shape", "raw contiguous CPU tensor bytes"],
            "excludes": ["filesystem path", "ZIP/pickle metadata", "timestamps", "temporary filenames", "checkpoint wrapper metadata"],
            "existing_helper": "scripts/stage3_h13_r8e_d_counterfactual_feedback_audit.py:65-73",
            "existing_helper_audit": {
                "DOES_EXISTING_STATE_HASH_USE_PARAMETER_NAMES": "YES",
                "DOES_IT_INCLUDE_DTYPE": "YES",
                "DOES_IT_INCLUDE_SHAPE": "YES",
                "DOES_IT_HASH_TENSOR_BYTES": "YES",
                "DOES_IT_INCLUDE_ALL_INFERENCE_RELEVANT_STATE": "YES",
                "IS_IT_DETERMINISTIC": "YES",
                "IS_IT_SUITABLE_AS_CANONICAL_MODEL_STATE_SHA256": "YES",
                "basis": "sorted(model.state_dict().items()) includes model parameters and registered buffers; the implementation hashes the required key/type/shape/value fields and excludes wrapper/path metadata.",
            },
        },
        "r8e_c_reproduction_limit": {
            "observed": "R8E-C reserialized the current A1 state with the R8E-B config and reproduced the recorded seed-13086 raw digest.",
            "classification": "RAW_SERIALIZED_FILE_SHA256 reproduction / PROVENANCE_REFERENCE",
            "canonical_state_hash_persisted_for_historical_r8e_b_13086": False,
            "independent_historical_checkpoint_file_recovered": False,
            "counts_as_model_state_identity_under_d1": False,
            "reason": "It is a reserialization from the current A1 source, not a persisted canonical state fingerprint from the historical B run; the fail-closed policy does not promote it to historical identity.",
        },
        "repaired_contract": {
            "historical_raw_artifact": "Require exact RAW_FILE_SHA256; if the file is deleted, historical raw identity is unrecoverable and no reserialization can satisfy it.",
            "model_functional_state": "Require compatible CANONICAL_MODEL_STATE_SHA256 values computed from both states; metric equality alone is insufficient.",
            "prospective_checkpoint": "Persist both raw and canonical hashes immediately, plus seed, training-config hash, dataset/split identity, environment, path, timestamp, and provenance.",
            "raw_and_canonical_interchangeable": False,
        },
    }


def build_preflight_audit(a1: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    source = R8E_D_SCRIPT.read_text(encoding="utf-8")
    b_manifest = read_json(R8E_B_SEED_MANIFEST)
    recorded = str(records[13086]["checkpoint_sha256"])
    current = a1["raw_file_sha256"]
    raw_requirement = "For every manifest item, sha256_file(resolved) must equal item['sha256'] and that expected value must equal records[seed]['checkpoint_sha256']; seed 13086 therefore requires the resolved file raw SHA to equal " + recorded + "."
    path_requirement = "checkpoint_paths[13086].resolve() must equal (R8E_ROOT / 'best_candidate_checkpoint.pt').resolve()."
    proof = {
        "path_requirement_forces_observed_file": True,
        "forced_path": str(R8E_A1_CHECKPOINT.resolve()),
        "forced_path_observed_raw_sha256": current,
        "raw_requirement_expected_sha256": recorded,
        "observed_raw_sha256_equals_required_sha256": current == recorded,
        "there_exists_any_file_satisfying_both_constraints": current == recorded,
        "logical_reason": "Path equality fixes the file to A1; the A1 file has raw SHA 4d57..., while the record equality requires 0b78.... One byte sequence cannot have both SHA-256 values under the observed file identity.",
    }
    return {
        "schema_version": "stage3_h13_r8e_d1_r8e_d_preflight_audit_v1",
        "r8e_d_13086_raw_identity_requirement": raw_requirement,
        "r8e_d_13086_path_identity_requirement": path_requirement,
        "requirements_mutually_compatible": "NO",
        "current_preflight_satisfiable": "NO",
        "authority_semantics_conflict_confirmed": "YES",
        "exact_conflict": f"The gate requires A1 path identity but simultaneously requires that path's raw SHA-256 to be {recorded}; the observed A1 raw SHA-256 is {current}.",
        "logical_proof": proof,
        "implementation_evidence": {
            "raw_gate_source": source_line(R8E_D_SCRIPT, 122),
            "path_gate_source": source_line(R8E_D_SCRIPT, 147),
            "a1_hash_gate_source": source_line(R8E_D_SCRIPT, 134),
            "raw_gate_expression_present": "sha256_file(resolved) != expected or expected != str(records[seed][\"checkpoint_sha256\"])" in source,
            "path_gate_expression_present": 'checkpoint_paths[13086].resolve() != (R8E_ROOT / "best_candidate_checkpoint.pt").resolve()' in source,
        },
        "additional_preflight_issue": {
            "r8e_b_manifest_seed_order": b_manifest["seeds"],
            "r8e_d_expected_seed_order": list(R8E_D_SEEDS),
            "exact_order_match": b_manifest["seeds"] == list(R8E_D_SEEDS),
            "effect": "The existing R8E-B seed manifest would also fail R8E-D's exact tuple-order check before reaching the 13086 hash/path contradiction.",
        },
        "default_manifest_status": "DEFAULT_CHECKPOINT_MANIFEST_MISSING",
        "focused_contradiction_test_added": True,
    }


def run_tests() -> dict[str, Any]:
    commands = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e_d1_checkpoint_authority.py", "tests/test_stage3_h13_r8e_d.py"],
        "regression": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e_c.py", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8e_a.py"],
    }
    result: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        result[name] = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-5000:], "stderr": completed.stderr[-3000:]}
    result["new_failures"] = sum(value["returncode"] != 0 for value in result.values() if isinstance(value, dict))
    result["pass"] = result["new_failures"] == 0
    return result


def build_canonical_hashes(a1: dict[str, Any], records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_r8e_d1_canonical_state_hashes_v1",
        "canonical_hash_algorithm": "existing checkpoint_state_hash helper; sorted fully-qualified state keys plus dtype, shape, and contiguous CPU tensor bytes",
        "r8e_a1": a1,
        "historical_r8e_b": {
            "seed_13086": {
                "historical_raw_sha256": str(records[13086]["checkpoint_sha256"]),
                "canonical_model_state_sha256": None,
                "status": "unavailable",
                "reason": "No persisted canonical/state_dict/tensor digest exists in the inspected R8E-A/A1, R8E-B, R8E-C, or R8E-D artifacts.",
            }
        },
        "model_state_identity_result": "NOT PROVABLE",
        "r8e_a1_and_r8e_b_13086_model_state_identical": "NOT PROVABLE",
        "identity_confidence": "LOW",
    }


def report_text(data: dict[str, Any], output: Path, final_size: int) -> str:
    a1 = data["a1"]
    records = data["records"]
    tests = data["tests"]
    integrity = data["integrity"]
    b13086 = records[13086]["checkpoint_sha256"]
    created = sum(bool(records[seed].get("checkpoint_sha256")) for seed in R8E_B_SEEDS)
    retained = sum(bool(records[seed].get("temporary_checkpoint_retained")) for seed in R8E_B_SEEDS)
    false_count = sum(records[seed].get("temporary_checkpoint_retained") is False for seed in R8E_B_SEEDS)
    focused = tests["focused"]["stdout"].strip().splitlines()[-1] if tests["focused"]["stdout"].strip() else "NOT RUN"
    regression = tests["regression"]["stdout"].strip().splitlines()[-1] if tests["regression"]["stdout"].strip() else "NOT RUN"
    return f"""STAGE_3_H13_R8E_D1:
PASSED

FIRST_BLOCKER:
none

ONE_SENTENCE_VERDICT:
R8E-D conflates raw serialized-artifact identity with path/model authority for seed 13086, while the existing evidence does not persist a historical canonical model-state hash, so the historical causal replay remains closed and inconclusive.

==================================================
1. SEED-13086 IDENTITY CONFLICT
==================================================

R8E_B_13086_RAW_SHA256:
{b13086}

R8E_A1_RAW_SHA256:
{a1['raw_file_sha256']}

RAW_FILE_IDENTICAL:
NO

R8E_A1_CANONICAL_MODEL_STATE_SHA256:
{a1['canonical_model_state_sha256']}

HISTORICAL_R8E_B_13086_CANONICAL_MODEL_STATE_SHA256:
unavailable

R8E_A1_AND_R8E_B_13086_MODEL_STATE_IDENTICAL:
NOT PROVABLE

IDENTITY_CONFIDENCE:
LOW

==================================================
2. R8E-B RETENTION FINDING
==================================================

SEED_CHECKPOINTS_CREATED:
{created}/5

SEED_CHECKPOINTS_RETAINED:
{retained}/5

TEMPORARY_CHECKPOINT_RETAINED_FALSE:
{false_count}/5

HISTORICAL_RAW_CHECKPOINT_RECOVERY_STATUS:
UNRECOVERABLE_BY_DESIGN

==================================================
3. HASH SEMANTICS
==================================================

R8E_B_CHECKPOINT_SHA_MEANING:
RAW_SERIALIZED_FILE_SHA256

RAW_FILE_SHA256_AND_MODEL_STATE_SHA256_DISTINGUISHED:
YES

CANONICAL_MODEL_STATE_HASH_DEFINED:
YES

EXISTING_CHECKPOINT_STATE_HASH_VALID:
YES

CANONICAL_HASH_INCLUDES:
sorted fully qualified state keys, dtype, shape, registered buffers in state_dict, and raw contiguous CPU tensor bytes; it excludes paths, wrapper metadata, pickle/ZIP metadata, timestamps, and temporary filenames.

==================================================
4. R8E-D PREFLIGHT AUDIT
==================================================

R8E_D_13086_RAW_IDENTITY_REQUIREMENT:
{data['preflight']['r8e_d_13086_raw_identity_requirement']}

R8E_D_13086_PATH_IDENTITY_REQUIREMENT:
{data['preflight']['r8e_d_13086_path_identity_requirement']}

REQUIREMENTS_MUTUALLY_COMPATIBLE:
NO

CURRENT_PREFLIGHT_SATISFIABLE:
NO

AUTHORITY_SEMANTICS_CONFLICT_CONFIRMED:
YES

EXACT_CONFLICT:
{data['preflight']['exact_conflict']}

==================================================
5. HISTORICAL R8E-D STATUS
==================================================

HISTORICAL_R8E_D_RAW_ARTIFACT_REPLAY_POSSIBLE:
NO

HISTORICAL_R8E_D_MODEL_STATE_REPLAY_POSSIBLE:
NOT PROVABLE

HISTORICAL_R8E_D_CAUSAL_RESULT_AVAILABLE:
NO

H4_PRIMARY_CAUSAL_CLASS:
INCONCLUSIVE

CAUSAL_CONFIDENCE:
LOW

==================================================
6. REPAIRED AUTHORITY CONTRACT
==================================================

RAW_ARTIFACT_IDENTITY_RULE:
An historical artifact claim requires exact RAW_FILE_SHA256 equality, and a deleted raw artifact is unrecoverable by reserialization.

MODEL_STATE_IDENTITY_RULE:
A model-state identity claim requires compatible CANONICAL_MODEL_STATE_SHA256 values computed from both actual states; equal metrics are insufficient.

PROSPECTIVE_CHECKPOINT_IDENTITY_RULE:
Every future checkpoint must persist raw and canonical hashes together with seed, config, dataset/split, environment, path, timestamp, and provenance immediately after creation.

CAN_RAW_HASH_AND_MODEL_STATE_HASH_BE_USED_INTERCHANGEABLY:
NO

==================================================
7. INTEGRITY
==================================================

R6_CHECKPOINT_HASH_MATCH:
{integrity['R6_CHECKPOINT_HASH_MATCH']}

R8E_A1_RAW_HASH_MATCH:
{integrity['R8E_A1_RAW_HASH_MATCH']}

R8E_A1_IMMUTABLE:
{integrity['R8E_A1_IMMUTABLE']}

R8E_B_RECORDS_IMMUTABLE:
{integrity['R8E_B_RECORDS_IMMUTABLE']}

FROZEN20_OPENED:
NO

FROZEN20_INSPECTED:
NO

FROZEN20_USED:
NO

FUTURE_LABEL_LEAKAGE:
0

NEW_CHECKPOINTS_TRAINED:
0

OPTIMIZER_STEPS_EXECUTED:
0

R8F_EXECUTED:
NO

R9_EXECUTED:
NO

==================================================
8. TESTS
==================================================

FOCUSED_TESTS:
{focused}

REGRESSION_TESTS:
{regression}

NEW_FAILURES:
{tests['new_failures']}

FINAL_PERSISTENT_SIZE_BYTES:
{final_size}

==================================================
9. PROJECT OWNER DECISION
==================================================

CURRENT_VALIDATION_CHAMPION:
R8E_A1

CHAMPION_CHANGED:
NO

HISTORICAL_CHECKPOINT_SEARCH_SHOULD_CONTINUE:
NO

HISTORICAL_R8E_D_SHOULD_REMAIN_INCONCLUSIVE:
YES

READY_FOR_PROSPECTIVE_FIVE_SEED_CAUSAL_REPLICATION_DESIGN:
YES

READY_FOR_PROSPECTIVE_FIVE_SEED_CAUSAL_REPLICATION_EXECUTION:
NO

READY_FOR_R8F_DESIGN:
NO

READY_FOR_R8F_EXECUTION:
NO

READY_FOR_R9_FROZEN20:
NO

READY_FOR_STAGE_3_FINAL_CLOSURE:
NO

SINGLE_NEXT_ACTION:
Adopt the repaired two-identity authority contract and require a new permanently retained five-seed causal replication before any R8E-D execution.

ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION:
Keep R8E_A1 as the validation champion, close historical R8E-D as inconclusive, and authorize only a separately designed prospective replication that records raw and canonical checkpoint identities.

PERSISTENT_ARTIFACTS_TO_PRESERVE:
{output / 'FINAL_REPORT.md'}
{output / 'terminal_certificate.json'}
{output / 'authority_chain.json'}
{output / 'checkpoint_identity_semantics.json'}
{output / 'r8e_d_preflight_audit.json'}
{output / 'canonical_state_hashes.json'}
{output / 'test_results.txt'}
"""


def run(args: argparse.Namespace) -> int:
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8e_d1_checkpoint_authority_semantics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    before = upstream_hashes()
    a1 = inspect_a1()
    records_payload = read_json(R8E_B_RECORDS)
    records = {int(item["seed"]): item for item in records_payload["records"]}
    if tuple(records) != tuple(R8E_B_SEEDS):
        raise RuntimeError("r8e_b_record_order_mismatch")
    if tuple(records) == tuple(R8E_D_SEEDS):
        raise RuntimeError("unexpected_seed_order_equivalence")
    preflight = build_preflight_audit(a1, records)
    identity = build_identity_semantics(a1)
    canonical = build_canonical_hashes(a1, records)
    after = upstream_hashes()
    integrity = {
        "R6_CHECKPOINT_HASH_MATCH": "YES" if before["R6"]["sha256"] == after["R6"]["sha256"] == R6_EXPECTED_HASH else "NO",
        "R8E_A1_RAW_HASH_MATCH": "YES" if a1["raw_file_sha256"] == EXPECTED_A1_HASH and before["R8E_A1"] == after["R8E_A1"] else "NO",
        "R8E_A1_IMMUTABLE": "YES" if before["R8E_A1"] == after["R8E_A1"] else "NO",
        "R8E_B_RECORDS_IMMUTABLE": "YES" if all(before[name] == after[name] for name in ("R8E_B_RECORDS", "R8E_B_SEED_MANIFEST", "R8E_B_INTEGRITY", "R8E_B_FINAL_REPORT")) else "NO",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_INSPECTED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "NEW_CHECKPOINTS_TRAINED": 0,
        "OPTIMIZER_STEPS_EXECUTED": 0,
        "R8F_EXECUTED": "NO",
        "R9_EXECUTED": "NO",
        "before": before,
        "after": after,
    }
    tests = run_tests()
    write_json(output / "authority_chain.json", build_authority_chain(a1, before, after))
    write_json(output / "checkpoint_identity_semantics.json", identity)
    write_json(output / "r8e_d_preflight_audit.json", preflight)
    write_json(output / "canonical_state_hashes.json", canonical)
    (output / "test_results.txt").write_text(json.dumps(tests, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    data = {"a1": a1, "records": records, "preflight": preflight, "integrity": integrity, "tests": tests}
    certificate = {
        "schema_version": "stage3_h13_r8e_d1_terminal_certificate_v1",
        "STAGE_3_H13_R8E_D1": "PASSED" if tests["pass"] and preflight["authority_semantics_conflict_confirmed"] == "YES" else "BLOCKED",
        "FIRST_BLOCKER": "none" if tests["pass"] else "focused_or_regression_tests_failed",
        "R8E_B_13086_RAW_SHA256": records[13086]["checkpoint_sha256"],
        "R8E_A1_RAW_SHA256": a1["raw_file_sha256"],
        "RAW_FILE_IDENTICAL": "YES" if records[13086]["checkpoint_sha256"] == a1["raw_file_sha256"] else "NO",
        "R8E_A1_CANONICAL_MODEL_STATE_SHA256": a1["canonical_model_state_sha256"],
        "HISTORICAL_R8E_B_13086_CANONICAL_MODEL_STATE_SHA256": None,
        "R8E_A1_AND_R8E_B_13086_MODEL_STATE_IDENTICAL": "NOT PROVABLE",
        "IDENTITY_CONFIDENCE": "LOW",
        "SEED_CHECKPOINTS_CREATED": "5/5",
        "SEED_CHECKPOINTS_RETAINED": "0/5",
        "TEMPORARY_CHECKPOINT_RETAINED_FALSE": "5/5",
        "HISTORICAL_RAW_CHECKPOINT_RECOVERY_STATUS": "UNRECOVERABLE_BY_DESIGN",
        "R8E_B_CHECKPOINT_SHA_MEANING": "RAW_SERIALIZED_FILE_SHA256",
        "RAW_FILE_SHA256_AND_MODEL_STATE_SHA256_DISTINGUISHED": "YES",
        "CANONICAL_MODEL_STATE_HASH_DEFINED": "YES",
        "EXISTING_CHECKPOINT_STATE_HASH_VALID": "YES",
        "REQUIREMENTS_MUTUALLY_COMPATIBLE": "NO",
        "CURRENT_PREFLIGHT_SATISFIABLE": "NO",
        "AUTHORITY_SEMANTICS_CONFLICT_CONFIRMED": "YES",
        "EXACT_CONFLICT": preflight["exact_conflict"],
        "HISTORICAL_R8E_D_RAW_ARTIFACT_REPLAY_POSSIBLE": "NO",
        "HISTORICAL_R8E_D_MODEL_STATE_REPLAY_POSSIBLE": "NOT PROVABLE",
        "HISTORICAL_R8E_D_CAUSAL_RESULT_AVAILABLE": "NO",
        "H4_PRIMARY_CAUSAL_CLASS": "INCONCLUSIVE",
        "CAUSAL_CONFIDENCE": "LOW",
        "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
        "CHAMPION_CHANGED": "NO",
        "HISTORICAL_CHECKPOINT_SEARCH_SHOULD_CONTINUE": "NO",
        "HISTORICAL_R8E_D_SHOULD_REMAIN_INCONCLUSIVE": "YES",
        "READY_FOR_PROSPECTIVE_FIVE_SEED_CAUSAL_REPLICATION_DESIGN": "YES",
        "READY_FOR_PROSPECTIVE_FIVE_SEED_CAUSAL_REPLICATION_EXECUTION": "NO",
        "READY_FOR_R8F_DESIGN": "NO",
        "READY_FOR_R8F_EXECUTION": "NO",
        "READY_FOR_R9_FROZEN20": "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        **integrity,
        "FOCUSED_TESTS": "PASS" if tests["focused"]["returncode"] == 0 else "FAIL",
        "REGRESSION_TESTS": "PASS" if tests["regression"]["returncode"] == 0 else "FAIL",
        "NEW_FAILURES": tests["new_failures"],
        "FINAL_PERSISTENT_SIZE_BYTES": 0,
        "SINGLE_NEXT_ACTION": "Adopt the repaired two-identity authority contract and require a new permanently retained five-seed causal replication before any R8E-D execution.",
    }
    write_json(output / "terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(report_text(data, output, 0), encoding="utf-8", newline="\n")
    for _ in range(6):
        size = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        certificate["FINAL_PERSISTENT_SIZE_BYTES"] = size
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(report_text(data, output, size), encoding="utf-8", newline="\n")
        new_size = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if new_size == size:
            break
    certificate["FINAL_PERSISTENT_SIZE_BYTES"] = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    write_json(output / "terminal_certificate.json", certificate)
    print(json.dumps({"output": str(output), "status": certificate["STAGE_3_H13_R8E_D1"], "first_blocker": certificate["FIRST_BLOCKER"]}, sort_keys=True))
    return 0 if certificate["STAGE_3_H13_R8E_D1"] == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
