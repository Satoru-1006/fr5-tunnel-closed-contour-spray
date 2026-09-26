"""Pure validation helpers for the Stage 3 H13 R8E-D2 design.

This module is deliberately a protocol/schema layer.  It does not import the
training runner, construct a model, load a checkpoint, run causal inference,
or access any sealed evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROSPECTIVE_SEEDS = (13086, 218309645, 258275761, 1638377564, 1727279519)
EXPECTED_SEED_SET = frozenset(PROSPECTIVE_SEEDS)
SEED_COUNT = 5
SCHEMA_VERSION = "stage3_h13_r8e_d2_prospective_manifest_v1"
MUTATION_POLICY = "FORBIDDEN_AFTER_CREATION"
D2_EXECUTION_AUTHORIZATION = False
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEALED_TOKENS = ("frozen20", "frozen_20", "frozen-20", "stage3_h13_r3_frozen")

REQUIRED_CHECKPOINT_FIELDS = frozenset(
    {
        "seed",
        "checkpoint_absolute_path",
        "raw_checkpoint_sha256",
        "canonical_model_state_sha256",
        "checkpoint_size_bytes",
        "checkpoint_creation_timestamp",
        "candidate_config_identifier",
        "training_config_sha256",
        "dataset_identity",
        "split_identity",
        "initialization_provenance_fingerprint",
        "software_environment_fingerprint",
        "device_backend_information",
        "determinism_configuration",
        "selection_criterion",
        "selection_split",
        "provenance",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON bytes used for all D2 config fingerprints."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_model_state_sha256(state_dict: Mapping[str, Any]) -> str:
    """Use the audited R8E-D1 state identity semantics.

    Keys are fully-qualified and sorted.  Each tensor contributes its key,
    dtype, shape, and contiguous CPU bytes.  Wrapper/path/timestamp metadata
    is intentionally outside this function.
    """

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "contiguous"):
            value = value.contiguous()
        if hasattr(value, "numpy"):
            array = value.numpy()
            dtype_text = str(getattr(value, "dtype", array.dtype))
        else:
            array = np.asarray(value)
            dtype_text = str(array.dtype)
        array = np.ascontiguousarray(array)
        digest.update(str(name).encode("utf-8"))
        digest.update(dtype_text.encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _seed_values(values: Sequence[Any], *, field: str) -> list[int]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field}_must_be_list")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{field}_contains_boolean")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}_contains_non_integer") from exc
        if converted != value:
            raise ValueError(f"{field}_contains_non_integral")
        result.append(converted)
    return result


def validate_seed_set(values: Sequence[Any], *, field: str = "seed_set") -> dict[str, Any]:
    """Validate seed identity by set membership, never by list position."""

    seeds = _seed_values(values, field=field)
    duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    actual = set(seeds)
    missing = sorted(EXPECTED_SEED_SET - actual)
    unexpected = sorted(actual - EXPECTED_SEED_SET)
    if duplicates:
        raise ValueError(f"duplicate_seeds:{duplicates}")
    if missing or unexpected:
        raise ValueError(f"seed_set_mismatch:missing_seeds:{missing}:unexpected_seeds:{unexpected}")
    if len(seeds) != SEED_COUNT or len(actual) != SEED_COUNT:
        raise ValueError("seed_count_not_five")
    return {
        "expected_seed_set": sorted(EXPECTED_SEED_SET),
        "observed_seed_set": sorted(actual),
        "seed_count": len(seeds),
        "unique_seed_count": len(actual),
        "duplicate_seeds": duplicates,
        "missing_seeds": missing,
        "unexpected_seeds": unexpected,
        "order_is_authoritative": False,
    }


def assert_validation_only_paths(paths: Sequence[str | Path]) -> None:
    for path in paths:
        lowered = str(path).replace("\\", "/").lower()
        if any(token in lowered for token in _SEALED_TOKENS):
            raise ValueError("sealed_evaluation_path_forbidden")


def validate_prospective_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prospective_manifest_schema_version_mismatch")
    if payload.get("experiment_type") != "PROSPECTIVE_FIVE_SEED_CAUSAL_REPLICATION":
        raise ValueError("prospective_manifest_experiment_type_mismatch")
    if payload.get("historical_replay_claim") is not False:
        raise ValueError("historical_replay_claim_must_be_false")
    if payload.get("mutation_policy") != MUTATION_POLICY:
        raise ValueError("manifest_mutation_policy_missing")
    seed_summary = validate_seed_set(payload.get("expected_seed_set", ()), field="expected_seed_set")
    display_order = payload.get("display_execution_order", payload.get("expected_seed_set", ()))
    validate_seed_set(display_order, field="display_execution_order")
    checkpoints = payload.get("checkpoints", ())
    if not isinstance(checkpoints, list):
        raise ValueError("checkpoints_must_be_list")
    if checkpoints:
        validate_seed_set([item.get("seed") for item in checkpoints], field="checkpoint_seeds")
        if any(item.get("provenance") != "prospective" for item in checkpoints):
            raise ValueError("prospective_checkpoint_provenance_missing")
    assert_validation_only_paths(payload.get("sealed_paths_checked", ()))
    return seed_summary


def validate_checkpoint_record(record: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_CHECKPOINT_FIELDS - set(record))
    if missing:
        raise ValueError(f"checkpoint_record_fields_missing:{missing}")
    seed = int(record["seed"])
    if seed not in EXPECTED_SEED_SET:
        raise ValueError(f"checkpoint_seed_unexpected:{seed}")
    path = Path(str(record["checkpoint_absolute_path"]))
    if not path.is_absolute():
        raise ValueError("checkpoint_path_must_be_absolute")
    assert_validation_only_paths([path])
    if record["provenance"] != "prospective":
        raise ValueError("checkpoint_provenance_must_be_prospective")
    if record["selection_split"] != "VALIDATION":
        raise ValueError("checkpoint_selection_split_must_be_validation")
    if record["selection_criterion"] != "fixed_terminal_H32_after_512_updates_with_frozen_H1_gate":
        raise ValueError("checkpoint_selection_criterion_mismatch")
    if record["training_config_sha256"] == "":
        raise ValueError("training_config_hash_missing")
    for field in ("raw_checkpoint_sha256", "canonical_model_state_sha256", "training_config_sha256"):
        value = str(record[field])
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid_sha256:{field}")
    if int(record["checkpoint_size_bytes"]) <= 0:
        raise ValueError("checkpoint_size_missing_or_nonpositive")
    if not str(record["checkpoint_creation_timestamp"]):
        raise ValueError("checkpoint_creation_timestamp_missing")


def verify_checkpoint_record(
    record: Mapping[str, Any],
    *,
    expected_training_config_sha256: str,
    actual_canonical_model_state_sha256: str,
) -> dict[str, Any]:
    validate_checkpoint_record(record)
    path = Path(str(record["checkpoint_absolute_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint_missing:{record['seed']}")
    actual_raw = sha256_file(path)
    actual_size = path.stat().st_size
    if actual_raw != record["raw_checkpoint_sha256"]:
        raise ValueError(f"raw_checkpoint_hash_mismatch:{record['seed']}")
    if actual_size != int(record["checkpoint_size_bytes"]):
        raise ValueError(f"checkpoint_size_mismatch:{record['seed']}")
    if actual_canonical_model_state_sha256 != record["canonical_model_state_sha256"]:
        raise ValueError(f"canonical_model_state_hash_mismatch:{record['seed']}")
    if str(record["training_config_sha256"]) != str(expected_training_config_sha256):
        raise ValueError(f"training_config_hash_mismatch:{record['seed']}")
    return {
        "seed": int(record["seed"]),
        "raw_hash_verified": True,
        "canonical_hash_verified": True,
        "size_verified": True,
        "config_hash_verified": True,
        "mutation_policy": MUTATION_POLICY,
    }


def require_d3_authorization(payload: Mapping[str, Any]) -> None:
    if payload.get("d2_execution_authorization") is not True or not D2_EXECUTION_AUTHORIZATION:
        raise PermissionError("d3_execution_authorization_false_in_d2")


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
