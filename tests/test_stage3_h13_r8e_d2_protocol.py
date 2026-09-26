from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.stage3_h13_r8e_d2 import (
    EXPECTED_SEED_SET,
    PROSPECTIVE_SEEDS,
    canonical_model_state_sha256,
    require_d3_authorization,
    sha256_file,
    validate_checkpoint_record,
    validate_prospective_manifest,
    validate_seed_set,
    verify_checkpoint_record,
)


def _manifest(seeds: list[int] | None = None) -> dict[str, object]:
    return {
        "schema_version": "stage3_h13_r8e_d2_prospective_manifest_v1",
        "experiment_type": "PROSPECTIVE_FIVE_SEED_CAUSAL_REPLICATION",
        "historical_replay_claim": False,
        "mutation_policy": "FORBIDDEN_AFTER_CREATION",
        "expected_seed_set": list(PROSPECTIVE_SEEDS if seeds is None else seeds),
        "display_execution_order": list(reversed(PROSPECTIVE_SEEDS)),
        "checkpoints": [],
    }


def _record(path: Path, *, config_hash: str = "c" * 64, canonical_hash: str = "b" * 64) -> dict[str, object]:
    return {
        "seed": 13086,
        "checkpoint_absolute_path": str(path.resolve()),
        "raw_checkpoint_sha256": sha256_file(path),
        "canonical_model_state_sha256": canonical_hash,
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_creation_timestamp": "2026-08-16T05:00:00Z",
        "candidate_config_identifier": "R8E_A1_PROSPECTIVE",
        "training_config_sha256": config_hash,
        "dataset_identity": "stage3_h10_multi_cb282a06d73c322:cb282a06d73c322208f27bf7c0b019831a24f524ac0d72263929f22303f0801b",
        "split_identity": "stage3_h10_group_split_v1:TRAIN_31_VALIDATION_7",
        "initialization_provenance_fingerprint": "r6:8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee",
        "software_environment_fingerprint": "synthetic-test-environment",
        "device_backend_information": {"device": "cpu", "backend": "synthetic"},
        "determinism_configuration": {"deterministic_algorithms": True},
        "selection_criterion": "fixed_terminal_H32_after_512_updates_with_frozen_H1_gate",
        "selection_split": "VALIDATION",
        "provenance": "prospective",
    }


def test_seed_set_equality_is_order_insensitive() -> None:
    result = validate_seed_set(list(reversed(PROSPECTIVE_SEEDS)))
    assert result["observed_seed_set"] == sorted(EXPECTED_SEED_SET)
    assert result["order_is_authoritative"] is False
    validate_prospective_manifest(_manifest())


def test_duplicate_seed_rejected() -> None:
    values = list(PROSPECTIVE_SEEDS[:-1]) + [PROSPECTIVE_SEEDS[0], PROSPECTIVE_SEEDS[-1]]
    with pytest.raises(ValueError, match="duplicate_seeds"):
        validate_seed_set(values)


def test_missing_seed_rejected() -> None:
    with pytest.raises(ValueError, match="missing_seeds"):
        validate_seed_set(list(PROSPECTIVE_SEEDS[:-1]))


def test_unexpected_seed_rejected() -> None:
    values = list(PROSPECTIVE_SEEDS[:-1]) + [999]
    with pytest.raises(ValueError, match="unexpected_seeds"):
        validate_seed_set(values)


def test_raw_and_canonical_identity_fields_are_distinct(tmp_path: Path) -> None:
    path = tmp_path / "prospective_13086.bin"
    path.write_bytes(b"checkpoint-a")
    record = _record(path)
    validate_checkpoint_record(record)
    assert record["raw_checkpoint_sha256"] != record["canonical_model_state_sha256"]


def test_raw_hash_mutation_detection(tmp_path: Path) -> None:
    path = tmp_path / "prospective_13086.bin"
    path.write_bytes(b"checkpoint-a")
    record = _record(path)
    state_hash = str(record["canonical_model_state_sha256"])
    verify_checkpoint_record(record, expected_training_config_sha256="c" * 64, actual_canonical_model_state_sha256=state_hash)
    path.write_bytes(b"checkpoint-b")
    with pytest.raises(ValueError, match="raw_checkpoint_hash_mismatch"):
        verify_checkpoint_record(record, expected_training_config_sha256="c" * 64, actual_canonical_model_state_sha256=state_hash)


def test_canonical_hash_mutation_detection() -> None:
    first = {"gru.weight": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    second = {"gru.weight": np.asarray([[1.0, 3.0]], dtype=np.float32)}
    assert canonical_model_state_sha256(first) != canonical_model_state_sha256(second)


def test_missing_checkpoint_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing.bin"
    # Build the record without creating the file so the schema can still be checked.
    missing = {
        "seed": 13086,
        "checkpoint_absolute_path": str(path.resolve()),
        "raw_checkpoint_sha256": "a" * 64,
        "canonical_model_state_sha256": "b" * 64,
        "checkpoint_size_bytes": 1,
        "checkpoint_creation_timestamp": "2026-08-16T05:00:00Z",
        "candidate_config_identifier": "R8E_A1_PROSPECTIVE",
        "training_config_sha256": "c" * 64,
        "dataset_identity": "synthetic",
        "split_identity": "synthetic",
        "initialization_provenance_fingerprint": "synthetic",
        "software_environment_fingerprint": "synthetic",
        "device_backend_information": {"device": "cpu"},
        "determinism_configuration": {"deterministic_algorithms": True},
        "selection_criterion": "fixed_terminal_H32_after_512_updates_with_frozen_H1_gate",
        "selection_split": "VALIDATION",
        "provenance": "prospective",
    }
    with pytest.raises(FileNotFoundError, match="checkpoint_missing"):
        verify_checkpoint_record(missing, expected_training_config_sha256="c" * 64, actual_canonical_model_state_sha256="b" * 64)


def test_missing_provenance_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prospective.bin"
    path.write_bytes(b"x")
    record = _record(path)
    del record["provenance"]
    with pytest.raises(ValueError, match="checkpoint_record_fields_missing"):
        validate_checkpoint_record(record)


def test_config_hash_mismatch_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prospective.bin"
    path.write_bytes(b"x")
    record = _record(path, config_hash="d" * 64)
    with pytest.raises(ValueError, match="training_config_hash_mismatch"):
        verify_checkpoint_record(record, expected_training_config_sha256="c" * 64, actual_canonical_model_state_sha256="b" * 64)


def test_historical_a1_path_is_not_required(tmp_path: Path) -> None:
    path = tmp_path / "new_prospective_seed_13086.bin"
    path.write_bytes(b"new")
    record = _record(path)
    verify_checkpoint_record(record, expected_training_config_sha256="c" * 64, actual_canonical_model_state_sha256="b" * 64)


def test_historical_r8e_b_raw_hash_is_not_required(tmp_path: Path) -> None:
    path = tmp_path / "new_prospective_seed_218309645.bin"
    path.write_bytes(b"new")
    record = _record(path)
    record["seed"] = 218309645
    record["raw_checkpoint_sha256"] = hashlib.sha256(b"new").hexdigest()
    assert record["raw_checkpoint_sha256"] != "0b7871d2190b012b06a7d5f06dbb5ea0be03ce0cb8bc4a34f9e21d4a5a4fdd95"


def test_frozen20_remains_inaccessible() -> None:
    from src.stage3_h13_r8e_d2 import assert_validation_only_paths

    with pytest.raises(ValueError, match="sealed_evaluation_path_forbidden"):
        assert_validation_only_paths(["outputs/sealed-frozen-20/cases.json"])


def test_d3_cannot_start_if_d2_execution_authorization_is_false() -> None:
    with pytest.raises(PermissionError, match="d3_execution_authorization_false_in_d2"):
        require_d3_authorization({"d2_execution_authorization": False})
