from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.certified_output_guard import (
    CertifiedOutputImmutableError,
    REFUSAL_TOKEN,
    assert_not_inside_certified_immutable_root,
    assert_output_path_writable,
    build_immutable_marker_payload,
    immutable_marker_for,
    install_immutable_marker,
)


def marker(root: Path) -> Path:
    certificate = root / "terminal_certificate.json"
    certificate.write_text("{}\n", encoding="utf-8")
    payload = build_immutable_marker_payload(
        root,
        reason="focused guard test",
        certificate=certificate,
        semantic_sha256="0" * 64,
    )
    return install_immutable_marker(root, payload)


def test_new_output_root_write_allowed(tmp_path: Path) -> None:
    assert_output_path_writable(tmp_path / "new" / "artifact.json", operation="create")


def test_existing_non_certified_scratch_write_allowed(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert_output_path_writable(scratch / "artifact.json", operation="overwrite")


@pytest.mark.parametrize("operation", ["write", "append", "truncate", "overwrite", "resume", "unlink_recreate", "rename_replace"])
def test_certified_root_mutations_denied(tmp_path: Path, operation: str) -> None:
    root = tmp_path / "certified"
    root.mkdir()
    installed = marker(root)
    target = root / "child" / "artifact.jsonl"
    with pytest.raises(CertifiedOutputImmutableError, match=REFUSAL_TOKEN):
        assert_output_path_writable(target, operation=operation)
    assert immutable_marker_for(target) == installed


def test_child_directory_is_denied(tmp_path: Path) -> None:
    root = tmp_path / "certified"
    child = root / "a" / "b"
    child.mkdir(parents=True)
    marker(root)
    with pytest.raises(CertifiedOutputImmutableError):
        assert_not_inside_certified_immutable_root(child / "x", operation="write")


def test_reads_and_hash_verification_remain_allowed(tmp_path: Path) -> None:
    root = tmp_path / "certified"
    root.mkdir()
    evidence = root / "evidence.json"
    evidence.write_text('{"passed":true}\n', encoding="utf-8")
    marker(root)
    assert json.loads(evidence.read_text(encoding="utf-8"))["passed"] is True


def test_marker_is_exclusive_and_schema_checked(tmp_path: Path) -> None:
    root = tmp_path / "certified"
    root.mkdir()
    installed = marker(root)
    with pytest.raises(FileExistsError):
        install_immutable_marker(root, json.loads(installed.read_text(encoding="utf-8")))
