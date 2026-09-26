"""D64 explicit frozen-artifact manifest and claim-fence tests."""

from __future__ import annotations

import json
from pathlib import Path

from tools.d64_freeze_project import (
    MANIFEST_NAME,
    collect_artifacts,
    manifest_payload,
    git_status_for_authority,
    verify_manifest,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_manifest_is_small_and_non_recursive() -> None:
    payload = manifest_payload()
    assert payload["schema_version"] == "d64-frozen-artifact-manifest-v1"
    assert payload["self_excluded"] is True
    assert payload["artifact_count"] == len(payload["artifacts"]) <= 24
    assert all(record["path"] != f"outputs/D64_PROJECT_FREEZE/{MANIFEST_NAME}" for record in payload["artifacts"])
    assert all(set(record) == {"role", "path", "size_bytes", "sha256"} for record in payload["artifacts"])
    assert all("case_metrics" not in record["path"] and not record["path"].endswith(".jsonl") for record in payload["artifacts"])


def test_manifest_detects_mutation_and_passes_after_exact_restore(tmp_path: Path) -> None:
    authority = tmp_path / "authority.txt"
    authority.write_bytes(b"frozen bytes\n")
    specs = ({"role": "test_authority", "path": "authority.txt"},)
    manifest = write_manifest(tmp_path, specs=specs)

    initial = verify_manifest(tmp_path, manifest_path=manifest, specs=specs)
    assert initial["status"] == "PASS"
    assert initial["checked_count"] == 1

    authority.write_bytes(b"tampered bytes\n")
    tampered = verify_manifest(tmp_path, manifest_path=manifest, specs=specs)
    assert tampered["status"] == "FAIL"
    assert any(item["kind"] in {"size", "digest"} for item in tampered["mismatches"])

    authority.write_bytes(b"frozen bytes\n")
    restored = verify_manifest(tmp_path, manifest_path=manifest, specs=specs)
    assert restored["status"] == "PASS"
    assert restored["mismatches"] == []


def test_manifest_scope_drift_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one\n", encoding="utf-8")
    second.write_text("two\n", encoding="utf-8")
    original_specs = ({"role": "first", "path": "first.txt"},)
    changed_specs = (
        {"role": "first", "path": "first.txt"},
        {"role": "second", "path": "second.txt"},
    )
    manifest = write_manifest(tmp_path, specs=original_specs)
    result = verify_manifest(tmp_path, manifest_path=manifest, specs=changed_specs)
    assert result["status"] == "FAIL"
    assert any(item["kind"] == "scope" for item in result["mismatches"])


def test_manifest_records_are_reproducible() -> None:
    first = collect_artifacts()
    second = collect_artifacts()
    assert first == second


def test_authority_surface_is_visible_to_git_status() -> None:
    status = git_status_for_authority()
    assert status["available"] is True
    assert status["ignored_paths"] == []


def test_lineage_claim_fence_keeps_d59_historical_and_d62_current_boundary() -> None:
    d59 = json.loads((ROOT / "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/D59_FINAL_STATUS.json").read_text(encoding="utf-8"))
    d62 = json.loads((ROOT / "outputs/D62_CONTINUOUS_CERTIFICATE_SHADOW/D62_FINAL_STATUS.json").read_text(encoding="utf-8"))
    d64 = json.loads((ROOT / "outputs/D64_PROJECT_FREEZE/D64_FREEZE_STATUS.json").read_text(encoding="utf-8"))

    assert d59["ARTICULATED_SELF_CCD_CLAIM"]["state"].startswith("HISTORICAL_SCOPED_")
    assert d59["ARTICULATED_SELF_CCD_CLAIM"]["exact_external_articulated_fk_qt_ccd"] == "UNAVAILABLE_UNVERIFIED"
    assert d62["claim_scope"].endswith("not a continuous-certificate PASS")
    assert d64["lineage"]["D59"]["state"].startswith("HISTORICAL_SCOPED_")
    assert d64["lineage"]["D62"]["cases"]["auto0"]["unresolved_intervals"] > 0
    assert d64["lineage"]["D63"]["safety_status"] == "UNRESOLVED"
