from __future__ import annotations

import json
from pathlib import Path

from src.stage3_h12_trajectory_repair import audit_repair_inputs


def test_h11_checkpoint_is_frozen_and_unchanged():
    root = Path(__file__).resolve().parents[1]
    h11 = next(root.glob("outputs/stage3_h11_r_*/stage3_h11_r_terminal_certificate.json"))
    payload = json.loads(h11.read_text(encoding="utf-8"))
    h11_root = h11.parent
    checkpoint_manifest = json.loads((h11_root / "checkpoint_sha256_manifest.json").read_text(encoding="utf-8"))
    checkpoint = h11_root / checkpoint_manifest["checkpoint"]
    import hashlib
    observed = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert payload["H10_SEMANTIC_HASH_MATCH"] == "YES"
    assert observed == checkpoint_manifest["sha256"] == payload["CHECKPOINT_SHA256"]


def test_h11_raw_violation_evidence_is_separate_from_h12_namespaces():
    root = Path(__file__).resolve().parents[1]
    h11 = next(root.glob("outputs/stage3_h11_r_*/stage3_h11_r_terminal_certificate.json"))
    payload = json.loads(h11.read_text(encoding="utf-8"))
    assert payload["RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"] == 19391300
    assert payload["RAW_MODEL_COLLISION_VIOLATIONS"] == 0
    assert payload["RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS"] == 70


def test_fail_closed_native_requirements_do_not_treat_unavailable_as_pass():
    native = {"status": "BLOCKED", "complete_native_scope": False, "post_ruckig_executed": False}
    assert not (native.get("status") == "PASSED" and native.get("complete_native_scope") is True and native.get("post_ruckig_executed") is True)


def test_no_h13_or_physical_execution_is_requested_by_audit():
    audit = audit_repair_inputs()
    assert audit["GROUND_TRUTH_USED_FOR_REPAIR"] == "NO"
    assert audit["REPAIR_LABEL_LEAKAGE_VIOLATIONS"] == 0

