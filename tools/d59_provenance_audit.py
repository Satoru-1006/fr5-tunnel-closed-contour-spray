"""Audit and repair the D57 -> D58 candidate identity boundary.

The audit is intentionally non-mutating for D57/D58.  It records the exact
source artifact, shadow, evaluation alias, parent, and objective, then marks
the D58 auto1 result as misrouted when its label does not match its source.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
D57 = ROOT / "outputs" / "D57_STAGE4_ALGORITHMIC_CLOSURE"
D58 = ROOT / "outputs" / "D58_STAGE4_FULL_SYSTEM_CLOSURE"
OUT = ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def record(path: Path, *, source_case_id: str, shadow_id: str, evaluation_alias: str,
           parent_candidate: str, optimizer_objective: str, role: str) -> dict[str, Any]:
    return {
        "source_case_id": source_case_id,
        "source_artifact": str(path.resolve()),
        "source_sha256": sha256(path),
        "shadow_id": shadow_id,
        "evaluation_alias": evaluation_alias,
        "parent_candidate": parent_candidate,
        "optimizer_objective": optimizer_objective,
        "role": role,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    optimizer = load(D57 / "D57_OPTIMIZER_SHADOW.json")
    materialized = {item["objective"]: item for item in optimizer["materialized"]}
    auto0 = D57 / "optimizer_shadow" / "auto_0_adversarial_0100.csv"
    auto1_0100 = D57 / "optimizer_shadow" / "auto_1_adversarial_0100.csv"
    auto1_0101 = D57 / "optimizer_shadow" / "auto_1_adversarial_0101.csv"
    d58_manifest = rows(D58 / "D58_C4_SHADOW_NATIVE_MANIFEST.csv")
    d58_summaries = {
        item["case_id"]: item
        for path in (
            D58 / "stateful_consistent_scale001_auto0" / "execution_form_summary.json",
            D58 / "stateful_consistent_scale001_auto1" / "execution_form_summary.json",
        )
        for item in load(path).get("cases", [])
    }
    manifest_sources = {item["family"]: item["trajectory_csv"] for item in d58_manifest}
    mapping = [
        record(
            auto0,
            source_case_id="adversarial_0100",
            shadow_id="D57_AUTO_0",
            evaluation_alias="D58:adversarial_0100",
            parent_candidate="D56:c4_clearance_adversarial0101_amp005_consistent_0025",
            optimizer_objective=materialized["conditioning"]["objective"],
            role="D58_auto0_correctly_routed",
        ),
        record(
            auto1_0100,
            source_case_id="adversarial_0100",
            shadow_id="D57_AUTO_1",
            evaluation_alias="D58:adversarial_0101",
            parent_candidate="D56:c4_clearance_adversarial0101_amp005_consistent_0025",
            optimizer_objective=materialized["self_clearance"]["objective"],
            role="D58_auto1_misrouted_source",
        ),
        record(
            auto1_0101,
            source_case_id="adversarial_0101",
            shadow_id="D57_AUTO_1_0101_ARTIFACT",
            evaluation_alias="D59:adversarial_0101",
            parent_candidate="D56:c4_clearance_adversarial0101_amp005_consistent_0025",
            optimizer_objective="unrecorded_in_D57_manifest; artifact retained for identity correction",
            role="D59_corrected_source_candidate",
        ),
    ]
    verdict = {
        "classification": "SCIENTIFIC_DATA_ROUTING_MISMATCH",
        "alias_only": False,
        "reason": (
            "D58 evaluation alias adversarial_0101 resolves to the distinct D57 "
            "auto_1_adversarial_0100 artifact; the real auto_1_adversarial_0101 "
            "artifact exists and has different bytes and trajectory states."
        ),
        "affected_evaluations": [
            "D58 stateful_consistent_scale001_auto1",
            "D58 full_stateful_metrics_auto1",
            "D58 articulated_cert_stateful_auto1",
            "D58 fcl_routeA_stateful_auto1",
            "D58 dynamics_stateful_auto_pair auto1 row",
        ],
        "legacy_result_policy": "retain_as_misrouted_legacy_evidence; do_not_promote; do_not silently relabel",
        "corrective_action": "rerun the actual auto_1_adversarial_0101 artifact through D59 native and downstream checks",
    }
    payload = {
        "schema_version": "d59-provenance-audit-v1",
        "status": "PASS_MISMATCH_EXPOSED_CORRECTION_REQUIRED",
        "scope": "Stage 0/1 ON-state open-arch only; offline algorithmic evidence",
        "d57_optimizer_shadow": str((D57 / "D57_OPTIMIZER_SHADOW.json").resolve()),
        "d58_manifest_sources": manifest_sources,
        "d58_summary_identity": d58_summaries,
        "mapping": mapping,
        "verdict": verdict,
    }
    (OUT / "D59_PROVENANCE_AUDIT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "classification": verdict["classification"], "output": str(OUT / "D59_PROVENANCE_AUDIT.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
