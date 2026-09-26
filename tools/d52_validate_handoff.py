"""Validate the D52 anti-drift handoff staging tree."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED = (
    "READ_FIRST.txt", "ROOT_GOAL.md", "CURRENT_MISSION.md", "ACCEPTANCE_GATES.md",
    "AUTHORITY_MAP.md", "STATE_TRACKER.json", "AUTHORITATIVE_STATE.md",
    "VERIFIED_FACTS.md", "OPEN_ITEMS.md", "FAILED_ATTEMPTS.md", "DECISIONS.md",
    "TAKEOVER_PROTOCOL.md", "HANDOFF.md", "USER_PROMPTS.md",
    "PROMPT_SUPERSESSION_MAP.md", "HANDOFF_MANIFEST.json",
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|cookie|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    stage = args.stage.resolve()
    missing = [path for path in REQUIRED if not (stage / path).is_file()]
    json_errors: list[str] = []
    for path in stage.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            json_errors.append(f"{path.relative_to(stage)}: {exc}")
    tracker = json.loads((stage / "STATE_TRACKER.json").read_text(encoding="utf-8"))
    metrics = json.loads((stage / "results/final_metrics/D52_FINAL_METRICS.json").read_text(encoding="utf-8"))
    evidence_paths: list[str] = []
    for unit in tracker.get("work_units", {}).values():
        evidence_paths.extend(unit.get("evidence_paths", []))
        if unit.get("verifier_reference"):
            evidence_paths.append(unit["verifier_reference"])
    missing_evidence = [path for path in evidence_paths if not (stage / path).is_file()]
    secret_hits: list[str] = []
    for path in stage.rglob("*"):
        if not path.is_file() or path.name == "consistency_audit.json":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secret_hits.append(str(path.relative_to(stage)))
    best = metrics["final_best_candidate"]["name"]
    checks = {
        "required_files": not missing,
        "all_json_parse": not json_errors,
        "state_tracker_evidence_paths": not missing_evidence,
        "best_candidate_consistent": best == "stateful_local_g1_w05_0025",
        "canonical_promotion_false": metrics.get("canonical_promotion") == "NO" and tracker["canonical_state"]["promotion_occurred"] is False,
        "task_status_consistent": metrics.get("task_status") == "PASS_SOFTWARE_CLOSED_EXTERNAL_VALIDATION_REQUIRED",
        "secret_scan_clean": not secret_hits,
        "no_private_reasoning_claim": "hidden chain-of-thought" not in (stage / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8").lower(),
    }
    result = {
        "schema_version": "d52-anti-drift-consistency-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_required_files": missing,
        "json_errors": json_errors,
        "missing_evidence_paths": missing_evidence,
        "secret_hits": secret_hits,
        "file_count": sum(1 for path in stage.rglob("*") if path.is_file()),
    }
    output = stage / "evidence/authority/consistency_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
