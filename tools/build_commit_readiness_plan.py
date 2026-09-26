from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


PRIMARY_COMMIT_GROUPS = {
    "runtime_bridge_and_strict_gates",
    "tests",
    "documentation",
    "formal_evidence",
}
REVIEW_SEPARATELY_GROUPS = {"review_separately_preexisting_or_mixed"}
LOCAL_OR_OPTIONAL_GROUPS = {"probe_evidence", "offline_diagnostic_artifacts"}
STAGE_COMMAND_CHUNK_SIZE = 12


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _entries(groups: dict[str, list[dict[str, str]]], names: set[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for name in sorted(names):
        for entry in groups.get(name, []):
            result.append({"group": name, **entry})
    return result


def _quote_path(path: str) -> str:
    return '"' + path.replace('"', '\\"') + '"'


def _stage_commands(entries: list[dict[str, str]]) -> list[str]:
    paths = [entry["path"] for entry in entries]
    commands = []
    for start in range(0, len(paths), STAGE_COMMAND_CHUNK_SIZE):
        chunk = paths[start : start + STAGE_COMMAND_CHUNK_SIZE]
        if chunk:
            commands.append("git add -- " + " ".join(_quote_path(path) for path in chunk))
    return commands


def build_commit_plan(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    groups = manifest.get("changed_files", {})
    if not isinstance(groups, dict):
        groups = {}
    primary = _entries(groups, PRIMARY_COMMIT_GROUPS)
    review = _entries(groups, REVIEW_SEPARATELY_GROUPS)
    optional = _entries(groups, LOCAL_OR_OPTIONAL_GROUPS)
    unclassified = list(groups.get("unclassified", []))
    return {
        "schema_version": 1,
        "purpose": "Commit-readiness plan for the current strict-validation worktree.",
        "source_manifest": str(manifest_path),
        "overall_status": manifest.get("overall_status", "missing"),
        "strict_audit_status": manifest.get("strict_audit_status", "missing"),
        "recommendation": (
            "Do not label the commit as production validation pass. Commit the strict-gate code, tests, "
            "documentation, and formal fail evidence together; review mixed pre-existing files separately."
        ),
        "primary_commit": {
            "description": "Keep together: runtime fixes, strict gates, docs, tests, and authoritative fail/pass evidence.",
            "file_count": len(primary),
            "files": primary,
            "stage_commands": _stage_commands(primary),
            "commit_message_suggestion": "Record strict MoveIt fail evidence for placeholder TCP",
        },
        "review_separately": {
            "description": "Inspect before staging because these files were pre-existing or mixed with unrelated work.",
            "file_count": len(review),
            "files": review,
            "stage_commands": _stage_commands(review),
            "stage_policy": "Do not include in the primary commit until each mixed/pre-existing change is reviewed.",
        },
        "local_or_optional_evidence": {
            "description": "Diagnostic probe outputs and offline generated artifacts. Keep only if reviewers need reproducible forensic evidence.",
            "file_count": len(optional),
            "files": optional,
            "stage_commands": _stage_commands(optional),
            "stage_policy": "Stage only if the review needs full probe evidence; otherwise keep local or archive externally.",
        },
        "unclassified": {
            "file_count": len(unclassified),
            "files": unclassified,
        },
        "blocking_items": manifest.get("blocking_items", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a commit-readiness plan from the validation handoff manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/validation_handoff_manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/commit_readiness_plan.json")
    args = parser.parse_args()

    payload = build_commit_plan(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote commit readiness plan to {args.out}")


if __name__ == "__main__":
    main()
