"""Freeze and verify the small D59-D64 authority surface.

The repository deliberately ignores most generated output trees.  That is a
useful default for large experiment data, but it must not make the few files
which define the current scientific state invisible to ``git status`` or to a
regression check.  This module therefore keeps one explicit, non-recursive
manifest for the canonical identity, protected floor, lineage status/report
files, and the current root claim-fence documents.

The manifest is a seal, not a repository inventory: only the paths listed in
``FROZEN_ARTIFACT_SPECS`` are read or hashed.  The manifest file itself is
intentionally excluded from its own records so that it can be regenerated
deterministically after an authorized state change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D64_PROJECT_FREEZE"
MANIFEST_NAME = "D64_FROZEN_ARTIFACT_MANIFEST.json"
MANIFEST_RELATIVE_PATH = f"outputs/D64_PROJECT_FREEZE/{MANIFEST_NAME}"

D56 = ROOT / "outputs" / "D56_STAGE4B_SOFTWARE_CLOSURE" / "D56_FINAL_STATUS.json"
D56_REPORT = ROOT / "outputs" / "D56_STAGE4B_SOFTWARE_CLOSURE" / "D56_FINAL_REPORT.md"
D59 = ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE" / "D59_FINAL_STATUS.json"
D59_REPORT = ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE" / "D59_FINAL_REPORT.md"
D60 = ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "D60_FINAL_STATUS.json"
D60_REPORT = ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "D60_FINAL_REPORT.md"
D61 = ROOT / "outputs" / "D61_CONTINUOUS_CERTIFICATE_SHADOW" / "D61_FINAL_STATUS.json"
D61_REPORT = ROOT / "outputs" / "D61_CONTINUOUS_CERTIFICATE_SHADOW" / "D61_FINAL_REPORT.md"
D62 = ROOT / "outputs" / "D62_CONTINUOUS_CERTIFICATE_SHADOW" / "D62_FINAL_STATUS.json"
D62_REPORT = ROOT / "outputs" / "D62_CONTINUOUS_CERTIFICATE_SHADOW" / "D62_FINAL_REPORT.md"
D63 = ROOT / "outputs" / "D63_FINAL_SYSTEM_VALIDATION" / "D63_FINAL_STATUS.json"
D64 = OUT / "D64_FREEZE_STATUS.json"
D64_REPORT = OUT / "D64_FREEZE_REPORT.md"
CANONICAL_RECORD = ROOT / "outputs" / "D47_STAGE4B_ROLLING_CHAMPION_V1" / "PROMOTED" / "B3" / "PROMOTION_RECORD.json"
CANONICAL_METHOD = ROOT / "outputs" / "D47_STAGE4B_ROLLING_CHAMPION_V1" / "PROMOTED" / "B3" / "method.json"


# Keep this list deliberately short and explicit.  Do not replace it with a
# recursive directory walk: large trajectories and disposable logs are not
# authority for the freeze decision.
FROZEN_ARTIFACT_SPECS: tuple[dict[str, str], ...] = (
    {"role": "canonical_identity", "path": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3/PROMOTION_RECORD.json"},
    {"role": "canonical_method", "path": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3/method.json"},
    {"role": "protected_floor_status", "path": "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/D56_FINAL_STATUS.json"},
    {"role": "protected_floor_report", "path": "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/D56_FINAL_REPORT.md"},
    {"role": "d59_status", "path": "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/D59_FINAL_STATUS.json"},
    {"role": "d59_report", "path": "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/D59_FINAL_REPORT.md"},
    {"role": "d60_status", "path": "outputs/D60_SYSTEM_LEVEL_CLOSURE/D60_FINAL_STATUS.json"},
    {"role": "d60_report", "path": "outputs/D60_SYSTEM_LEVEL_CLOSURE/D60_FINAL_REPORT.md"},
    {"role": "d61_status", "path": "outputs/D61_CONTINUOUS_CERTIFICATE_SHADOW/D61_FINAL_STATUS.json"},
    {"role": "d61_report", "path": "outputs/D61_CONTINUOUS_CERTIFICATE_SHADOW/D61_FINAL_REPORT.md"},
    {"role": "d62_status", "path": "outputs/D62_CONTINUOUS_CERTIFICATE_SHADOW/D62_FINAL_STATUS.json"},
    {"role": "d62_report", "path": "outputs/D62_CONTINUOUS_CERTIFICATE_SHADOW/D62_FINAL_REPORT.md"},
    {"role": "d63_status", "path": "outputs/D63_FINAL_SYSTEM_VALIDATION/D63_FINAL_STATUS.json"},
    {"role": "d64_status", "path": "outputs/D64_PROJECT_FREEZE/D64_FREEZE_STATUS.json"},
    {"role": "d64_report", "path": "outputs/D64_PROJECT_FREEZE/D64_FREEZE_REPORT.md"},
    {"role": "current_goal_claim_fence", "path": "ROOT_GOAL.md"},
    {"role": "verified_facts_claim_fence", "path": "VERIFIED_FACTS.md"},
    {"role": "decision_ledger_claim_fence", "path": "DECISION_LEDGER.md"},
    {"role": "freeze_checker", "path": "tools/d64_freeze_project.py"},
)
FROZEN_ARTIFACTS: tuple[str, ...] = tuple(item["path"] for item in FROZEN_ARTIFACT_SPECS)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _specs(specs: Iterable[Mapping[str, str]] | None = None) -> tuple[dict[str, str], ...]:
    chosen = tuple(dict(item) for item in (FROZEN_ARTIFACT_SPECS if specs is None else specs))
    for item in chosen:
        if set(item) != {"role", "path"} or not item["role"] or not item["path"]:
            raise ValueError(f"invalid frozen-artifact spec: {item!r}")
        candidate = Path(item["path"])
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"frozen-artifact path must be relative and contained: {item['path']!r}")
        if item["path"].replace("\\", "/") == MANIFEST_RELATIVE_PATH:
            raise ValueError("the manifest cannot be included in its own records")
    paths = [item["path"].replace("\\", "/") for item in chosen]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate frozen-artifact path")
    return chosen


def sha256_file(path: Path) -> str:
    """Hash one explicitly listed regular file, in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_artifacts(root: Path = ROOT, specs: Iterable[Mapping[str, str]] | None = None) -> list[dict[str, Any]]:
    """Return records for the explicit authority list only."""

    root = root.resolve()
    records: list[dict[str, Any]] = []
    for spec in _specs(specs):
        relative = spec["path"].replace("\\", "/")
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"frozen-artifact escapes repository root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"frozen-artifact missing: {relative}")
        records.append(
            {
                "role": spec["role"],
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def manifest_payload(root: Path = ROOT, specs: Iterable[Mapping[str, str]] | None = None) -> dict[str, Any]:
    chosen = _specs(specs)
    return {
        "schema_version": "d64-frozen-artifact-manifest-v1",
        "manifest_path": MANIFEST_RELATIVE_PATH,
        "digest_algorithm": "sha256",
        "self_excluded": True,
        "scope": "explicit authority files only; no recursive repository or output-tree hashing",
        "artifact_count": len(chosen),
        "artifacts": collect_artifacts(root, chosen),
    }


def write_manifest(
    root: Path = ROOT,
    manifest_path: Path | None = None,
    specs: Iterable[Mapping[str, str]] | None = None,
) -> Path:
    """Write the deterministic authority manifest and return its path."""

    root = root.resolve()
    destination = (manifest_path or (root / MANIFEST_RELATIVE_PATH)).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest_payload(root, specs), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def verify_manifest(
    root: Path = ROOT,
    manifest_path: Path | None = None,
    specs: Iterable[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify exactly the listed authority files and reject scope drift."""

    root = root.resolve()
    path = (manifest_path or (root / MANIFEST_RELATIVE_PATH)).resolve()
    result: dict[str, Any] = {
        "status": "FAIL",
        "manifest_path": _relative_path(root, path) if path.is_relative_to(root) else str(path),
        "checked_count": 0,
        "mismatches": [],
    }
    try:
        manifest = load(path)
        chosen = _specs(specs)
        expected_paths = [item["path"].replace("\\", "/") for item in chosen]
        actual_records = manifest.get("artifacts")
        if manifest.get("schema_version") != "d64-frozen-artifact-manifest-v1":
            result["mismatches"].append({"kind": "schema", "expected": "d64-frozen-artifact-manifest-v1", "actual": manifest.get("schema_version")})
        if manifest.get("manifest_path") != MANIFEST_RELATIVE_PATH:
            result["mismatches"].append({"kind": "manifest_path", "expected": MANIFEST_RELATIVE_PATH, "actual": manifest.get("manifest_path")})
        if manifest.get("digest_algorithm") != "sha256":
            result["mismatches"].append({"kind": "digest_algorithm", "expected": "sha256", "actual": manifest.get("digest_algorithm")})
        if manifest.get("self_excluded") is not True:
            result["mismatches"].append({"kind": "self_excluded", "expected": True, "actual": manifest.get("self_excluded")})
        if not isinstance(actual_records, list):
            result["mismatches"].append({"kind": "records", "expected": "list", "actual": type(actual_records).__name__})
            actual_records = []
        actual_paths = [str(item.get("path", "")).replace("\\", "/") for item in actual_records if isinstance(item, dict)]
        if actual_paths != expected_paths:
            result["mismatches"].append({"kind": "scope", "expected": expected_paths, "actual": actual_paths})
        if manifest.get("artifact_count") != len(actual_records):
            result["mismatches"].append({"kind": "artifact_count", "expected": len(actual_records), "actual": manifest.get("artifact_count")})
        expected_by_path = {item["path"].replace("\\", "/"): item for item in actual_records if isinstance(item, dict) and "path" in item}
        result["checked_count"] = len(expected_by_path)
        for spec in chosen:
            relative = spec["path"].replace("\\", "/")
            record = expected_by_path.get(relative)
            if record is None:
                continue
            if record.get("role") != spec["role"]:
                result["mismatches"].append({"kind": "role", "path": relative, "expected": spec["role"], "actual": record.get("role")})
            file_path = (root / Path(relative)).resolve()
            if not file_path.is_file():
                result["mismatches"].append({"kind": "missing", "path": relative})
                continue
            actual_size = file_path.stat().st_size
            actual_sha = sha256_file(file_path)
            if record.get("size_bytes") != actual_size:
                result["mismatches"].append({"kind": "size", "path": relative, "expected": record.get("size_bytes"), "actual": actual_size})
            if record.get("sha256") != actual_sha:
                result["mismatches"].append({"kind": "digest", "path": relative, "expected": record.get("sha256"), "actual": actual_sha})
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        result["mismatches"].append({"kind": "manifest_error", "error": str(error)})
    result["status"] = "PASS" if not result["mismatches"] else "FAIL"
    return result


# Small compatibility aliases make the checker convenient to call from shell
# probes without introducing a second implementation or a second manifest.
build_manifest = manifest_payload
check_manifest = verify_manifest
verify = verify_manifest


def git_status_for_authority(root: Path = ROOT, specs: Iterable[Mapping[str, str]] | None = None) -> dict[str, Any]:
    """Report ordinary and ignored status for the explicit list.

    A clean result is impossible when an authority file is ignored.  This is
    intentionally separate from the digest check: a dirty/untracked checkout
    is not evidence that the file contents changed, while an ignored file must
    never be reported as clean merely because Git cannot see it.
    """

    root = root.resolve()
    paths = [item["path"].replace("\\", "/") for item in _specs(specs)]
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all", "--", *paths],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in status.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "clean": False, "status_lines": [], "ignored_paths": [], "error": str(error)}

    ignored: list[str] = []
    for relative in paths:
        try:
            probe = subprocess.run(
                ["git", "check-ignore", "-q", "--no-index", "--", relative],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            ignored.append(relative)
            continue
        if probe.returncode == 0:
            ignored.append(relative)
    return {
        "available": True,
        "clean": not lines and not ignored,
        "status_lines": lines,
        "ignored_paths": ignored,
    }


def _d62_summary(d62: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    unresolved: dict[str, dict[str, Any]] = {}
    for case, value in d62.get("cases", {}).items():
        if not isinstance(value, dict):
            continue
        unresolved[str(case)] = {
            "certified_intervals": value.get("certified_intervals"),
            "requested_intervals": value.get("requested_intervals"),
            "unresolved_intervals": int(value.get("unresolved_intervals", -1)),
            "collision_intervals": value.get("collision_intervals"),
            "resource_limit_reached": value.get("resource_limit_reached"),
            "status": value.get("status"),
        }
    return unresolved


def build_lineage(d59: Mapping[str, Any], d60: Mapping[str, Any], d61: Mapping[str, Any], d62: Mapping[str, Any], d63: Mapping[str, Any]) -> dict[str, Any]:
    """Create a compact, machine-readable historical claim fence."""

    d62_cases = _d62_summary(d62)
    return {
        "current_authority": "D64_PROJECT_FREEZE",
        "reader_guard": "D59 zero-collision/zero-unresolved observations are historical and scoped to its conservative model; they are not the current complete exact articulated self-CCD proof.",
        "D59": {
            "state": "HISTORICAL_SCOPED_SUPERSEDED_AS_CURRENT_CONTINUOUS_CCD_AUTHORITY",
            "historical_observation": {"collision_regions": 0, "unresolved_regions": 0, "candidate_count": 2},
            "scope": "two correctly routed finalists under the accepted conservative FK-aware software model",
            "exact_external_articulated_fk_qt_ccd": "UNAVAILABLE_UNVERIFIED",
            "superseded_by": ["D61", "D62", "D63", "D64"],
            "source_task_status": d59.get("D59_TASK_STATUS"),
        },
        "D60": {
            "state": "HISTORICAL_SCOPED_MODEL_RESULT_SUPERSEDED_BY_LATER_COVERAGE_ACCOUNTING",
            "scope": "D60 conservative FK(q(t)) model and endpoint-FCL cross-check; not exact external articulated CCD",
            "exact_external_articulated_fk_qt_ccd": "UNAVAILABLE_UNVERIFIED",
            "source_task_status": d60.get("D60_TASK_STATUS"),
        },
        "D61": {
            "state": "SHADOW_PARTIAL_COVERAGE_UNRESOLVED",
            "scope": "explicit FK(q(t)) interval certificate windows only",
            "exact_external_articulated_fk_qt_ccd": "UNAVAILABLE_UNVERIFIED",
            "source_task_status": d61.get("task_status"),
        },
        "D62": {
            "state": "CURRENT_MEASURED_SHADOW_UNRESOLVED",
            "scope": "full native interval evaluation with sound triangle-preserving BVH/OBB coarse enclosure",
            "cases": d62_cases,
            "source_task_status": d62.get("D62_TASK_STATUS"),
        },
        "D63": {
            "state": "SYSTEM_VALIDATION_PARTIAL_SAFETY_UNRESOLVED",
            "scope": "software evidence-chain validation only",
            "safety_status": d63.get("safety", {}).get("status") if isinstance(d63.get("safety"), dict) else None,
            "source_status": d63.get("status"),
        },
        "D64": {
            "state": "FROZEN_WITH_EXPLICIT_P0_BOUNDARY",
            "scope": "measured software evidence and explicit nonclaims",
            "promotion": "NO_PROMOTION",
        },
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_freeze_outputs() -> dict[str, Any]:
    """Materialize D64 status/report, then seal the explicit authority list."""

    d59 = load(D59)
    d60 = load(D60)
    d61 = load(D61)
    d62 = load(D62)
    d63 = load(D63)
    coverage = _d62_summary(d62)
    unresolved_values = [item["unresolved_intervals"] for item in coverage.values()]
    freeze_status = "SOFTWARE_SCOPE_FROZEN_WITH_EXPLICIT_P0_BOUNDARY" if any(value > 0 for value in unresolved_values) else "SOFTWARE_SCOPE_FROZEN"
    lineage = build_lineage(d59, d60, d61, d62, d63)
    git_state = git_status_for_authority()
    result: dict[str, Any] = {
        "schema_version": "d64-project-freeze-v2",
        "D64_STATUS": freeze_status,
        "freeze_authority": "D59/D60/D61/D62/D63 measured software evidence and explicit unavailable domains",
        "claim_scope": "software evidence-chain freeze only; no exact external articulated FK(q(t)) self-CCD or physical safety claim",
        "lineage": lineage,
        "canonical": d59.get("CURRENT_CANONICAL"),
        "protected_floor": d59.get("CURRENT_PROTECTED_FLOOR"),
        "promotion": "NO_PROMOTION",
        "protected_paths_clean": git_state["clean"],
        "protected_paths_git_status": git_state,
        "protected_artifacts_verified": "PENDING_MANIFEST_WRITE",
        "stage3_and_canonical_mutated": False,
        "d59_status": d59.get("D59_TASK_STATUS"),
        "d60_status": d60.get("D60_TASK_STATUS"),
        "d61_status": d61.get("task_status"),
        "d62_status": d62.get("D62_TASK_STATUS"),
        "d62_coverage": coverage,
        "d63_status": d63.get("status"),
        "frozen_measured_claims": [
            "native trajectory motion/continuity/velocity/acceleration and Profile.j evidence",
            "model-only Pinocchio dynamics and bounded robustness evidence",
            "deterministic replay and reproducibility evidence",
            "FCL rigid endpoint sweep and MoveIt robot-world segment cross-checks",
            "D62 full interval evaluation with sound triangle-preserving BVH/OBB coarse certification",
        ],
        "frozen_nonclaims": [
            "D59/D60 zero-collision observations are scoped conservative-model observations, not exact articulated FK(q(t)) continuous self-CCD",
            "D62 unresolved intervals are not continuous-certificate PASS",
            "no exact external nonlinear articulated CCD backend is claimed",
            "minimum physical clearance is unavailable; projection gap is not clearance",
            "no hardware torque/current/thermal/calibration/tracking safety claim",
        ],
        "remaining_p0": d62.get("remaining_p0", []),
        "evidence": {
            "d59": _relative_path(ROOT, D59),
            "d60": _relative_path(ROOT, D60),
            "d61": _relative_path(ROOT, D61),
            "d62": _relative_path(ROOT, D62),
            "d63": _relative_path(ROOT, D63),
            "frozen_artifact_manifest": MANIFEST_RELATIVE_PATH,
            "protected_status_command": "git status --short --untracked-files=all -- <explicit frozen-artifact paths>",
        },
        "post_freeze_policy": "stop optimization; proceed only with paper organization, reproduction, or a separately authorized Stage 4B/backend implementation that cannot rewrite this frozen evidence",
    }
    _write_text(D64, json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# D64 Project Freeze",
        "",
        f"Status: `{freeze_status}`",
        "",
        "D64 freezes the measured software scope, its reproducibility evidence, and its explicit nonclaims. It does not promote the D62 shadow and does not convert unresolved intervals or unavailable physical quantities into safety claims.",
        "",
        "## Claim fence and D59-D64 lineage",
        "",
        "The D59 zero-collision/zero-unresolved observation is retained as a historical fact, but it is scoped to two correctly routed candidates under D59's conservative FK-aware software model. It is superseded as the current continuous-collision authority by the later D61-D64 chain and must not be read as a complete exact external articulated `FK(q(t))` self-CCD proof.",
        "",
        "- `D59`: historical scoped model observation; exact external articulated `FK(q(t))` self-CCD is `UNAVAILABLE_UNVERIFIED`.",
        "- `D60`: historical conservative FK(q(t)) model and endpoint-FCL cross-check; not exact external articulated CCD.",
        "- `D61`: explicit FK(q(t)) shadow with partial windows and unresolved coverage.",
        "- `D62`: all native intervals evaluated, but unresolved intervals remain (therefore not a continuous-certificate PASS).",
        "- `D63`: software evidence-chain validation; safety status remains `UNRESOLVED`.",
        "- `D64`: freezes the measured evidence and the above boundaries; promotion remains `NO_PROMOTION`.",
        "",
        "## Frozen floor",
        "",
        f"- Canonical: `{d59.get('CURRENT_CANONICAL')}`",
        f"- Protected floor: `{d59.get('CURRENT_PROTECTED_FLOOR')}`",
        "- Promotion: `NO_PROMOTION`",
        f"- Protected-path git status: `{'CLEAN' if git_state['clean'] else 'NOT_CLEAN_OR_UNTRACKED'}` (digest verification is separate).",
        "",
        "## Frozen D62 boundary",
        "",
    ]
    for case, value in coverage.items():
        lines.append(f"- `{case}`: {value.get('certified_intervals')} certified / {value.get('requested_intervals')} evaluated, {value.get('unresolved_intervals')} unresolved, {value.get('collision_intervals')} collision intervals, resource hit `{value.get('resource_limit_reached')}`.")
    lines.extend(
        [
            "",
            "The coarse route is sound as an outer-enclosure proof search and full interval coverage was evaluated. The unresolved set remains an explicit P0 boundary. The attempted full unresolved-set depth-2 refinement was interrupted before producing a valid result under the Python reference budget; it is not part of the scientific pass/fail result.",
            "",
            "## Frozen-artifact manifest",
            "",
            f"`{MANIFEST_RELATIVE_PATH}` is the single authority seal for {len(FROZEN_ARTIFACT_SPECS)} explicitly listed files (canonical identity, protected floor, D59-D64 state/reports, root claim-fence docs, and this checker). It records each path, byte size, and SHA-256 digest; it does not hash the repository or generated output trees recursively.",
            "",
            "## Next authorized work",
            "",
            "Stop optimization in this frozen scope. Paper preparation and reproduction are allowed. Any future exact articulated CCD or compiled interval/BVH backend must be a separately authorized Stage 4B shadow and must reproduce the known-answer, fail-closed, and protected regression contracts before it can affect the evidence chain.",
        ]
    )
    _write_text(D64_REPORT, "\n".join(lines) + "\n")
    manifest_path = write_manifest()
    manifest_check = verify_manifest()
    result["protected_artifacts_verified"] = manifest_check["status"]
    # Update the status once with the post-write verification result.  The
    # status is itself listed in the manifest, so seal it again after this
    # small, deterministic field update.
    _write_text(D64, json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest_path = write_manifest()
    manifest_check = verify_manifest()
    if manifest_check["status"] != "PASS":
        raise RuntimeError(f"D64 frozen-artifact manifest failed immediately after write: {manifest_check}")
    return {"status": freeze_status, "manifest": str(manifest_path.resolve()), "manifest_check": manifest_check}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("write", "check", "verify"), help="optional operation name; default is write")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="materialize D64 status/report and refresh the explicit manifest")
    mode.add_argument("--check", "--verify", dest="check", action="store_true", help="verify the existing explicit manifest without writing files")
    args = parser.parse_args(argv)
    try:
        if args.check or args.command in {"check", "verify"}:
            result = verify_manifest()
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "PASS" else 2
        result = write_freeze_outputs()
    except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
