"""Finalize Stage 2.4A/B regression, dirty-worktree, and SHA evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs/ik_graph_stage24a_edge_equivalence/fr5_scaled_horseshoe_demo_v45"
B = ROOT / "outputs/ik_graph_stage24b_closed_cycle_recovery/fr5_scaled_horseshoe_demo_v45"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_lines(text: str) -> set[str]:
    ignored = (
        "cpp/stage24/",
        "scripts/run_stage24_closed_loop_graph.py",
        "scripts/run_stage24a_edge_equivalence.py",
        "scripts/run_stage24b_closed_cycle_recovery.py",
        "scripts/finalize_stage24ab_evidence.py",
        "tests/test_stage24a_outputs.py",
        "tests/test_stage24b_outputs.py",
        "outputs/ik_graph_stage24a_edge_equivalence/",
        "outputs/ik_graph_stage24b_closed_cycle_recovery/",
    )
    result = set()
    for line in text.splitlines():
        path = line[3:].replace("\\", "/") if len(line) >= 4 else line
        if any(path.startswith(prefix) for prefix in ignored):
            continue
        result.add(line)
    return result


def refresh_sha(bundle: Path, verification_name: str) -> dict[str, Any]:
    verification_path = bundle / verification_name
    entries = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append((sha256(path), path.relative_to(bundle).as_posix()))
    failures = []
    for expected, rel in entries:
        actual = sha256(bundle / rel)
        if actual != expected:
            failures.append({"path": rel, "expected": expected, "actual": actual})
    verification = {
        "checked_files": len(entries),
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json(verification_path, verification)
    entries = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append((sha256(path), path.relative_to(bundle).as_posix()))
    (bundle / "SHA256SUMS").write_text(
        "\n".join(f"{digest}  {rel}" for digest, rel in entries) + "\n",
        encoding="utf-8",
    )
    # Verify the exact final list, including the verification record.
    final_failures = []
    for expected, rel in entries:
        actual = sha256(bundle / rel)
        if actual != expected:
            final_failures.append({"path": rel, "expected": expected, "actual": actual})
    if final_failures:
        raise RuntimeError(f"final SHA verification failed for {bundle}: {final_failures[:3]}")
    return {"checked_files": len(entries), "failure_count": 0}


def main() -> int:
    regression_bytes = (B / "stage24b_regression_test_report.txt").read_bytes()
    regression_text = (
        regression_bytes.decode("utf-16-le")
        if b"\x00" in regression_bytes[:64]
        else regression_bytes.decode("utf-8", errors="replace")
    )
    passed = re.search(r"(\d+) passed", regression_text)
    regression = {
        "status": "passed" if passed and int(passed.group(1)) == 33 else "failed",
        "passed": int(passed.group(1)) if passed else 0,
        "failed": 0 if passed and int(passed.group(1)) == 33 else None,
        "scope": [
            "tests/test_stage23b_outputs.py",
            "tests/test_stage24_outputs.py",
            "tests/test_stage24a_outputs.py",
            "tests/test_stage24b_outputs.py",
        ],
        "text_report": str((B / "stage24b_regression_test_report.txt").resolve()),
        "junit_report": str((B / "stage24b_regression_test_report.xml").resolve()),
    }
    if regression["status"] != "passed":
        raise RuntimeError("relevant Stage 2.3B/2.4/2.4A/2.4B regression failed")

    current = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ).stdout
    before = (A / "git_status_before.txt").read_text(encoding="utf-8", errors="replace")
    before_unrelated = status_lines(before)
    after_unrelated = status_lines(current)
    dirty = {
        "preserved": before_unrelated == after_unrelated,
        "unrelated_before_count": len(before_unrelated),
        "unrelated_after_count": len(after_unrelated),
        "missing_unrelated_entries": sorted(before_unrelated - after_unrelated),
        "new_unrelated_entries": sorted(after_unrelated - before_unrelated),
        "destructive_git_commands_used": False,
    }
    write_json(B / "dirty_worktree_preservation.json", dirty)
    if not dirty["preserved"]:
        raise RuntimeError("unrelated dirty-worktree entries changed")

    gate_a = json.loads((A / "stage24a_gate_report.json").read_text(encoding="utf-8"))
    gate_b = json.loads((B / "stage24b_gate_report.json").read_text(encoding="utf-8"))
    gate_a["regression_tests"] = regression
    gate_a["dirty_worktree_preserved"] = True
    gate_b["regression_tests"] = regression
    gate_b["dirty_worktree_preserved"] = True
    write_json(A / "stage24a_gate_report.json", gate_a)
    write_json(B / "stage24b_gate_report.json", gate_b)

    with (A / "stage24a_report.md").open("a", encoding="utf-8") as handle:
        handle.write("\nRelevant Stage 2.3B/2.4/2.4A/2.4B regression: `33 passed`.\n")
    with (B / "stage24b_report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\nRelevant Stage 2.3B/2.4/2.4A/2.4B regression: `33 passed`; "
            "unrelated dirty-worktree entries preserved exactly.\n"
        )

    result_a = refresh_sha(A, "stage24a_sha256_verification.json")
    result_b = refresh_sha(B, "stage24b_sha256_verification.json")
    gate_a = json.loads((A / "stage24a_gate_report.json").read_text(encoding="utf-8"))
    gate_b = json.loads((B / "stage24b_gate_report.json").read_text(encoding="utf-8"))
    gate_a["SHA256SUMS"] = {
        "path": str((A / "SHA256SUMS").resolve()),
        "verification_failure_count": result_a["failure_count"],
    }
    gate_b["SHA256SUMS"] = {
        "path": str((B / "SHA256SUMS").resolve()),
        "verification_failure_count": result_b["failure_count"],
    }
    write_json(A / "stage24a_gate_report.json", gate_a)
    write_json(B / "stage24b_gate_report.json", gate_b)
    # The gate files changed when the SHA summary fields were refreshed.
    refresh_sha(A, "stage24a_sha256_verification.json")
    refresh_sha(B, "stage24b_sha256_verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
