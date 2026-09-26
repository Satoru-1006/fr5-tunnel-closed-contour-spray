"""Build and independently validate the compact D61 cross-chat handoff.

The archive is deliberately assembled from explicit authority/state files and
selected evidence.  It does not include the raw transcript, nested archives,
caches, or unrelated dirty-worktree material.  The final archive hash is
written outside the archive to avoid recursive self-reference.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\86198\Desktop")
STAGE_ROOT = ROOT / "tmp"
D61_OUT = ROOT / "outputs" / "D61_CONTINUOUS_CERTIFICATE_SHADOW"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_file(source: Path, destination: Path, copied: list[str]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(destination.as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and not name.startswith(("/", "\\"))


def secret_scan(root: Path) -> list[str]:
    patterns = (
        re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9+/=_-]{20,}"),
    )
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            if pattern.search(content):
                findings.append(str(path.relative_to(root)))
                break
    return findings


def build_stage(stamp: str) -> tuple[Path, list[str]]:
    stage = STAGE_ROOT / f"D61_CROSS_CHAT_HANDOFF_STAGING_{stamp}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    copied: list[str] = []

    for name in ("ROOT_GOAL.md", "HANDOFF.md", "CURRENT_STATE.md", "VERIFIED_FACTS.md", "DECISION_LEDGER.md", "OPEN_TASKS.md", "USER_PROMPTS.md"):
        copy_file(ROOT / name, stage / name, copied)

    write_text(
        stage / "AUTHORITY_ORDER.md",
        """# Authority Order\n\n"
        "The receiving Agent must use this order:\n\n"
        "1. New explicit user instruction.\n"
        "2. `ROOT_GOAL.md`.\n"
        "3. `CURRENT_STATE.md` plus repository and protected-state checks.\n"
        "4. `VERIFIED_FACTS.md` and machine-readable evidence.\n"
        "5. `DECISION_LEDGER.md`.\n"
        "6. `OPEN_TASKS.md`.\n"
        "7. `HANDOFF.md` and selected references.\n"
        "8. Raw history, if ever provided, as reference only.\n\n"
        "A packaged summary never overrides current repository evidence.\n"
        "Canonical, protected floor, verified champion, and shadow candidates\n"
        "remain separate concepts.\n""",
    )

    copy_file(ROOT / "state" / "progress.json", stage / "state" / "progress.json", copied)
    copy_file(ROOT / "state" / "verification.json", stage / "state" / "verification.json", copied)

    result_files = (
        (D61_OUT / "D61_FINAL_REPORT.md", "results/D61_FINAL_REPORT.md"),
        (D61_OUT / "D61_FINAL_STATUS.json", "results/D61_FINAL_STATUS.json"),
        (D61_OUT / "D61_PROBLEM_MAP.md", "results/D61_PROBLEM_MAP.md"),
        (D61_OUT / "D61_ROUTE_MATRIX.md", "results/D61_ROUTE_MATRIX.md"),
        (D61_OUT / "D61_STAGE4B_HANDOFF.md", "results/D61_STAGE4B_HANDOFF.md"),
        (ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "D60_FINAL_STATUS.json", "results/D60_FINAL_STATUS.json"),
        (ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "D60_FINAL_REPORT.md", "results/D60_FINAL_REPORT.md"),
    )
    for source, destination in result_files:
        copy_file(source, stage / destination, copied)

    changed_files = (
        (ROOT / "tools" / "d61_fk_interval_certificate.py", "changed_files/tools/d61_fk_interval_certificate.py"),
        (ROOT / "tools" / "build_d61_cross_chat_handoff.py", "changed_files/tools/build_d61_cross_chat_handoff.py"),
        (ROOT / "tests" / "test_d61_fk_interval_certificate.py", "changed_files/tests/test_d61_fk_interval_certificate.py"),
    )
    for source, destination in changed_files:
        copy_file(source, stage / destination, copied)

    evidence_files = (
        (D61_OUT / "auto0" / "D61_CERTIFICATE_final_window.json", "evidence/auto0/D61_CERTIFICATE_final_window.json"),
        (D61_OUT / "auto1" / "D61_CERTIFICATE_final_window.json", "evidence/auto1/D61_CERTIFICATE_final_window.json"),
        (ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "fk_qt_certificate" / "D60_FK_QT_CERTIFICATE_AUDIT.json", "evidence/D60_FK_QT_CERTIFICATE_AUDIT.json"),
        (ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "fk_qt_certificate" / "D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl", "evidence/D60_FULL_CERTIFICATE_CASES.jsonl"),
        (ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "semantic_probes" / "D60_SEMANTIC_PROBES.json", "evidence/D60_SEMANTIC_PROBES.json"),
    )
    for source, destination in evidence_files:
        copy_file(source, stage / destination, copied)

    write_text(
        stage / "logs" / "verification_summary.txt",
        """D61 verification summary\n"
        "========================\n"
        "python -m pytest -q tests/test_d61_fk_interval_certificate.py\n"
        "PASS: 8 passed\n\n"
        "python -m pytest -q tests/test_d50_measurement_repairs.py tests/test_d54_articulated_certificate.py tests/test_d57_system_closure.py tests/test_d61_fk_interval_certificate.py\n"
        "PASS: 20 passed\n\n"
        "D61 real windows\n"
        "auto0: 50/7004 intervals, 450 certified regions, 50 unresolved regions, 0 collision witnesses\n"
        "corrected auto1: 50/7029 intervals, 450 certified regions, 50 unresolved regions, 0 collision witnesses\n"
        "Both windows are UNRESOLVED and coverage-incomplete.\n\n"
        "Protected paths D47/D56/D57/D58/D59/D60: no status changes observed after D61.\n"
        "Raw transcript, nested archives, caches, and unrelated dirty-worktree material: excluded.\n""",
    )
    write_text(
        stage / "references" / "REFERENCES.md",
        """# References\n\n"
        "External documentation consulted for route semantics:\n\n"
        "- MoveIt Bullet CCD: https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html\n"
        "- Tesseract collision detection: https://tesseract-robotics.github.io/tesseract/collision.html\n"
        "- Tesseract Python collision API: https://tesseract-robotics.github.io/tesseract_python/_source/modules/tesseract_collision/tesseract_collision.html\n\n"
        "These sources document rigid/two-state or casted link-pose continuous\n"
        "operations. They are not treated as exact nonlinear articulated\n"
        "FK(q(t)) authority in this handoff.\n\n"
        "Local predecessor reference: `results/D60_FINAL_REPORT.md`.\n""",
    )

    manifest = {
        "schema_version": "d61-cross-chat-handoff-staging-v1",
        "stage": "D61",
        "created_at_local": stamp,
        "authority_first": True,
        "raw_transcript_included": False,
        "nested_archives_included": False,
        "copied_files": sorted(copied),
        "external_checksum": f"Desktop/D61_CROSS_CHAT_HANDOFF_{stamp}.sha256.txt",
    }
    write_text(stage / "state" / "archive_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return stage, copied


def make_archive(stage: Path, stamp: str) -> Path:
    DESKTOP.mkdir(parents=True, exist_ok=True)
    existing = sorted(DESKTOP.glob("D61_CROSS_CHAT_HANDOFF_*.tar.xz"))
    if existing:
        raise FileExistsError(f"formal D61 archive already exists: {existing}")
    archive = DESKTOP / f"D61_CROSS_CHAT_HANDOFF_{stamp}.tar.xz"
    with tarfile.open(archive, "w:xz", preset=9 | lzma.PRESET_EXTREME) as stream:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                stream.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
    return archive


def validate_archive(archive: Path, expected_required: set[str], stamp: str) -> dict[str, object]:
    size = archive.stat().st_size
    if size > 20 * 1024 * 1024:
        raise RuntimeError(f"archive exceeds 20 MB: {size}")
    with tarfile.open(archive, "r:xz") as stream:
        members = stream.getmembers()
        names = {member.name for member in members}
        if any(not safe_member_name(name) for name in names):
            raise RuntimeError("unsafe archive path")
        if any(name.lower().endswith((".tar", ".tar.gz", ".tar.xz", ".zip")) for name in names):
            raise RuntimeError("nested archive found")
        if not expected_required.issubset(names):
            raise RuntimeError(f"missing required archive members: {sorted(expected_required - names)}")
        if any(not member.isfile() for member in members):
            raise RuntimeError("non-regular archive member found")
        with tempfile.TemporaryDirectory(prefix="d61_handoff_extract_") as directory:
            target = Path(directory)
            stream.extractall(target)
            if not all((target / name).is_file() for name in expected_required):
                raise RuntimeError("required member missing after extraction")
            secret_findings = secret_scan(target)
            if secret_findings:
                raise RuntimeError(f"secret scan findings: {secret_findings}")
    checksum = sha256(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256.txt")
    write_text(checksum_path, f"{checksum}  {archive.name}\n")
    report = {
        "schema_version": "d61-cross-chat-handoff-validation-v1",
        "archive": str(archive),
        "archive_name": archive.name,
        "archive_sha256": checksum,
        "archive_size_bytes": size,
        "size_limit_bytes": 20 * 1024 * 1024,
        "xz_lzma2": True,
        "tar_readable": True,
        "member_count": len(names),
        "required_members_present": True,
        "safe_paths": True,
        "nested_archives": False,
        "extraction": "PASS",
        "secret_scan": "PASS",
        "formal_d61_archives_in_desktop": len(list(DESKTOP.glob("D61_CROSS_CHAT_HANDOFF_*.tar.xz"))),
        "external_checksum_file": str(checksum_path),
        "stamp": stamp,
    }
    report_path = D61_OUT / "D61_ARCHIVE_VALIDATION.json"
    write_text(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S+0800")
    stage, copied = build_stage(stamp)
    archive = make_archive(stage, stamp)
    required = {
        "ROOT_GOAL.md",
        "HANDOFF.md",
        "CURRENT_STATE.md",
        "VERIFIED_FACTS.md",
        "DECISION_LEDGER.md",
        "OPEN_TASKS.md",
        "USER_PROMPTS.md",
        "results/D61_FINAL_STATUS.json",
        "evidence/auto0/D61_CERTIFICATE_final_window.json",
        "evidence/auto1/D61_CERTIFICATE_final_window.json",
        "changed_files/tools/d61_fk_interval_certificate.py",
        "state/progress.json",
        "state/verification.json",
    }
    report = validate_archive(archive, required, stamp)
    print(json.dumps({"archive": str(archive), "stage": str(stage), "copied_file_count": len(copied), "validation": report}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
