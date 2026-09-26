"""Create the single compact D31 cross-chat handoff archive."""

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3_h13_post_d30_d31_dual_mode_direction_recovery_20260823T235000+0800"
D30 = ROOT / "outputs" / "stage3_h13_post_d29_d30_active_boundary_20260823T200000+0800"
STAGE = ROOT / "tmp" / "d31_cross_chat_handoff_stage_20260824T010100+0800"
ARCHIVE = Path(r"C:\Users\86198\Desktop\STAGE3_H13_D31_DUAL_MODE_DIRECTION_RECOVERY_CROSS_CHAT_HANDOFF_20260824T010100+0800.tar.xz")
FINAL_CHECKPOINT_SHA256 = "8e4bb2bc0b28d90f353fe9030ccbfd3a732460a5ce24884694796121add84d37"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: Path, relative: str) -> None:
    destination = STAGE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    if ARCHIVE.exists():
        raise RuntimeError(f"ARCHIVE_ALREADY_EXISTS:{ARCHIVE}")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    required = {
        "START_HERE_NEXT_CHAT.md": OUT / "START_HERE_NEXT_CHAT.md",
        "01_REPORT/FINAL_REPORT.md": OUT / "FINAL_REPORT.md",
        "01_REPORT/TASK_AND_RESULTS_SUMMARY.md": OUT / "TASK_AND_RESULTS_SUMMARY.md",
        "01_REPORT/STAGE3_COMPLETION_CONTRACT_EVALUATION.json": OUT / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json",
        "02_EXECUTION/D31_EXECUTION_MANIFEST.json": OUT / "D31_EXECUTION_MANIFEST.json",
        "02_EXECUTION/D31_CANONICAL_TRAJECTORY.json": OUT / "D31_CANONICAL_TRAJECTORY.json",
        "02_EXECUTION/D31_CANONICAL_TRAJECTORY.csv": OUT / "D31_CANONICAL_TRAJECTORY.csv",
        "02_EXECUTION/D31_CANDIDATE_RESULTS.json": OUT / "D31_CANDIDATE_RESULTS.json",
        "02_EXECUTION/D31_CANDIDATE_RESULTS.csv": OUT / "D31_CANDIDATE_RESULTS.csv",
        "02_EXECUTION/UPDATE_424_RECOVERY_COMPARISON.json": OUT / "UPDATE_424_RECOVERY_COMPARISON.json",
        "02_EXECUTION/UPDATE_424_MODE_A_REALIZED_GEOMETRY.json": OUT / "UPDATE_424_MODE_A_REALIZED_GEOMETRY.json",
        "02_EXECUTION/TARGETED_UPDATE_427_BOUNDARY_DIAGNOSTIC.json": OUT / "TARGETED_UPDATE_427_BOUNDARY_DIAGNOSTIC.json",
        "02_EXECUTION/OPTIMIZER_REPLAY_VALIDATION_SUMMARY.json": OUT / "OPTIMIZER_REPLAY_VALIDATION_SUMMARY.json",
        "02_EXECUTION/IMPLEMENTATION_CHANGES.json": OUT / "IMPLEMENTATION_CHANGES.json",
        "02_EXECUTION/EXECUTION_CONSOLE.log": ROOT / "tmp" / "d31_execution_console.log",
        "03_AUTHORIZATION/ORIGINAL_D31_EXECUTION_PROMPT.md": OUT / "ORIGINAL_D31_EXECUTION_PROMPT.md",
        "04_CHECKPOINT/committed_update_426.pt": OUT / "checkpoints" / "committed_update_426.pt",
        "05_SOURCE/stage3_h13_post_d30_d31_dual_mode_direction_recovery.py": ROOT / "tools" / "stage3_h13_post_d30_d31_dual_mode_direction_recovery.py",
        "05_SOURCE/stage3_h13_d31_targeted_boundary_check.py": ROOT / "tools" / "stage3_h13_d31_targeted_boundary_check.py",
        "05_SOURCE/stage3_h13_d31_instrument_mode_a_control.py": ROOT / "tools" / "stage3_h13_d31_instrument_mode_a_control.py",
        "05_SOURCE/package_stage3_h13_d31_cross_chat_handoff.py": Path(__file__).resolve(),
        "06_TESTS/test_stage3_h13_d31_dual_mode_direction_recovery.py": ROOT / "tests" / "test_stage3_h13_d31_dual_mode_direction_recovery.py",
        "06_TESTS/test_stage3_h13_d26_optimizer_consistency_recovery.py": ROOT / "tests" / "test_stage3_h13_d26_optimizer_consistency_recovery.py",
        "07_D30_PARENT_CONTEXT/D30_FINAL_REPORT.md": D30 / "FINAL_REPORT.md",
        "07_D30_PARENT_CONTEXT/D30_CANONICAL_TRAJECTORY.json": D30 / "D30_CANONICAL_TRAJECTORY.json",
        "07_D30_PARENT_CONTEXT/D30_EXECUTION_MANIFEST.json": D30 / "D30_EXECUTION_MANIFEST.json",
    }
    for relative, source in required.items():
        if not source.is_file():
            raise RuntimeError(f"REQUIRED_SOURCE_MISSING:{source}")
        copy(source, relative)
    excluded = """# Deliberately excluded artifacts

- Per-candidate binary optimizer tensor diagnostics from D30 (large and redundant with compact JSON summaries).
- Rejected shadow model/optimizer checkpoints (never canonical).
- D30 update-423 binary checkpoint (identity is recorded; D31 final update-426 checkpoint is included).
- Prior D28/D29/D30 tar.xz archives (nested archives are prohibited).
- Unrelated repository outputs, documents, MoveIt artifacts, and legacy Stage 0/1 material.
"""
    (STAGE / "EXCLUDED_LARGE_AND_REDUNDANT_ARTIFACTS.md").write_text(excluded, encoding="utf-8", newline="\n")
    files = sorted(path for path in STAGE.rglob("*") if path.is_file())
    listing = "path\tsize_bytes\n" + "\n".join(f"{path.relative_to(STAGE).as_posix()}\t{path.stat().st_size}" for path in files) + "\n"
    (STAGE / "FILE_LISTING.tsv").write_text(listing, encoding="utf-8", newline="\n")
    with ARCHIVE.open("wb") as raw:
        with lzma.LZMAFile(raw, "w", format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(STAGE.rglob("*")):
                    if path.is_file():
                        tar.add(path, arcname=path.relative_to(STAGE).as_posix(), recursive=False)
    archive_sha = sha256(ARCHIVE)
    with lzma.open(ARCHIVE, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    with tarfile.open(ARCHIVE, "r:xz") as tar:
        names = set(tar.getnames())
        checkpoint = tar.extractfile("04_CHECKPOINT/committed_update_426.pt")
        if checkpoint is None or hashlib.sha256(checkpoint.read()).hexdigest() != FINAL_CHECKPOINT_SHA256:
            raise RuntimeError("ARCHIVED_CHECKPOINT_IDENTITY_MISMATCH")
        for name in ("START_HERE_NEXT_CHAT.md", "01_REPORT/FINAL_REPORT.md", "02_EXECUTION/D31_EXECUTION_MANIFEST.json"):
            if name not in names or tar.extractfile(name) is None:
                raise RuntimeError(f"REQUIRED_ARCHIVE_MEMBER_MISSING:{name}")
    if ARCHIVE.stat().st_size >= 20_000_000:
        raise RuntimeError("ARCHIVE_SIZE_LIMIT_EXCEEDED")
    verification = {
        "status": "PASS", "archive": str(ARCHIVE), "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_sha, "compression": "XZ/LZMA2 preset 9 EXTREME",
        "xz_decompression": "PASS", "tar_traversal_readability": "PASS",
        "final_checkpoint_present": True, "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        "start_here_present": True, "nested_tar_xz": False,
    }
    (OUT / "FINAL_CROSS_CHAT_ARCHIVE_VERIFICATION.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
