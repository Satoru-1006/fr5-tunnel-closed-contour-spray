"""Build and verify the single maximum-compression D28 Desktop archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\86198\Desktop")
MAX_BYTES = 20 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def copy_selected(output: Path, stage: Path) -> None:
    d28 = stage / "D28"
    for source in sorted(output.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(output)
        if relative.as_posix().startswith("optimizer_consistency/") and source.suffix == ".pt":
            continue
        if relative.as_posix().startswith("checkpoints/") and source.name != "committed_update_412.pt":
            continue
        target = d28 / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    implementation = stage / "IMPLEMENTATION"
    implementation.mkdir(parents=True, exist_ok=True)
    for source in (
        ROOT / "tools" / "stage3_h13_post_d27_d28_forward_continuation.py",
        ROOT / "tools" / "stage3_h13_post_d26_d27_forward_continuation.py",
        ROOT / "tools" / "package_stage3_h13_d28_evidence.py",
    ):
        shutil.copy2(source, implementation / source.name)

    d27_root = ROOT / "outputs" / "stage3_h13_post_d26_d27_forward_continuation_20260822T094500+0800"
    d27_handoff = stage / "D27_HANDOFF"
    d27_handoff.mkdir(parents=True, exist_ok=True)
    for name in ("START_HERE_NEXT_CHAT.txt", "FINAL_REPORT.md", "D27_EXECUTION_MANIFEST.json", "D27_FORWARD_TRAJECTORY.json", "OPTIMIZER_CONTINUITY_REPORT.json", "CHECKPOINT_HASH_MANIFEST.json"):
        shutil.copy2(d27_root / name, d27_handoff / name)

    authorization = Path(r"C:\Users\86198\.codex\attachments\2748ff68-b994-42c3-a8b7-b0b22e60e25f\pasted-text.txt")
    auth_dir = stage / "AUTHORIZATION"
    auth_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(authorization, auth_dir / "D28_EXECUTION_AUTHORIZATION.txt")


def build_internal_manifest(stage: Path) -> dict[str, Any]:
    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "ARCHIVE_INTERNAL_MANIFEST.json":
            files.append({"relative_path": path.relative_to(stage).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {"schema_version": "stage3_h13_d28_archive_internal_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "managed_file_count": len(files), "managed_files": files}
    write_json(stage / "ARCHIVE_INTERNAL_MANIFEST.json", manifest)
    return manifest


def deterministic_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.isfile():
        info.mode = 0o644
    elif info.isdir():
        info.mode = 0o755
    return info


def create_archive(stage: Path, archive: Path) -> None:
    raw_tar = stage.parent / f"{stage.name}.tar"
    if raw_tar.exists():
        raise RuntimeError(f"Refusing pre-existing temporary tar: {raw_tar}")
    with tarfile.open(raw_tar, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.add(stage, arcname=stage.name, recursive=True, filter=deterministic_filter)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=9 | lzma.PRESET_EXTREME)
    with raw_tar.open("rb") as source, archive.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(compressor.compress(chunk))
        target.write(compressor.flush())
    raw_tar.unlink()


def verify_archive(archive: Path, stage_name: str) -> dict[str, Any]:
    decompressed_bytes = 0
    with lzma.open(archive, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            decompressed_bytes += len(chunk)
    with tarfile.open(archive, "r:xz") as tar:
        members = tar.getmembers()
        files = [member for member in members if member.isfile()]
        names = {member.name for member in files}
        manifest_name = f"{stage_name}/ARCHIVE_INTERNAL_MANIFEST.json"
        manifest_stream = tar.extractfile(manifest_name)
        if manifest_stream is None:
            raise RuntimeError("Internal manifest missing")
        manifest = json.loads(manifest_stream.read().decode("utf-8"))
        mismatches = []
        for entry in manifest["managed_files"]:
            member_name = f"{stage_name}/{entry['relative_path']}"
            extracted = tar.extractfile(member_name)
            if extracted is None:
                mismatches.append({"relative_path": entry["relative_path"], "reason": "missing"})
                continue
            data = extracted.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry["sha256"] or len(data) != entry["size_bytes"]:
                mismatches.append({"relative_path": entry["relative_path"], "reason": "hash_or_size_mismatch"})
        expected_names = {f"{stage_name}/{entry['relative_path']}" for entry in manifest["managed_files"]} | {manifest_name}
        unexpected_or_missing = sorted(names.symmetric_difference(expected_names))
    size = archive.stat().st_size
    return {
        "schema_version": "stage3_h13_d28_final_archive_verification_v1",
        "archive_path": str(archive.resolve()), "archive_size_bytes": size, "archive_size_mib": size / (1024 * 1024), "archive_sha256": sha256_file(archive),
        "archive_file_count": len(files), "compression": "XZ / LZMA2 preset 9 EXTREME", "decompressed_tar_bytes": decompressed_bytes,
        "xz_decompression_test": "PASS", "tar_content_read_test": "PASS", "internal_manifest_verification": "PASS" if not mismatches and not unexpected_or_missing else "FAIL",
        "included_file_sha256_verification": "PASS" if not mismatches else "FAIL", "manifest_topology_verification": "PASS" if not unexpected_or_missing else "FAIL",
        "cross_chat_handoff_present": f"{stage_name}/D28/START_HERE_NEXT_CHAT.txt" in names, "final_report_present": f"{stage_name}/D28/FINAL_REPORT.md" in names,
        "final_classification_present": f"{stage_name}/D28/FINAL_CLASSIFICATION.json" in names, "size_le_20_mib": size <= MAX_BYTES,
        "mismatches": mismatches, "unexpected_or_missing": unexpected_or_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    stage = ROOT / "tmp" / f"d28_final_archive_stage_{args.stamp}"
    archive = DESKTOP / f"STAGE3_H13_D28_FORWARD_CONTINUATION_EVIDENCE_MAX_XZ_{args.stamp}.tar.xz"
    existing = sorted(DESKTOP.glob("*D28*FORWARD_CONTINUATION*EVIDENCE*.tar.xz"))
    if existing:
        raise SystemExit(f"Refusing competing D28 archive(s): {existing}")
    if stage.exists() or archive.exists():
        raise SystemExit("Refusing to overwrite staging directory or archive")
    stage.mkdir(parents=True)
    copy_selected(output, stage)
    contents = ["D28 scientific reports/manifests and per-update JSON evidence", "final committed update-412 checkpoint", "D28/D27 implementation sources", "compact D27 handoff/provenance", "D28 authorization"]
    (stage / "ARCHIVE_CONTENTS.txt").write_text("\n".join(contents) + "\n", encoding="utf-8", newline="\n")
    build_internal_manifest(stage)
    create_archive(stage, archive)
    verification = verify_archive(archive, stage.name)
    required = (verification["internal_manifest_verification"] == "PASS" and verification["cross_chat_handoff_present"] and verification["final_report_present"] and verification["final_classification_present"] and verification["size_le_20_mib"])
    verification["overall"] = "PASS" if required else "FAIL"
    write_json(output / "FINAL_ARCHIVE_VERIFICATION.json", verification)
    print(json.dumps(verification, sort_keys=True))
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
