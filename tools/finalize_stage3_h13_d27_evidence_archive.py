"""Build and verify the single compressed D27 evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"MISSING_SOURCE:{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def archive_files(stage: Path) -> list[dict[str, Any]]:
    return [{"relative_path": path.relative_to(stage).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "ARCHIVE_INTERNAL_VERIFICATION.json"]


def refresh_d27_hash_manifest(source: Path) -> None:
    manifest_path = source / "CHECKPOINT_HASH_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    managed = [{"relative_path": path.relative_to(source).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sorted(source.rglob("*")) if path.is_file() and path.name != manifest_path.name]
    manifest["managed_files"] = managed
    manifest["file_count"] = len(managed)
    manifest["checkpoint_entries"] = [entry for entry in managed if entry["relative_path"].startswith("checkpoints/")]
    write_json(manifest_path, manifest)


def verify_internal(extracted: Path) -> dict[str, Any]:
    verification_path = extracted / "ARCHIVE_INTERNAL_VERIFICATION.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    actual = {entry["relative_path"]: entry for entry in archive_files(extracted)}
    expected = {entry["relative_path"]: entry for entry in verification["managed_files"]}
    managed_pass = actual == expected
    d27_manifest = json.loads((extracted / "D27" / "D27_EXECUTION_MANIFEST.json").read_text(encoding="utf-8"))
    checkpoint_results = []
    for entry in d27_manifest["commit_evidence"]:
        update = int(entry["update"])
        path = extracted / "D27" / "checkpoints" / f"committed_update_{update}.pt"
        checkpoint_results.append({"update": update, "expected_sha256": entry["checkpoint_sha256"], "actual_sha256": sha256(path), "pass": sha256(path) == entry["checkpoint_sha256"]})
    d26_parent = extracted / "D26_HANDOFF" / "CHECKPOINTS" / "committed_update_400.pt"
    d26_parent_hash = sha256(d26_parent)
    d26_expected = d27_manifest["authoritative_d26"]["update_400_checkpoint_sha256"]
    return {"managed_files_pass": managed_pass, "checkpoint_results": checkpoint_results, "checkpoints_pass": all(item["pass"] for item in checkpoint_results), "d26_parent_expected_sha256": d26_expected, "d26_parent_actual_sha256": d26_parent_hash, "d26_parent_pass": d26_parent_hash == d26_expected, "status": "PASS" if managed_pass and all(item["pass"] for item in checkpoint_results) and d26_parent_hash == d26_expected else "FAIL"}


def create_tar_xz(stage: Path, archive_path: Path) -> None:
    preset = 9 | lzma.PRESET_EXTREME
    with lzma.open(archive_path, "wb", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=preset) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            archive.add(stage, arcname=stage.name, recursive=True)


def read_tar_xz(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path, mode="r:xz") as archive:
        return archive.getnames()


def extract_tar_xz(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:xz") as archive:
        archive.extractall(destination)


def run(source: Path, d26_archive: Path, stage: Path, archive_path: Path, finalization_report: Path, refresh: bool = False) -> dict[str, Any]:
    if stage.exists() and not refresh:
        raise RuntimeError(f"STAGING_ALREADY_EXISTS:{stage}")
    if archive_path.exists() and not refresh:
        raise RuntimeError(f"ARCHIVE_ALREADY_EXISTS:{archive_path}")
    stage.mkdir(parents=True, exist_ok=True)
    refresh_d27_hash_manifest(source)
    required_d27 = [
        "D27_EXECUTION_MANIFEST.json", "FINAL_REPORT.md", "D27_FORWARD_TRAJECTORY.json", "CHECKPOINT_HASH_MANIFEST.json", "OPTIMIZER_CONTINUITY_REPORT.json", "NEXT_CHAT_HANDOFF.txt", "START_HERE_NEXT_CHAT.txt", "D27_CANDIDATE_RESULTS.csv", "D27_FORWARD_TRAJECTORY.csv",
    ]
    for name in required_d27:
        copy_file(source / name, stage / "D27" / name)
    for path in sorted((source / "checkpoints").glob("committed_update_*.pt")):
        copy_file(path, stage / "D27" / "checkpoints" / path.name)
    for path in sorted((source / "metrics").glob("update_*.json")):
        copy_file(path, stage / "D27" / "metrics" / path.name)
    for path in sorted((source / "optimizer_consistency").glob("update_*.json")):
        copy_file(path, stage / "D27" / "optimizer_consistency" / path.name)

    d26_files = {
        "START_HERE_NEXT_CHAT.txt": d26_archive / "START_HERE_NEXT_CHAT.txt",
        "FINAL_REPORT.md": d26_archive / "FINAL_RESULTS" / "FINAL_REPORT.md",
        "D26_EXECUTION_MANIFEST.json": d26_archive / "FINAL_RESULTS" / "D26_EXECUTION_MANIFEST.json",
        "D26_OPTIMIZER_CONSISTENCY_AB.json": d26_archive / "FINAL_RESULTS" / "D26_OPTIMIZER_CONSISTENCY_AB.json",
        "D26_COMMITTED_FORWARD_TRAJECTORY.csv": d26_archive / "NUMERICAL_TRAJECTORY" / "D26_COMMITTED_FORWARD_TRAJECTORY.csv",
        "D26_CANDIDATE_RESULTS.csv": d26_archive / "NUMERICAL_TRAJECTORY" / "D26_CANDIDATE_RESULTS.csv",
        "D26_SHA256_MANIFEST.json": d26_archive / "HASH_MANIFESTS" / "D26_SHA256_MANIFEST.json",
        "CHECKPOINT_IDENTITY.json": d26_archive / "CHECKPOINT_IDENTITY.json",
    }
    for name, path in d26_files.items():
        copy_file(path, stage / "D26_HANDOFF" / name)
    copy_file(d26_archive / "CHECKPOINTS" / "committed_update_400.pt", stage / "D26_HANDOFF" / "CHECKPOINTS" / "committed_update_400.pt")
    copy_file(source / "START_HERE_NEXT_CHAT.txt", stage / "START_HERE_NEXT_CHAT.txt")

    contents = [
        "STAGE 3 H13 D27 FINAL EVIDENCE ARCHIVE",
        "",
        "D27: reports, manifests, six committed checkpoints (401-406), trajectory CSV/JSON, per-update metrics, and optimizer continuity summaries.",
        "D26_HANDOFF: D26 handoff/report/manifests/trajectory/hash identity and the authenticated update-400 parent checkpoint.",
        "START_HERE_NEXT_CHAT.txt: D27 next-chat handoff.",
        "",
        "EXCLUDED: D27 optimizer_consistency/*.pt replay/diagnostic tensor binaries. Their summary JSONs, hashes, and all committed checkpoints are retained; exclusion is solely to keep the single archive below 20 MiB.",
        "",
        "VERIFICATION: ARCHIVE_INTERNAL_VERIFICATION.json is self-excluded from its own managed-file list and is checked after XZ extraction.",
    ]
    (stage / "ARCHIVE_CONTENTS.txt").write_text("\n".join(contents) + "\n", encoding="utf-8", newline="\n")
    managed = archive_files(stage)
    write_json(stage / "ARCHIVE_INTERNAL_VERIFICATION.json", {"schema_version": "stage3_h13_d27_archive_internal_verification_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "managed_files": managed, "excluded_source_files": "D27/optimizer_consistency/*.pt", "d27_checkpoint_manifest": "D27/CHECKPOINT_HASH_MANIFEST.json", "d26_parent_checkpoint": "D26_HANDOFF/CHECKPOINTS/committed_update_400.pt"})

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    create_tar_xz(stage, archive_path)
    try:
        with lzma.open(archive_path, "rb") as compressed:
            while compressed.read(1024 * 1024):
                pass
        xz_status = "PASS"
    except Exception:
        xz_status = "FAIL"
    try:
        tar_names = read_tar_xz(archive_path)
        tar_status = "PASS" if tar_names else "FAIL"
    except Exception:
        tar_names = []
        tar_status = "FAIL"
    with tempfile.TemporaryDirectory(prefix="d27_archive_verify_") as temp:
        extract_parent = Path(temp)
        extract_tar_xz(archive_path, extract_parent)
        internal = verify_internal(extract_parent / stage.name)
    result = {"schema_version": "stage3_h13_d27_archive_finalization_report_v1", "archive": str(archive_path.resolve()), "archive_size_bytes": archive_path.stat().st_size, "archive_sha256": sha256(archive_path), "compression_format": "XZ / LZMA2", "compression_preset": "9e EXTREME", "compression_backend": "Python lzma", "xz_test_status": xz_status, "tar_read_status": tar_status, "tar_content_count": len(tar_names), "internal_verification": internal, "status": "PASS" if xz_status == "PASS" and tar_status == "PASS" and internal["status"] == "PASS" and archive_path.stat().st_size < 20 * 1024 * 1024 else "FAIL", "archive_contents_file_count": len(managed) + 1, "excluded_source_files": "D27/optimizer_consistency/*.pt"}
    write_json(finalization_report, result)
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize and verify the single D27 tar.xz evidence archive")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--d26-archive", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--finalization-report", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.source.resolve(), args.d26_archive.resolve(), args.stage.resolve(), args.archive.resolve(), args.finalization_report.resolve(), args.refresh), sort_keys=True))
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}:{error}"}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
