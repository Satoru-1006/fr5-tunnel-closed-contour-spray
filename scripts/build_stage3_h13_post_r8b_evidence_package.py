#!/usr/bin/env python3
"""Copy-only Stage 3 H13 Post-R8B project-owner evidence package builder."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "outputs/stage3_h13_post_r8b_rollout_root_cause_audit_20260814T220200Z"
DESKTOP_BASE = Path(r"C:\Users\86198\Desktop\Stage3_H13_Post_R8B_Root_Cause_Audit_20260814")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_destination() -> Path:
    if not DESKTOP_BASE.exists():
        return DESKTOP_BASE
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = DESKTOP_BASE / timestamp
    suffix = 1
    while candidate.exists():
        candidate = DESKTOP_BASE / f"{timestamp}_{suffix}"
        suffix += 1
    return candidate


def source(category: str, path: Path, reason: str) -> dict[str, Any]:
    return {"category": category, "source": path, "reason": reason}


def all_sources() -> list[dict[str, Any]]:
    A = "A_AUTHORITATIVE_AUDIT"
    B = "B_IMPLEMENTATION_AND_TESTS"
    C = "C_BASELINE_REFERENCE_EVIDENCE"
    return [
        source(A, AUDIT_ROOT / "FINAL_REPORT.md", "authoritative audit conclusion and project-owner answers"),
        source(A, AUDIT_ROOT / "stage3_h13_post_r8b_rollout_root_cause_certificate.json", "terminal certificate"),
        source(A, AUDIT_ROOT / "rollout_mode_comparison.json", "teacher-forced, free-running and controlled-feedback metrics"),
        source(A, AUDIT_ROOT / "horizon_error_growth.csv", "registered-horizon error growth"),
        source(A, AUDIT_ROOT / "state_component_drift.csv", "position/velocity/acceleration drift metrics"),
        source(A, AUDIT_ROOT / "feature_domain_drift.json", "TRAIN-only feature-domain drift diagnostic"),
        source(A, AUDIT_ROOT / "hidden_state_diagnostic.json", "GRU hidden-state diagnostic"),
        source(A, AUDIT_ROOT / "root_cause_decision.json", "root-cause classification and decision questions"),
        source(A, AUDIT_ROOT / "trajectory_region_stratification.json", "curvature/local-region evidence"),
        source(A, AUDIT_ROOT / "provenance_manifest.json", "authoritative paths and SHA256 provenance"),
        source(A, AUDIT_ROOT / "test_summary.txt", "focused and regression test summary"),
        source(B, ROOT / "scripts/stage3_h13_post_r8b_rollout_root_cause_audit.py", "audit implementation"),
        source(B, ROOT / "scripts/build_stage3_h13_post_r8b_evidence_package.py", "copy-only evidence package builder"),
        source(B, ROOT / "tests/test_stage3_h13_post_r8b_rollout_root_cause_audit.py", "focused audit tests"),
        source(B, ROOT / "src/stage3_h13_r6.py", "authoritative R6 rollout helper semantics"),
        source(B, ROOT / "src/stage3_h11_r2_model.py", "R6 residual GRU and causal feature semantics"),
        source(B, ROOT / "src/stage3_h11_dataset.py", "H10 split and TRAIN normalization semantics"),
        source(C, ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z/stage3_h13_r6_terminal_certificate.json", "R6 terminal certificate"),
        source(C, ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z/training_summary.json", "R6 TRAIN/VALIDATION metrics and checkpoint provenance"),
        source(C, ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z/training_config.json", "R6 rollout-aware configuration and semantics"),
        source(C, ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z/checkpoint_manifest.json", "R6 checkpoint hash/size manifest; checkpoint file intentionally not copied"),
        source(C, ROOT / "outputs/stage3_h13_r8_autoregressive_state_semantics_redesign_20260814T163000Z/stage3_h13_r8_terminal_certificate.json", "R8 terminal certificate"),
        source(C, ROOT / "outputs/stage3_h13_r8_autoregressive_state_semantics_redesign_20260814T163000Z/FINAL_REPORT.md", "R8 relevant conclusion"),
        source(C, ROOT / "outputs/stage3_h13_r8_autoregressive_state_semantics_redesign_20260814T163000Z/rollout_horizon_metrics.json", "R8 relevant rollout metrics"),
        source(C, ROOT / "outputs/stage3_h13_r8a_reference_washout_redesign_20260814T095233Z/stage3_h13_r8a_terminal_certificate.json", "R8A terminal certificate"),
        source(C, ROOT / "outputs/stage3_h13_r8a_reference_washout_redesign_20260814T095233Z/FINAL_REPORT.md", "R8A reference-washout conclusion"),
        source(C, ROOT / "outputs/stage3_h13_r8a_reference_washout_redesign_20260814T095233Z/dense_rollout_horizon_metrics.json", "R8A relevant rollout metrics"),
        source(C, ROOT / "outputs/stage3_h13_r8b_causal_curvature_reference_20260814T124600Z/stage3_h13_r8b_terminal_certificate.json", "R8B terminal certificate"),
        source(C, ROOT / "outputs/stage3_h13_r8b_causal_curvature_reference_20260814T124600Z/metric_comparison.json", "R8B material-gate comparison"),
        source(C, ROOT / "outputs/stage3_h13_r8b_causal_curvature_reference_20260814T124600Z/curvature_conditioned_reference_audit.json", "R8B curvature stratification evidence"),
        source(C, ROOT / "outputs/stage3_h13_r7_strict_reconstruction_closure_20260814T150000Z/stage3_h13_r7_terminal_certificate.json", "R7 semantics-related reconstruction status"),
        source(C, ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z/dataset_semantic_hash.json", "H10 dataset semantic identity"),
        source(C, ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z/dataset_split_manifest.json", "TRAIN/VALIDATION split metadata"),
        source(C, ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z/dataset_split_audit.json", "split leakage audit"),
        source(C, ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z/normalization_stats.json", "TRAIN-only normalization statistics"),
    ]


def main() -> int:
    entries = all_sources()
    missing = [str(item["source"]) for item in entries if not item["source"].is_file()]
    if missing:
        raise RuntimeError("missing_source_files:" + ";".join(missing))
    destination = choose_destination()
    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for item in entries:
        category = str(item["category"])
        src = Path(item["source"])
        target_dir = destination / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        if target.exists():
            target = target_dir / f"{src.parent.name}__{src.name}"
        shutil.copy2(src, target)
        source_hash = sha256_file(src)
        target_hash = sha256_file(target)
        records.append({
            "category": category,
            "filename": src.name,
            "original_absolute_path": str(src.resolve()),
            "desktop_copy_path": str(target.resolve()),
            "size_bytes": int(src.stat().st_size),
            "SHA256": source_hash,
            "desktop_copy_SHA256": target_hash,
            "reason_included": item["reason"],
            "hash_match": source_hash == target_hash,
        })
    hash_match = bool(records) and all(bool(row["hash_match"]) for row in records)
    certificate_path = AUDIT_ROOT / "stage3_h13_post_r8b_rollout_root_cause_certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    checkpoint_manifest_path = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z/checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "stage3_h13_post_r8b_project_owner_evidence_manifest_v1",
        "package_type": "copy_only_project_owner_evidence",
        "source_authoritative_audit": str(AUDIT_ROOT.resolve()),
        "desktop_folder": str(destination.resolve()),
        "DESKTOP_COPY_HASH_MATCH": "YES" if hash_match else "NO",
        "COPY_ONLY": "YES",
        "MOVE_SOURCE_FILES": "NO",
        "DELETE_SOURCE_FILES": "NO",
        "MODIFY_SOURCE_FILES": "NO",
        "MODIFY_AUTHORITATIVE_ARTIFACTS": "NO",
        "FROZEN20_OPENED_OR_INSPECTED": "NO",
        "FROZEN20_USED_FOR_DIAGNOSTIC": "NO",
        "R6_CHECKPOINT_COPIED": "NO",
        "R6_CHECKPOINT_PATH": certificate.get("provenance", {}).get("paths", {}).get("R6_CHECKPOINT"),
        "R6_CHECKPOINT_SHA256": certificate.get("provenance", {}).get("R6_CURRENT_CHECKPOINT_SHA256"),
        "R6_CHECKPOINT_SIZE_BYTES": checkpoint_manifest.get("final_size_bytes"),
        "files": records,
        "summary": {
            "authoritative_audit_files": sum(row["category"] == "A_AUTHORITATIVE_AUDIT" for row in records),
            "implementation_test_files": sum(row["category"] == "B_IMPLEMENTATION_AND_TESTS" for row in records),
            "baseline_reference_files": sum(row["category"] == "C_BASELINE_REFERENCE_EVIDENCE" for row in records),
            "total_files": len(records),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in records),
            "total_size_mb": round(sum(int(row["size_bytes"]) for row in records) / (1024.0 * 1024.0), 3),
        },
    }
    (destination / "EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = [
        "# Stage 3 H13 Post-R8B Project-Owner Evidence Manifest",
        "",
        f"DESKTOP_COPY_HASH_MATCH: {'YES' if hash_match else 'NO'}",
        "COPY_ONLY: YES",
        "MOVE_SOURCE_FILES: NO",
        "DELETE_SOURCE_FILES: NO",
        "MODIFY_SOURCE_FILES: NO",
        "MODIFY_AUTHORITATIVE_ARTIFACTS: NO",
        "FROZEN20_OPENED_OR_INSPECTED: NO",
        "FROZEN20_USED_FOR_DIAGNOSTIC: NO",
        "R6_CHECKPOINT_COPIED: NO",
        "",
        f"TOTAL_FILES: {len(records)}",
        f"TOTAL_SIZE_MB: {manifest['summary']['total_size_mb']}",
        "",
        "| Category | Filename | Size (bytes) | SHA256 | Hash match |",
        "|---|---|---:|---|---|",
    ]
    for row in records:
        lines.append(f"| {row['category']} | `{row['filename']}` | {row['size_bytes']} | `{row['SHA256']}` | {'YES' if row['hash_match'] else 'NO'} |")
    (destination / "EVIDENCE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"DESKTOP_FOLDER_CREATED: YES")
    print(f"DESKTOP_FOLDER: {destination.resolve()}")
    print(f"TOTAL_FILES: {len(records)}")
    print(f"TOTAL_SIZE_MB: {manifest['summary']['total_size_mb']}")
    print(f"EVIDENCE_MANIFEST_CREATED: YES")
    print(f"DESKTOP_COPY_HASH_MATCH: {'YES' if hash_match else 'NO'}")
    return 0 if hash_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
