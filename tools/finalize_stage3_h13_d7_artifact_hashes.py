"""Finalize the non-circular D7 artifact ledger after execution."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h13_post_d6_d7_phase4_pareto_localization_replay import render_report  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    output = ROOT / "outputs/stage3_h13_post_d6_d7_phase4_pareto_localization_replay_20260817T023840Z"
    certificate_path = output / "terminal_certificate.json"
    trajectory_path = output / "trajectory_metrics.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))["trajectory"]
    stable_names = [
        "protocol_integrity_identity.json",
        "execution_configuration_snapshot.json",
        "source_revision_record.json",
        "preflight_identity.json",
        "trajectory_metrics.json",
        "pareto_classification.json",
        "training_diagnostics.csv",
        "module_diagnostics.csv",
        "model_optimizer_state_digest_manifest.json",
        "terminal_state.pt",
    ]
    stable_hashes = {name: sha256_file(output / name) for name in stable_names}
    certificate["artifact_sha256"] = stable_hashes
    write_json(certificate_path, certificate)
    certificate_hash = sha256_file(certificate_path)
    ledger_files = {name: sha256_file(output / name) for name in stable_names + ["terminal_certificate.json"]}
    write_json(output / "SHA256_MANIFEST.json", {"schema_version": "stage3_h13_d7_sha256_manifest_v2", "hash_algorithm": "SHA-256", "self_excluded": True, "report_excluded": True, "files": ledger_files})
    manifest_hash = sha256_file(output / "SHA256_MANIFEST.json")
    report_artifacts = {**stable_hashes, "terminal_certificate.json": certificate_hash, "SHA256_MANIFEST.json": manifest_hash}
    report = render_report(certificate, trajectory, report_artifacts)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "FINALIZED", "certificate_sha256": certificate_hash, "manifest_sha256": manifest_hash, "report_sha256": sha256_file(output / "FINAL_REPORT.md")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
