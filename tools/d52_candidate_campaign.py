"""D52 shadow candidate campaign helpers.

The campaign deliberately keeps candidate material in ``outputs/D52_STAGE4B_SHADOW``.
It can prepare a native MoveIt input manifest, repair the one verified exporter
bridge in a native post-Ruckig result, and create the summary/manifest shape
used by the existing FK and geometry evaluators.  It never writes to D47, D50,
D51, or the Stage 3 release.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.d52_motion_repair import CASES, case_family, repair_execution_form, wsl_path


D47_CASES = ROOT / "outputs" / "D47_STAGE4B_ROLLING_CHAMPION_V1" / "PROMOTED" / "B3" / "cases"


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_input_manifest(path: Path, source_dir: Path, case_ids: tuple[str, ...] = CASES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_csv", "family"))
        for case_id in case_ids:
            trajectory = source_dir / f"{case_id}.csv"
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            writer.writerow((case_id, wsl_path(trajectory), case_family(case_id)))


def write_baseline_duration_csv(path: Path, native_summary: Path, case_ids: tuple[str, ...] = CASES) -> None:
    summary = json.loads(native_summary.read_text(encoding="utf-8"))
    by_id = {str(item["case_id"]): item for item in summary.get("cases", [])}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_duration_s"))
        for case_id in case_ids:
            item = by_id.get(case_id)
            if not item or item.get("status") != "PASS":
                raise RuntimeError(f"missing_passing_baseline_case:{case_id}")
            writer.writerow((case_id, float(item["duration_s"])))


def repair_native_summary(native_dir: Path, output_dir: Path, manifest: Path | None = None) -> dict[str, Any]:
    """Copy a native summary into a shadow tree with the verified bridge repaired."""
    native_dir = native_dir.resolve()
    output_dir = output_dir.resolve()
    source_summary_path = native_dir / "execution_form_summary.json"
    if not source_summary_path.is_file():
        raise FileNotFoundError(source_summary_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    for item in source_summary.get("cases", []):
        if item.get("status") != "PASS":
            raise RuntimeError(f"native_candidate_case_not_pass:{item.get('case_id')}")
        case_id = str(item["case_id"])
        source_text = str(item["trajectory_csv"])
        source = Path(source_text)
        if source_text.startswith("/mnt/"):
            drive, rest = source_text[5], source_text[7:]
            source = Path(f"{drive.upper()}:/{rest}")
        source = source.resolve()
        destination = output_dir / f"{case_id}.csv"
        evidence = repair_execution_form(source, destination, expected_dt=0.01)
        repaired_item = dict(item)
        repaired_item["trajectory_csv"] = wsl_path(destination)
        repaired_item["d52_source_trajectory_csv"] = wsl_path(source)
        repaired_item["d52_execution_form_repair"] = evidence
        records.append(repaired_item)
        repair_records.append(evidence)
    output_summary = dict(source_summary)
    output_summary.update({
        "schema_version": "d52-shadow-execution-form-v1",
        "candidate_tree": str(output_dir),
        "source_execution_form_summary": str(source_summary_path),
        "d52_measurement_repair": "verified_single_exporter_bridge_timestamp_marker_with_reconstructed_bridge_derivatives",
        "cases": records,
        "case_count": len(records),
        "status": "PASS" if len(records) == len(CASES) else "BLOCKED",
    })
    dump_json(output_dir / "execution_form_summary.json", output_summary)
    dump_json(output_dir / "repair_summary.json", {
        "schema_version": "d52-execution-form-repair-v1",
        "status": "PASS" if all(item["status"] in {"REPAIRED", "NO_REPAIR_NEEDED"} for item in repair_records) else "BLOCKED",
        "method": "repair_one_proven_bridge_timestamp_marker_and_shift_following_native_states",
        "collision_method": "adaptive_discrete_interpolation",
        "case_count": len(repair_records),
        "repaired_case_count": sum(item["status"] == "REPAIRED" for item in repair_records),
        "cases": repair_records,
    })
    if manifest is not None:
        write_input_manifest(manifest.resolve(), output_dir, tuple(str(item["case_id"]) for item in records))
    return output_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    manifest_parser = sub.add_parser("input-manifest")
    manifest_parser.add_argument("--source-dir", type=Path, default=D47_CASES)
    manifest_parser.add_argument("--output", type=Path, required=True)

    duration_parser = sub.add_parser("baseline-durations")
    duration_parser.add_argument("--summary", type=Path, required=True)
    duration_parser.add_argument("--output", type=Path, required=True)

    repair_parser = sub.add_parser("repair-native-summary")
    repair_parser.add_argument("--native-dir", type=Path, required=True)
    repair_parser.add_argument("--output-dir", type=Path, required=True)
    repair_parser.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "input-manifest":
        write_input_manifest(args.output.resolve(), args.source_dir.resolve())
        print(json.dumps({"status": "PASS", "manifest": str(args.output.resolve()), "case_count": len(CASES)}))
        return 0
    if args.command == "baseline-durations":
        write_baseline_duration_csv(args.output.resolve(), args.summary.resolve())
        print(json.dumps({"status": "PASS", "output": str(args.output.resolve()), "case_count": len(CASES)}))
        return 0
    result = repair_native_summary(args.native_dir, args.output_dir, args.manifest)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve()), "case_count": result["case_count"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
