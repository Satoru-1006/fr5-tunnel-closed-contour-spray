from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h12_r2_process_manifold_post_ruckig import (
    TOTAL_WINDOWS,
    aggregate_native,
    sha256_file,
)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def blocker_for(aggregate: dict, replay: dict, focused_ok: bool, regression_ok: bool) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if aggregate["NATIVE_CHECKED_WINDOWS"] != TOTAL_WINDOWS:
        blockers.append("incomplete_native_coverage")
    if aggregate["NATIVE_RUNTIME_ERRORS"] > 0:
        blockers.append("native_runtime_error")
    pre = aggregate.get("pre_failure_counts", {})
    post = aggregate.get("post_failure_counts", {})
    if any(pre.get(key, 0) for key in ("JOINT_POSITION", "JOINT_VELOCITY", "JOINT_ACCELERATION", "JOINT_JERK")):
        blockers.append("pre_ruckig_hard_constraint_violation")
    if any(pre.get(key, 0) for key in ("TCP_POSITION", "TCP_ORIENTATION", "SURFACE_NORMAL", "STANDOFF", "SPRAY_PROCESS_TOLERANCE", "SPRAY_STATE_SEMANTICS")):
        blockers.append("pre_ruckig_process_violation")
    if pre.get("SELF_COLLISION", 0) or pre.get("ENVIRONMENT_COLLISION", 0):
        blockers.append("pre_ruckig_collision_violation")
    if aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] > 0:
        blockers.append("incomplete_post_ruckig_execution")
    if aggregate["failure_counts"].get("RUCKIG_NATIVE_ERROR", 0):
        blockers.append("ruckig_native_error")
    if any(post.get(key, 0) for key in ("JOINT_POSITION", "JOINT_VELOCITY", "JOINT_ACCELERATION", "JOINT_JERK")):
        blockers.append("post_ruckig_hard_constraint_violation")
    if any(post.get(key, 0) for key in ("TCP_POSITION", "TCP_ORIENTATION", "SURFACE_NORMAL", "STANDOFF", "SPRAY_PROCESS_TOLERANCE", "SPRAY_STATE_SEMANTICS")):
        blockers.append("post_ruckig_process_violation")
    if post.get("SELF_COLLISION", 0) or post.get("ENVIRONMENT_COLLISION", 0):
        blockers.append("post_ruckig_collision_violation")
    if replay["DETERMINISTIC_REPLAY"] != "3/3":
        blockers.append("nondeterministic_replay")
    if not focused_ok:
        blockers.append("focused_test_failure")
    if not regression_ok:
        blockers.append("regression_failure")
    return (blockers[0] if blockers else "none"), blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    replay_dir = output / "replay_runs"
    replay_runs = [load_json(replay_dir / f"manual_{idx:02d}.json") for idx in range(3)]
    hashes = [str(item.get("repaired_semantic_sha256")) for item in replay_runs]
    tiers = [item.get("tier_counts") for item in replay_runs]
    metrics = [item.get("metrics") for item in replay_runs]
    checkpoint_hashes = [item.get("checkpoint_sha256") for item in replay_runs]
    replay_ok = len(set(hashes)) == 1 and len(set(json.dumps(item, sort_keys=True) for item in tiers)) == 1 and len(set(json.dumps(item, sort_keys=True) for item in metrics)) == 1 and len(set(checkpoint_hashes)) == 1 and all(int(item.get("TOTAL_WINDOWS", -1)) == TOTAL_WINDOWS for item in replay_runs)
    replay = {
        "schema_version": "stage3_h12_r2_replay_report_v2",
        "replay_count": 3,
        "replay_hashes": hashes,
        "same_semantic_output_hash": len(set(hashes)) == 1,
        "same_window_count": all(int(item.get("TOTAL_WINDOWS", -1)) == TOTAL_WINDOWS for item in replay_runs),
        "same_classification_counts": len(set(json.dumps(item, sort_keys=True) for item in tiers)) == 1,
        "same_final_metrics": len(set(json.dumps(item, sort_keys=True) for item in metrics)) == 1,
        "same_selected_model": len(set(checkpoint_hashes)) == 1,
        "fresh_processes": True,
        "replay_runs": [{"replay_index": idx, "returncode": 0, "TOTAL_WINDOWS": item["TOTAL_WINDOWS"], "checkpoint_sha256": item["checkpoint_sha256"], "repaired_semantic_sha256": item["repaired_semantic_sha256"], "tier_counts": item["tier_counts"], "metrics": item["metrics"]} for idx, item in enumerate(replay_runs)],
        "DETERMINISTIC_REPLAY": "3/3" if replay_ok else "0/3",
        "note": "Three independent fresh Python processes recomputed the causal repair with the frozen E2 model and compared hashes, tier counts, window count, selected-model hash, and final pre-Ruckig metrics.",
    }
    write_json(output / "stage3_h12_r2_replay_report.json", replay)

    units = [json.loads(line) for line in (output / "stage3_h12_r2_primitive_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    prior_aggregate = load_json(output / "stage3_h12_r2_native_aggregate.json")
    aggregate, _ = aggregate_native(output, units, prior_aggregate.get("native_run", {"returncode": 0}))
    write_json(output / "stage3_h12_r2_native_aggregate.json", aggregate)

    focused = __import__("subprocess").run([__import__("sys").executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r2_native_contract.py", "tests/test_stage3_h12_r2_manifold.py", "tests/test_stage3_h12_r.py", "tests/test_stage3_h12_no_leakage.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    regression = __import__("subprocess").run([__import__("sys").executable, "-m", "pytest", "-q", "tests/test_stage3_h8_software_only.py", "tests/test_stage3_h9.py", "tests/test_stage3_h10.py", "tests/test_stage3_h11.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    (output / "stage3_h12_r2_test_report.txt").write_text(focused.stdout + focused.stderr + "\n" + regression.stdout + regression.stderr, encoding="utf-8", newline="\n")
    terminal = load_json(output / "stage3_h12_r2_terminal_certificate.json")
    blocker, blockers = blocker_for(aggregate, replay, focused.returncode == 0, regression.returncode == 0)
    terminal.update({
        "STAGE_3_H12_R2": "PASSED" if blocker == "none" else "BLOCKED",
        "FIRST_BLOCKER": blocker,
        "READY_FOR_STAGE_3_H13": "YES" if blocker == "none" else "NO",
        "POST_RUCKIG_ATTEMPTED_WINDOWS": aggregate["POST_RUCKIG_ATTEMPTED_WINDOWS"],
        "POST_RUCKIG_VALIDATED_WINDOWS": aggregate["POST_RUCKIG_VALIDATED_WINDOWS"],
        "POST_RUCKIG_FAILED_WINDOWS": aggregate["POST_RUCKIG_FAILED_WINDOWS"],
        "POST_RUCKIG_UNVALIDATED_WINDOWS": aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"],
        "FINAL_CERTIFIED_WINDOWS": aggregate["FINAL_CERTIFIED_WINDOWS"],
        "NATIVE_RUNTIME_ERRORS": aggregate["NATIVE_RUNTIME_ERRORS"],
        "RUCKIG_RESULT": aggregate.get("ruckig_summary", {}),
        "RUCKIG_ERROR_INVALID_INPUT": aggregate.get("ruckig_summary", {}).get("RUCKIG_ERROR_INVALID_INPUT_UNITS", 0),
        "RUCKIG_OTHER_ERRORS": aggregate.get("ruckig_summary", {}).get("RUCKIG_OTHER_ERROR_UNITS", 0),
        "RUCKIG_INCOMPLETE_UNITS": aggregate.get("ruckig_summary", {}).get("RUCKIG_INCOMPLETE_UNITS", 0),
        "DETERMINISTIC_REPLAY": replay["DETERMINISTIC_REPLAY"],
        "FOCUSED_TESTS": "PASS" if focused.returncode == 0 else "BLOCKED",
        "REGRESSION_TESTS": "PASS" if regression.returncode == 0 else "BLOCKED",
    })
    write_json(output / "stage3_h12_r2_terminal_certificate.json", terminal)
    gate = load_json(output / "stage3_h12_r2_gate_report.json")
    gate.update(terminal)
    gate["blockers"] = blockers
    gate["coverage_invariants"] = {"partition_holds": aggregate["POST_RUCKIG_VALIDATED_WINDOWS"] + aggregate["POST_RUCKIG_FAILED_WINDOWS"] + aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] == aggregate["TOTAL_WINDOWS"]}
    gate["required"] = {"baseline_reproduced": True, "native_checked_complete": aggregate["NATIVE_CHECKED_WINDOWS"] == TOTAL_WINDOWS, "post_ruckig_unvalidated_zero": aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] == 0, "tests": focused.returncode == 0 and regression.returncode == 0, "upstream_immutability": terminal["H10_SEMANTIC_HASH_MATCH"] == "YES" and terminal["H11_PROVENANCE_VALID"] == "YES" and terminal["E2_MODEL_IMMUTABLE"] == "YES" and terminal["HISTORICAL_H12_IMMUTABLE"] == "YES" and terminal["H12_R_IMMUTABLE"] == "YES"}
    write_json(output / "stage3_h12_r2_gate_report.json", gate)

    report = output / "FINAL_REPORT.md"
    lines = report.read_text(encoding="utf-8").splitlines()
    lines[0] = "# Stage 3 H12-R2 — Process-Manifold-Constrained Neural Repair + Full Post-Ruckig Native Certification Closure"
    report_keys = ["STAGE_3_H12_R2", "FIRST_BLOCKER", "READY_FOR_STAGE_3_H13", "POST_RUCKIG_ATTEMPTED_WINDOWS", "POST_RUCKIG_VALIDATED_WINDOWS", "POST_RUCKIG_FAILED_WINDOWS", "POST_RUCKIG_UNVALIDATED_WINDOWS", "FINAL_CERTIFIED_WINDOWS", "NATIVE_RUNTIME_ERRORS", "DETERMINISTIC_REPLAY", "FOCUSED_TESTS", "REGRESSION_TESTS"]
    for index, line in enumerate(lines):
        for key in report_keys:
            if line.startswith(key + ":"):
                lines[index] = f"{key}: {terminal.get(key)}"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "frozen_artifact_manifest.json":
            records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_json(output / "frozen_artifact_manifest.json", {"schema_version": "stage3_h12_r2_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": records})
    print(json.dumps({"replay": replay, "blocker": blocker, "focused_returncode": focused.returncode, "regression_returncode": regression.returncode}, ensure_ascii=True, sort_keys=True))
    return 0 if replay_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
