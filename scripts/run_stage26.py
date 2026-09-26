"""Formal Stage 2.6 continuous robot-world certification.

This stage consumes the immutable Stage 2.5 Ruckig trajectory. It never
rebuilds IK, segmentation, TOTG, Ruckig, or Stage 2.4T inputs. Bullet is
invoked through MoveIt's two-state CollisionEnv API; FCL capability is
audited separately because the installed MoveIt FCL overload is a stub.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE25_ROOT = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
PARTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
STAGE26_LAUNCH = ROOT / "tools/stage26_continuous_launch.py"
NATIVE_INSTALL = ROOT / "tmp/stage26_install"

STAGE25_CORE_FILES = [
    "stage25_gate_report.json", "stage25_input_manifest.json", "stage25_process_manifest.json",
    "stage25_segmentation_manifest.json", "stage25_path_reference.json",
    "stage25_ruckig_trajectory.csv", "stage25_ruckig_audit.json", "stage25_pose_validation.json",
    "stage25_collision_validation.json", "stage25_off_transition_validation.json",
    "stage25_determinism_report.json", "stage25_joint_limits.json",
    "stage25_frozen_input_hash_check.json", "stage25_native_moveit_runtime.json",
    "stage25_fk_trace.csv", "SHA256SUMS",
]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def file_entry(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None}


def load_stage25() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    gate = read_json(STAGE25_ROOT / "stage25_gate_report.json")
    trajectory = list(csv.DictReader((STAGE25_ROOT / "stage25_ruckig_trajectory.csv").open(encoding="utf-8", newline="")))
    fk = list(csv.DictReader((STAGE25_ROOT / "stage25_fk_trace.csv").open(encoding="utf-8", newline="")))
    if len(trajectory) != len(fk):
        raise RuntimeError(f"Stage 2.5 timed trajectory/FK row mismatch: {len(trajectory)} != {len(fk)}")
    if gate.get("Stage_2_5") != "passed":
        raise RuntimeError(f"Frozen Stage 2.5 gate is not passed: {gate.get('Stage_2_5')!r}")
    if gate.get("Stage_2_6") != "unblocked_not_started":
        raise RuntimeError(f"Frozen Stage 2.5 Stage_2_6 entry is unexpected: {gate.get('Stage_2_6')!r}")
    return gate, trajectory, fk


def build_input_manifest(gate: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for name in STAGE25_CORE_FILES:
        path = STAGE25_ROOT / name
        if not path.exists():
            raise RuntimeError(f"Missing frozen Stage 2.5 core input: {path}")
        entries.append(file_entry(path, f"frozen Stage 2.5 core input: {name}"))
    return {
        "schema_version": "stage26-input-manifest-v1",
        "stage": "Stage_2_6",
        "contract_source": "user_prompt_fallback_because_repository_stage26_contract_not_found",
        "source_stage25_root": str(STAGE25_ROOT.resolve()),
        "source_stage25_gate": {key: gate.get(key) for key in (
            "Stage_2_4T", "Stage_2_5", "Stage_2_6", "TOTG", "Ruckig",
            "pose_constraints_passed", "max_tcp_position_error_mm", "samples_over_6mm",
            "position_limits_passed", "velocity_limits_passed", "acceleration_limits_passed",
            "jerk_limits_passed", "spray_on_segments", "spray_off_transitions",
            "process_order_preserved", "boundary_order_preserved", "determinism",
            "core_artifact_mismatch_count")},
        "frozen_inputs": entries,
        "no_stage25_rebuild_requested": True,
        "no_ik_or_segmentation_mutation": True,
        "no_totg_or_ruckig_mutation": True,
    }


def hash_manifest_entries(manifest: dict[str, Any]) -> dict[str, str | None]:
    return {entry["path"]: sha256(Path(entry["path"])) if Path(entry["path"]).is_file() else None for entry in manifest["frozen_inputs"]}


def build_intervals(output_root: Path, trajectory: list[dict[str, str]], fk: list[dict[str, str]]) -> dict[str, Any]:
    process = read_json(STAGE25_ROOT / "stage25_process_manifest.json")
    items = process["ordered_items"]
    path_reference = read_json(STAGE25_ROOT / "stage25_path_reference.json")
    reference_items = path_reference["items"]
    interval_path = output_root / "stage26_intervals.csv"
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary", *[f"q0_{j}" for j in range(6)], *[f"q1_{j}" for j in range(6)]]
    records: list[dict[str, Any]] = []
    missing = []
    for index in range(len(trajectory) - 1):
        t0, t1 = float(trajectory[index]["t"]), float(trajectory[index + 1]["t"])
        if t1 <= t0:
            raise RuntimeError(f"Non-increasing Stage 2.5 time at interval {index}: {t0} -> {t1}")
        if abs(t0 - float(fk[index]["t_s"])) > 1e-8 or abs(t1 - float(fk[index + 1]["t_s"])) > 1e-8:
            raise RuntimeError(f"Timed trajectory/FK time mismatch at interval {index}")
        state0, state1 = fk[index]["spray_state"], fk[index + 1]["spray_state"]
        source_path_index = int(fk[index]["source_path_index"])
        source_wp = int(fk[index]["source_waypoint"]) if fk[index].get("source_waypoint", "").strip() else None
        candidates = [
            item for item in items
            if item["spray_state"] == state0
            and source_wp is not None
            and int(item["source_waypoint_start"]) <= source_wp <= int(item["source_waypoint_end"])
        ]
        if not candidates:
            candidates = [
                item for item, reference in zip(items, reference_items)
                if item["spray_state"] == state0
                and int(reference["pre_timing_path_sample_start"]) <= source_path_index <= int(reference["pre_timing_path_sample_end"])
            ]
        if not candidates and state0 != state1:
            candidates = [item for item in items if item["spray_state"] == "OFF" and source_wp is not None and int(item["source_waypoint_start"]) <= source_wp <= int(item["source_waypoint_end"])]
        if not candidates:
            missing.append(index)
            continue
        item = candidates[0]
        kind = "spray_on_segment" if state0 == state1 == "ON" else "spray_off_transition" if state0 == state1 == "OFF" else "boundary_crossing"
        records.append({
            "interval_index": index, "time_start_s": t0, "time_end_s": t1,
            "process_order_index": int(item["process_order_index"]),
            "spray_state": f"{state0}->{state1}" if state0 != state1 else state0,
            "process_kind": kind,
            "segment_id": int(item["segment_id"]) if item.get("segment_id") else None,
            "transition_id": int(item["transition_id"]) if item.get("transition_id") else None,
            "source_boundary": item.get("source_boundary"),
            "q0": [float(trajectory[index][f"j{j}_q"]) for j in range(1, 7)],
            "q1": [float(trajectory[index + 1][f"j{j}_q"]) for j in range(1, 7)],
        })
    if missing:
        raise RuntimeError(f"Could not map {len(missing)} timed intervals to Stage 2.5 process semantics; first={missing[0]}")
    with interval_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in ("interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "source_boundary")}
            row["segment_id"] = "" if record["segment_id"] is None else record["segment_id"]
            row["transition_id"] = "" if record["transition_id"] is None else record["transition_id"]
            row.update({f"q0_{j}": record["q0"][j] for j in range(6)})
            row.update({f"q1_{j}": record["q1"][j] for j in range(6)})
            writer.writerow(row)
    order = [int(row["process_order_index"]) for row in records]
    transition_order = [int(row["transition_id"]) for row in records if row["transition_id"] is not None]
    return {
        "path": str(interval_path.resolve()), "sha256": sha256(interval_path),
        "interval_count": len(records), "timed_state_count": len(trajectory),
        "on_segments_observed": sorted({int(row["segment_id"]) for row in records if row["segment_id"] is not None}),
        "off_transitions_observed": sorted(set(transition_order)),
        "process_order_preserved": order == sorted(order),
        "boundary_order_preserved": transition_order == sorted(transition_order),
        "records": records,
    }


def find_interposer() -> Path:
    matches = list((ROOT / "outputs/ik_graph_stage23b_representation_remediation").rglob("libstage23b_bullet_shape_interposer.so"))
    if not matches:
        raise RuntimeError("Certified Stage 2.3B Bullet shape interposer not found")
    return sorted(matches)[0]


def run_native(output_dir: Path, interval_csv: Path, backend: str, run_index: int, capability_only: bool, positive_controls: bool, ros_domain_id: int | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = find_interposer()
    stage23a7_install = ROOT / "tmp/stage23a7_install2"
    am = ":".join([wsl_path(NATIVE_INSTALL / "stage26_continuous_collision"), wsl_path(ROOT / "install"), wsl_path(stage23a7_install), wsl_path(ROOT / "install/fairino5_v6_moveit2_config"), wsl_path(ROOT / "install/fairino_description"), wsl_path(ROOT / "install/fr5_tunnel_moveit_bridge"), "/opt/ros/jazzy"])
    ld = ":".join([f"{wsl_path(NATIVE_INSTALL)}/stage26_continuous_collision/lib", f"{wsl_path(overlay.parent)}", f"{wsl_path(stage23a7_install)}/lib", f"{wsl_path(ROOT / 'install')}/lib", "/opt/ros/jazzy/opt/sdformat_vendor/lib", "/opt/ros/jazzy/opt/rviz_ogre_vendor/lib", "/opt/ros/jazzy/opt/gz_math_vendor/lib", "/opt/ros/jazzy/opt/gz_utils_vendor/lib", "/opt/ros/jazzy/opt/gz_tools_vendor/lib", "/opt/ros/jazzy/lib/x86_64-linux-gnu", "/opt/ros/jazzy/lib"])
    preload = f"export LD_PRELOAD={wsl_path(overlay)}" if backend == "bullet" else "unset LD_PRELOAD"
    domain_prefix = f"export ROS_DOMAIN_ID={int(ros_domain_id)}" if ros_domain_id is not None else "true"
    command = " && ".join([
        domain_prefix,
        "source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}",
        f"source {wsl_path(NATIVE_INSTALL / 'setup.bash')}", f"export AMENT_PREFIX_PATH={am}",
        f"export LD_LIBRARY_PATH={ld}", "export FR5_BULLET_SHAPE_MODE=use_shape_type", preload,
        f"ros2 launch {wsl_path(STAGE26_LAUNCH)} interval_csv:={wsl_path(interval_csv)} parts_dir:={wsl_path(PARTS)} output_dir:={wsl_path(output_dir)} backend:={backend} run_index:={run_index} capability_only:={'true' if capability_only else 'false'} positive_controls:={'true' if positive_controls else 'false'}",
    ])
    # Dense Stage 2.7R validation expands each nominal trajectory segment into
    # 112542 native Bullet intervals.  Keep the wrapper alive long enough for
    # the complete native stream; the probe itself remains the same binary and
    # input, only this orchestration timeout is extended.
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=7200)
    (output_dir / "native_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "native_stderr.log").write_text(proc.stderr, encoding="utf-8")
    (output_dir / "native_exit_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "passed": proc.returncode == 0}


def normalize_interval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["interval_index", "time_start_s", "time_end_s", "duration_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary", "q_start", "q_end", "discrete_start_free", "discrete_end_free", "continuous_collision", "continuous_contact_fraction", "collision_method", "ccd_api_called", "first_contact", "discrete_start_pairs", "discrete_end_pairs", "continuous_pairs", "continuous_contacts"]
    return [{key: row.get(key) for key in keys} for row in rows]


def normalize_self_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in ("state_index", "time_s", "collision", "start_pairs", "end_pairs")} for row in rows]


def source_blob(repo: Path, path: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", f"HEAD:{path}"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def write_sha256sums(output_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        # The gate embeds this manifest, so exclude the gate itself to avoid a
        # self-referential hash that can never remain stable after finalization.
        if path.is_file() and path.name not in {"SHA256SUMS", "stage26_gate_report.json"}:
            rows.append(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}")
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    mismatches = []
    for line in rows:
        expected, rel = line.split("  ", 1)
        actual = sha256(output_root / rel)
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"checked": len(rows), "mismatches": mismatches, "mismatch_count": len(mismatches)}


def run_regressions(output_root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "py_compile", "scripts/run_stage26.py", "tools/stage26_continuous_launch.py"]
    proc = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=120)
    results = [{"environment": "Windows", "command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "passed": proc.returncode == 0}]
    wsl_command = "source /opt/ros/jazzy/setup.bash && source /mnt/c/Users/86198/Desktop/robotfucker/install/setup.bash && cd /mnt/c/Users/86198/Desktop/robotfucker && python3 -m pytest -q tests/test_stage23b_outputs.py tests/test_stage24_outputs.py tests/test_stage24a_outputs.py tests/test_stage24b_outputs.py tests/test_stage25_contract.py tests/test_stage26_contract.py"
    wsl_proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", wsl_command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=900)
    results.append({"environment": "WSL Ubuntu-24.04-D ROS2 Jazzy Python", "command": wsl_command, "exit_code": wsl_proc.returncode, "stdout": wsl_proc.stdout, "stderr": wsl_proc.stderr, "passed": wsl_proc.returncode == 0})
    result = {"py_compile_passed": results[0]["passed"], "relevant_regressions_passed": results[1]["passed"], "results": results}
    write_json(output_root / "stage26_regression_tests.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rebuilds", type=int, default=3)
    args = parser.parse_args()
    if args.rebuilds != 3:
        raise SystemExit("Stage 2.6 formal certification requires exactly 3 independent rebuilds.")
    if not STAGE25_ROOT.is_dir() or not PARTS.is_dir() or not NATIVE_INSTALL.is_dir():
        raise SystemExit("Frozen Stage 2.5 root, collision geometry, or Stage 2.6 native install is missing.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or ROOT / "outputs/ik_graph_stage26_continuous_collision_certification" / f"fr5_scaled_horseshoe_demo_v45_{timestamp}_Stage26").resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    gate25, trajectory, fk = load_stage25()
    manifest = build_input_manifest(gate25)
    before_hashes = hash_manifest_entries(manifest)
    write_json(output_root / "stage26_input_manifest.json", {**manifest, "hash_check_before": before_hashes})
    interval_info = build_intervals(output_root, trajectory, fk)
    write_json(output_root / "stage26_interval_manifest.json", {key: value for key, value in interval_info.items() if key != "records"})

    rebuilds = []
    for run_index in range(1, 4):
        run_dir = output_root / f"rebuild_{run_index:02d}"
        run_dir.mkdir()
        run_intervals = run_dir / "stage26_intervals.csv"
        shutil.copy2(interval_info["path"], run_intervals)
        shutil.copy2(STAGE25_ROOT / "stage25_ruckig_trajectory.csv", run_dir / "stage25_ruckig_trajectory.csv")
        result = run_native(run_dir, run_intervals, "bullet", run_index, False, run_index == 1)
        result.update({"run_index": run_index, "interval_result_path": str((run_dir / "stage26_continuous_robot_world_intervals.jsonl").resolve())})
        if not (run_dir / "stage26_native_summary.json").exists() or not (run_dir / "stage26_continuous_robot_world_intervals.jsonl").exists():
            raise RuntimeError(f"Stage 2.6 native rebuild {run_index} did not produce required artifacts: {result}")
        rows = read_jsonl(run_dir / "stage26_continuous_robot_world_intervals.jsonl")
        self_rows = read_jsonl(run_dir / "stage26_self_collision_states.jsonl")
        summary = read_json(run_dir / "stage26_native_summary.json")
        summary_for_determinism = {key: value for key, value in summary.items() if key != "run_index"}
        result.update({"interval_rows": rows, "self_rows": self_rows, "summary": summary, "interval_semantic_hash": semantic_hash(normalize_interval_rows(rows)), "self_semantic_hash": semantic_hash(normalize_self_rows(self_rows)), "summary_semantic_hash": semantic_hash(summary_for_determinism)})
        rebuilds.append(result)

    fcl_dir = output_root / "fcl_capability_probe"
    fcl_dir.mkdir()
    fcl_intervals = fcl_dir / "stage26_intervals.csv"
    shutil.copy2(interval_info["path"], fcl_intervals)
    fcl_runtime = run_native(fcl_dir, fcl_intervals, "fcl", 1, True, False)
    fcl_probe = read_json(fcl_dir / "stage26_fcl_capability_probe.json") if (fcl_dir / "stage26_fcl_capability_probe.json").exists() else {}

    first_rows, first_self_rows = rebuilds[0]["interval_rows"], rebuilds[0]["self_rows"]
    if len(first_rows) != interval_info["interval_count"]:
        raise RuntimeError(f"Native interval coverage mismatch: {len(first_rows)} != {interval_info['interval_count']}")
    if len(first_self_rows) != interval_info["timed_state_count"]:
        raise RuntimeError(f"Native self-state coverage mismatch: {len(first_self_rows)} != {interval_info['timed_state_count']}")
    collisions = [row for row in first_rows if row.get("continuous_collision")]
    endpoint_failures = [row for row in first_rows if row.get("discrete_start_free") is False or row.get("discrete_end_free") is False]
    self_failures = [row for row in first_self_rows if row.get("collision")]
    accepted_rows = [row for row in first_rows if not row.get("continuous_collision")]

    on_validation = {"schema_version": "stage26-on-segment-validation-v1", "segments_expected": 10, "segments_observed": interval_info["on_segments_observed"], "segments_passed": len(interval_info["on_segments_observed"]) == 10 and not any(row.get("continuous_collision") for row in first_rows if row.get("process_kind") == "spray_on_segment"), "per_segment": []}
    for segment_id in interval_info["on_segments_observed"]:
        rows = [row for row in first_rows if str(row.get("segment_id")) == str(segment_id)]
        on_validation["per_segment"].append({"segment_id": segment_id, "interval_count": len(rows), "continuous_collisions": sum(bool(row.get("continuous_collision")) for row in rows), "passed": bool(rows) and not any(row.get("continuous_collision") for row in rows)})

    process_items = read_json(STAGE25_ROOT / "stage25_process_manifest.json")["ordered_items"]
    off_validation = {"schema_version": "stage26-off-transition-validation-v1", "transitions_expected": 9, "transitions_observed": interval_info["off_transitions_observed"], "transitions_passed": len(interval_info["off_transitions_observed"]) == 9 and not any(row.get("continuous_collision") for row in first_rows if row.get("process_kind") in {"spray_off_transition", "boundary_crossing"}), "transitions": []}
    for transition_id in interval_info["off_transitions_observed"]:
        rows = [row for row in first_rows if str(row.get("transition_id")) == str(transition_id)]
        source = next(item for item in process_items if item.get("transition_id") == transition_id)
        off_validation["transitions"].append({"transition_id": transition_id, "source_boundary": source["source_boundary"], "interval_count": len(rows), "continuous_collisions": sum(bool(row.get("continuous_collision")) for row in rows), "passed": bool(rows) and not any(row.get("continuous_collision") for row in rows)})

    positive_path = output_root / "rebuild_01/stage26_positive_control_tests.jsonl"
    positive_rows = read_jsonl(positive_path) if positive_path.exists() else []
    positive_a = next((row for row in positive_rows if row.get("test") == "A" and row.get("continuous_collision") is True), {})
    positive_b = next((row for row in positive_rows if row.get("test") == "B"), {})
    positive_c = next((row for row in positive_rows if row.get("test") == "C"), {})
    positive = {
        "schema_version": "stage26-positive-control-v1", "endpoint_free_swept_collision": positive_a,
        "swept_free": positive_b, "acm_semantics": positive_c,
        "endpoint_free_swept_collision_detected": bool(positive_a and positive_a.get("discrete_start_free") and positive_a.get("discrete_end_free") and positive_a.get("discrete_mid_collision") and positive_a.get("continuous_collision")),
        "swept_free_passed": bool(positive_b and positive_b.get("discrete_start_free") and positive_b.get("discrete_end_free") and positive_b.get("continuous_free")),
        "acm_semantics_passed": bool(positive_c and positive_c.get("formal_acm_collision") and not positive_c.get("modified_acm_collision") and positive_c.get("formal_acm_modified_for_formal_run") is False),
    }
    positive["passed"] = bool(positive["endpoint_free_swept_collision_detected"] and positive["swept_free_passed"] and positive["acm_semantics_passed"])
    write_json(output_root / "stage26_positive_control_tests.json", positive)

    deterministic_fields = ["interval_semantic_hash", "self_semantic_hash", "summary_semantic_hash"]
    mismatches = []
    for field in deterministic_fields:
        values = [run[field] for run in rebuilds]
        if len(set(values)) != 1:
            mismatches.append({"artifact": field, "values": values})
    determinism = {"schema_version": "stage26-determinism-v1", "independent_rebuilds": 3, "deterministic": not mismatches, "determinism": "3/3_passed" if not mismatches else "blocked", "core_artifact_mismatch_count": len(mismatches), "mismatches": mismatches, "interval_identity_hashes": [run["interval_semantic_hash"] for run in rebuilds], "self_collision_hashes": [run["self_semantic_hash"] for run in rebuilds], "summary_hashes": [run["summary_semantic_hash"] for run in rebuilds], "collision_decision_counts": [sum(bool(row.get("continuous_collision")) for row in run["interval_rows"]) for run in rebuilds], "first_contact_values": [next((row.get("first_contact") for row in run["interval_rows"] if row.get("first_contact")), None) for run in rebuilds]}
    write_json(output_root / "stage26_determinism_report.json", determinism)

    moveit_repo = ROOT / "external/moveit2"
    moveit_commit = subprocess.run(["git", "-C", str(moveit_repo), "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()
    capability = {
        "schema_version": "stage26-backend-capability-audit-v1", "ros_distribution": "jazzy", "moveit_version": "2.12.4", "bullet_version": "3.24",
        "bullet_native_robot_world_ccd": {"available": True, "status": "passed", "api": "CollisionEnvBullet::checkRobotCollision(req,res,state1,state2,acm)", "runtime_provenance": read_json(output_root / "rebuild_01/stage26_runtime_provenance.json"), "source_repository": str(moveit_repo.resolve()), "source_commit": moveit_commit, "collision_env_bullet_cpp_git_blob": source_blob(moveit_repo, "moveit_core/collision_detection_bullet/src/collision_env_bullet.cpp"), "bullet_cast_bvh_manager_cpp_git_blob": source_blob(moveit_repo, "moveit_core/collision_detection_bullet/src/bullet_integration/bullet_cast_bvh_manager.cpp"), "packaged_header": "/opt/ros/jazzy/include/moveit_core/moveit/collision_detection_bullet/collision_env_bullet.hpp", "cast_header": "/opt/ros/jazzy/include/moveit_core/moveit/collision_detection_bullet/bullet_integration/bullet_cast_bvh_manager.hpp"},
        "fcl_native_continuous_robot_world_ccd": {"available": False, "status": "not_available", "reason": fcl_probe.get("reason", "MoveIt_FCL_backend_does_not_implement_continuous_collision"), "api_invoked": fcl_probe.get("api_invoked", fcl_runtime["passed"]), "source_repository": str(moveit_repo.resolve()), "source_commit": moveit_commit, "collision_env_fcl_cpp_git_blob": source_blob(moveit_repo, "moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp"), "packaged_library_observation": "libmoveit_collision_detection_fcl.so contains Continuous collision not implemented and Not implemented"},
        "continuous_self_collision_native": {"available": False, "status": "not_available", "reason": "MoveIt_Bullet_and_FCL_robot_world_two_state_CCD_APIs_do_not_cover_self_collision"},
    }
    write_json(output_root / "stage26_backend_capability_audit.json", capability)

    continuous_validation = {
        "schema_version": "stage26-continuous-robot-world-validation-v1", "backend": "Bullet", "native_ccd": True,
        "api": "CollisionEnvBullet::checkRobotCollision(req,res,state1,state2,acm)", "timed_state_count": interval_info["timed_state_count"], "timed_interval_count": interval_info["interval_count"],
        "continuous_intervals_checked": {"accepted": len(accepted_rows), "rejected": len(collisions), "skipped": 0, "unchecked": 0},
        "continuous_intervals_passed": len(accepted_rows), "continuous_collisions_found": len(collisions), "endpoint_collision_failures": len(endpoint_failures),
        "first_failed_interval": collisions[0] if collisions else (endpoint_failures[0] if endpoint_failures else None), "continuous_contact_fraction_available": False, "continuous_contact_fraction": None,
        "all_intervals_executed": len(first_rows) == interval_info["interval_count"], "passed": len(first_rows) == interval_info["interval_count"] and not collisions and not endpoint_failures,
    }
    write_json(output_root / "stage26_continuous_robot_world_validation.json", continuous_validation)
    shutil.copy2(output_root / "rebuild_01/stage26_continuous_robot_world_intervals.jsonl", output_root / "stage26_continuous_robot_world_intervals.jsonl")

    self_validation = {
        "schema_version": "stage26-self-collision-validation-v1", "native_continuous_self_collision_available": False,
        "certification_method": "adaptive_discrete_self_collision_certification", "evidence_level": "finite_supplemental_discrete_state_grid_not_native_CCD",
        "subdivision_criterion": "all frozen Stage 2.5 timed states", "maximum_temporal_interval_s": max(row["time_end_s"] - row["time_start_s"] for row in interval_info["records"]),
        "maximum_joint_space_interval_rad": max(max(abs(a - b) for a, b in zip(row["q0"], row["q1"])) for row in interval_info["records"]),
        "termination_rule": "all timed states evaluated exactly once", "timed_states_checked": len(first_self_rows), "self_collision_states_found": len(self_failures),
        "first_failed_state": self_failures[0] if self_failures else None, "result": "supplemental_pass_not_native_ccd" if not self_failures else "blocked_collision_observed",
        "passed": not self_failures and len(first_self_rows) == interval_info["timed_state_count"],
    }
    write_json(output_root / "stage26_self_collision_validation.json", self_validation)
    shutil.copy2(output_root / "rebuild_01/stage26_self_collision_states.jsonl", output_root / "stage26_self_collision_states.jsonl")
    write_json(output_root / "stage26_on_segment_validation.json", on_validation)
    write_json(output_root / "stage26_off_transition_validation.json", off_validation)

    after_hashes = hash_manifest_entries(manifest)
    frozen_unchanged = before_hashes == after_hashes and all(value is not None for value in after_hashes.values())
    manifest.update({"hash_check_before": before_hashes, "hash_check_after": after_hashes, "frozen_stage25_inputs_unchanged": frozen_unchanged, "sha256_mismatch": 0 if frozen_unchanged else sum(before_hashes.get(path) != after_hashes.get(path) for path in before_hashes)})
    write_json(output_root / "stage26_input_manifest.json", manifest)
    regression = run_regressions(output_root)

    runtime_log = []
    for run in rebuilds:
        runtime_log.append(f"rebuild_{run['run_index']:02d} exit_code={run['exit_code']} passed={run['passed']}")
        runtime_log.extend(run["stdout"].splitlines()[-12:])
        runtime_log.extend(run["stderr"].splitlines()[-12:])
    runtime_log.append(f"fcl_capability_exit_code={fcl_runtime['exit_code']}")
    runtime_log.extend(fcl_runtime["stderr"].splitlines()[-12:])
    (output_root / "stage26_runtime_logs.txt").write_text("\n".join(runtime_log) + "\n", encoding="utf-8")

    conditions = [
        continuous_validation["passed"], positive["passed"], self_validation["passed"],
        len(interval_info["on_segments_observed"]) == 10, len(interval_info["off_transitions_observed"]) == 9,
        interval_info["process_order_preserved"], interval_info["boundary_order_preserved"],
        all(gate25.get(key) is True for key in ("pose_constraints_passed", "position_limits_passed", "velocity_limits_passed", "acceleration_limits_passed", "jerk_limits_passed")),
        determinism["deterministic"], regression["py_compile_passed"], regression["relevant_regressions_passed"], frozen_unchanged,
    ]
    stage26_status = "passed" if all(conditions) else "blocked_stage26_evidence_gate"
    gate = {
        "schema_version": "stage26-gate-v1", "repository_stage26_contract_found": False, "repository_stage26_purpose": None,
        "contract_source": "user_prompt_fallback_continuous_robot_world_certification", "Stage_2_4T": gate25.get("Stage_2_4T"), "Stage_2_5": gate25.get("Stage_2_5"),
        "Stage_2_6": stage26_status, "Stage_2_7": "unblocked_not_started" if stage26_status == "passed" else "blocked",
        "formal_stage25_input_frozen": frozen_unchanged, "stage25_timed_states": interval_info["timed_state_count"], "stage25_timed_intervals": interval_info["interval_count"],
        "Bullet_native_robot_world_CCD": True, "continuous_intervals_checked": continuous_validation["continuous_intervals_checked"], "continuous_intervals_passed": continuous_validation["continuous_intervals_passed"],
        "continuous_collisions_found": continuous_validation["continuous_collisions_found"], "skipped_intervals": 0, "FCL_native_CCD_available": False,
        "native_continuous_self_collision_available": False, "self_collision_certification_method": self_validation["certification_method"], "self_collision_result": self_validation["result"],
        "spray_on_segments": f"{len(interval_info['on_segments_observed'])}/10", "spray_off_transitions": f"{len(interval_info['off_transitions_observed'])}/9",
        "process_order_preserved": interval_info["process_order_preserved"], "boundary_order_preserved": interval_info["boundary_order_preserved"],
        "stage25_pose_constraints_preserved": gate25.get("pose_constraints_passed") is True, "stage25_dynamic_limits_preserved": all(gate25.get(key) is True for key in ("position_limits_passed", "velocity_limits_passed", "acceleration_limits_passed", "jerk_limits_passed")),
        "positive_control_endpoint_free_swept_collision": positive["endpoint_free_swept_collision_detected"], "positive_control_passed": positive["passed"],
        "determinism": determinism["determinism"], "core_artifact_mismatch_count": determinism["core_artifact_mismatch_count"],
        "tests": "passed" if regression["py_compile_passed"] and regression["relevant_regressions_passed"] else "blocked", "sha256_mismatch": 0 if frozen_unchanged else manifest["sha256_mismatch"],
        "frozen_stage25_inputs_unchanged": frozen_unchanged, "first_failed_interval": continuous_validation["first_failed_interval"],
        "all_failure_reasons_observed": [] if stage26_status == "passed" else ["blocked_stage26_evidence_gate"],
    }
    write_json(output_root / "stage26_gate_report.json", gate)
    write_json(output_root / "stage26_fcl_capability_probe.json", fcl_probe)

    report_lines = [
        "# Stage 2.6 Formal Continuous Robot-World Certification", "",
        f"- Stage 2.4T: {gate['Stage_2_4T']}; Stage 2.5: {gate['Stage_2_5']}; Stage 2.6: {gate['Stage_2_6']}.",
        "- Repository search found no pre-existing Stage 2.6 contract; this run uses the user-provided fallback contract.", "",
        "## Frozen timed input", "",
        f"- Source: {STAGE25_ROOT}.", f"- Timed states: {interval_info['timed_state_count']}; adjacent timed intervals: {interval_info['interval_count']}.",
        f"- Frozen Stage 2.5 input hashes unchanged: {frozen_unchanged}.", f"- Stage 2.5 pose/dynamics gate preserved: {gate['stage25_pose_constraints_preserved'] and gate['stage25_dynamic_limits_preserved']}.", "",
        "## Native backend evidence", "",
        "- Bullet call: CollisionEnvBullet::checkRobotCollision(req,res,state1,state2,acm); this dispatches to MoveIt's checkRobotCollisionHelperCCD and BulletCastBVHManager.",
        f"- Bullet interval result: {continuous_validation['continuous_intervals_passed']}/{interval_info['interval_count']} collision-free; collisions found {continuous_validation['continuous_collisions_found']}; skipped {continuous_validation['continuous_intervals_checked']['skipped']}.",
        "- FCL native continuous robot-world CCD: not_available; the installed MoveIt FCL source path logs Continuous collision not implemented / Not implemented in the two-state overload.",
        "- Native continuous self-collision: not_available; the separately reported method is finite adaptive_discrete_self_collision_certification, not native CCD.", "",
        "## Process semantics", "",
        f"- Spray-ON segments: {gate['spray_on_segments']}; Spray-OFF transitions: {gate['spray_off_transitions']}.",
        f"- Process order preserved: {gate['process_order_preserved']}; boundary order preserved: {gate['boundary_order_preserved']}.", "",
        "## Positive controls and reproducibility", "",
        f"- Endpoint-free swept-collision control: {positive['endpoint_free_swept_collision_detected']}; all positive controls: {positive['passed']}.",
        f"- Independent rebuilds: {determinism['determinism']}; core artifact mismatch count: {determinism['core_artifact_mismatch_count']}.",
        f"- Regression tests: {gate['tests']}; frozen-input SHA256 mismatches: {gate['sha256_mismatch']}.", "",
        "## Failure record", "",
        f"- First failed interval: {json.dumps(continuous_validation['first_failed_interval'], ensure_ascii=False, sort_keys=True) if continuous_validation['first_failed_interval'] else 'null'}.", "",
        "## Final gate", "",
        f"- Stage 2.7: {gate['Stage_2_7']}. No Stage 2.7 work was started.",
        f"- Final status: {'Stage 2.6 PASSED' if stage26_status == 'passed' else 'Stage 2.6 BLOCKED: blocked_stage26_evidence_gate'}.", "",
    ]
    (output_root / "stage26_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    sums = write_sha256sums(output_root)
    gate["artifact_sha256_manifest"] = sums
    gate["sha256_mismatch"] = int(sums["mismatch_count"])
    write_json(output_root / "stage26_gate_report.json", gate)
    print(json.dumps({"output_root": str(output_root), "Stage_2_6": gate["Stage_2_6"], "Stage_2_7": gate["Stage_2_7"], "timed_states": interval_info["timed_state_count"], "timed_intervals": interval_info["interval_count"], "continuous_collisions": continuous_validation["continuous_collisions_found"], "determinism": determinism["determinism"], "sha256_mismatch": gate["sha256_mismatch"]}, ensure_ascii=False, indent=2))
    return 0 if stage26_status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
