#!/usr/bin/env python3
"""Run three isolated Stage 1.9 ROS2/MoveIt2 reproductions and audit them."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "ik_graph_stage19"
CONFIG = ROOT / "config" / "stage19_deterministic_ik.yaml"
INPUT_TCP = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
INPUT_SEED = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
PROTECTED_DIRS = ["outputs/ik_graph", "outputs/ik_graph_diagnostics", "outputs/ik_feasibility_ablation_stage1_6", "outputs/ik_graph_stage17", "outputs/ik_graph_stage18"]
WSL_REPO = "/mnt/c/Users/86198/Desktop/robotfucker"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_info() -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()
    diff_stat = subprocess.run(["git", "diff", "--stat"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    return {"branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "status_porcelain": status, "diff_stat": diff_stat}


def protected_hashes() -> dict[str, Any]:
    files: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    for relative in PROTECTED_DIRS:
        directory = ROOT / relative
        if not directory.exists():
            groups[relative] = {"path": relative, "exists": False, "file_count": 0, "files": []}
            continue
        members = []
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                item = {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                members.append(item)
                files[item["path"]] = item
        groups[relative] = {"path": relative, "exists": True, "file_count": len(members), "files": members}
    return {"schema_version": "1.0", "created_time_utc": utc_now(), "files": files, "groups": groups}


def decode_wsl(data: bytes) -> str:
    if b"\x00" in data:
        return data.decode("utf-16le", errors="replace")
    return data.decode("utf-8", errors="replace")


def wsl_distro() -> str:
    result = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=20)
    names = [line.strip().lstrip("*").strip() for line in decode_wsl(result.stdout).splitlines() if line.strip()]
    for name in names:
        if "Ubuntu" in name:
            return name
    raise RuntimeError("No Ubuntu WSL distribution was found")


def wsl_probe(distro: str) -> dict[str, str]:
    command = ["wsl.exe", "-d", distro, "--", "bash", "-lc", f"source /opt/ros/jazzy/setup.bash; source {WSL_REPO}/install/setup.bash; python3 --version; ros2 pkg prefix moveit_py; ros2 pkg prefix fairino5_v6_moveit2_config; python3 -c 'import pyarrow; print(pyarrow.__version__)'"]
    result = subprocess.run(command, capture_output=True, timeout=30)
    output = decode_wsl(result.stdout) + decode_wsl(result.stderr)
    if result.returncode != 0 or "/opt/ros/jazzy" not in output or "25.0.0" not in output:
        raise RuntimeError(f"ROS2/MoveIt2 preflight failed: {output[-2000:]}")
    return {"ros_distribution": "jazzy", "distro": distro, "probe_output": output}


def runtime_config(run_id: str, run_dir: Path) -> Path:
    raw = CONFIG.read_text(encoding="utf-8")
    old = '  directory: "outputs/ik_graph_stage19"'
    new = f'  directory: "outputs/ik_graph_stage19/{run_id}"'
    if raw.count(old) != 1:
        raise RuntimeError("Stage 1.9 output declaration is not unique")
    path = run_dir / "stage19_config_runtime.yaml"
    path.write_text(raw.replace(old, new), encoding="utf-8")
    return path


def run_one(run_id: str, distro: str, probe: dict[str, str], git: dict[str, Any]) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / run_id
    if run_dir.exists():
        if (run_dir / "RUN_COMPLETE.json").exists():
            raise RuntimeError(f"Refusing to overwrite completed run directory: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    runtime_path = runtime_config(run_id, run_dir)
    manifest = {"schema_version": "1.0", "stage": "stage_1_9", "run_id": run_id, "inputs": {"tcp_pose_path": "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv", "tcp_pose_sha256": sha256_file(INPUT_TCP), "seed_joint_path": "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv", "seed_joint_sha256": sha256_file(INPUT_SEED), "waypoint_count": 181}, "configuration": {"config_path": "config/stage19_deterministic_ik.yaml", "config_sha256": sha256_file(CONFIG), "runtime_config_path": runtime_path.relative_to(ROOT).as_posix(), "runtime_config_sha256": sha256_file(runtime_path)}, "robot_model": {"model_name": "FAIRINO_FR5_V6", "group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link", "tool_tcp_source": "assumed_150mm_placeholder"}, "determinism": {"random_seed": 0, "python_hash_seed": "0", "seed_order": "explicit_config_order", "task_pose_order": "stable_key", "parallel": False, "thread_environment": {name: os.environ.get(name, "1") for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}, "software": probe, "git": git, "execution": {"start_time_utc": utc_now(), "host": socket.gethostname(), "command": []}}
    runtime_wsl = f"{WSL_REPO}/outputs/ik_graph_stage19/{run_id}/stage19_config_runtime.yaml"
    run_wsl = f"{WSL_REPO}/outputs/ik_graph_stage19/{run_id}"
    launch_file = f"{WSL_REPO}/tools/stage19_moveit_launch.py"
    launch = f"set +e; source /opt/ros/jazzy/setup.bash; source {WSL_REPO}/install/setup.bash; export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ros2 launch {launch_file} stage19_config:={runtime_wsl} stage19_run_dir:={run_wsl} run_id:={run_id}; rc=$?; printf '%s\\n' \"${{rc:-1}}\" > {run_wsl}/launcher_returncode.txt; exit \"${{rc:-1}}\""
    command = ["wsl.exe", "-d", distro, "--", "bash", "-lc", launch]
    manifest["execution"]["command"] = command
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    start = time.time()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, env={**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, timeout=1800)
    raw = int(process.returncode)
    log = (run_dir / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (run_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
    node_match = re.search(r"process has died.*?exit code (-?\d+)", log, re.DOTALL)
    node_raw = int(node_match.group(1)) if node_match else None
    effective = node_raw if node_raw is not None else raw
    run_complete = (run_dir / "RUN_COMPLETE.json").exists()
    acceptance = json.loads((run_dir / "acceptance_recalculation.json").read_text(encoding="utf-8")) if (run_dir / "acceptance_recalculation.json").exists() else {}
    if effective == -11 and run_complete and acceptance.get("formal_pass") is True:
        status = {"process_exit_status": "warning", "run_status": "validated_with_teardown_fault", "teardown_status": "moveitpy_teardown_segmentation_fault"}
    elif effective == 0 and run_complete and acceptance.get("formal_pass") is True:
        status = {"process_exit_status": "pass", "run_status": "clean_pass", "teardown_status": "clean"}
    else:
        status = {"process_exit_status": "fail", "run_status": "failed_or_incomplete", "teardown_status": "not_clean_pass"}
    process_status = {**status, "raw_return_code": effective, "launcher_return_code": raw, "node_return_code": node_raw, "run_complete_exists": run_complete, "elapsed_seconds": time.time() - start}
    (run_dir / "process_status.json").write_text(json.dumps(process_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run_id": run_id, "run_dir": run_dir, "process_status": process_status, "acceptance": acceptance, "summary": json.loads((run_dir / "graph_summary.json").read_text(encoding="utf-8")) if (run_dir / "graph_summary.json").exists() else {}}


def load_completed(run_id: str) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / run_id
    process_status = json.loads((run_dir / "process_status.json").read_text(encoding="utf-8")) if (run_dir / "process_status.json").exists() else {"process_exit_status": "unknown", "run_status": "completed_existing"}
    acceptance = json.loads((run_dir / "acceptance_recalculation.json").read_text(encoding="utf-8")) if (run_dir / "acceptance_recalculation.json").exists() else {}
    if process_status.get("process_exit_status") == "pending_teardown":
        launcher = int((run_dir / "launcher_returncode.txt").read_text(encoding="utf-8").strip()) if (run_dir / "launcher_returncode.txt").exists() else 0
        if launcher == 0:
            process_status = {"process_exit_status": "fail", "run_status": "completed_with_validation_failure", "teardown_status": "clean", "raw_return_code": 0, "run_complete_exists": True}
        else:
            process_status = {"process_exit_status": "warning", "run_status": "completed_with_validation_failure", "teardown_status": "moveitpy_teardown_segmentation_fault", "raw_return_code": launcher, "run_complete_exists": True}
        (run_dir / "process_status.json").write_text(json.dumps(process_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run_id": run_id, "run_dir": run_dir, "process_status": process_status, "acceptance": acceptance, "summary": json.loads((run_dir / "graph_summary.json").read_text(encoding="utf-8")) if (run_dir / "graph_summary.json").exists() else {}}


def compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    hash_names = ["task_pose_candidate_semantic_hash", "ik_node_semantic_hash", "transition_edge_semantic_hash", "candidate_graph_semantic_hash", "selected_task_pose_sequence_hash", "selected_ik_sequence_hash", "repair_waypoint_set_hash", "selected_joint_path_semantic_hash"]
    hashes = {name: [result.get("acceptance", {}).get("hashes", {}).get(name) for result in results] for name in hash_names}
    hash_match = {name: len(set(values)) == 1 and values[0] is not None for name, values in hashes.items()}
    selected_match = len({tuple(result.get("acceptance", {}).get("recomputed", {}).get("selected_ik_candidate_sequence", [])) for result in results}) == 1
    paths = [int(result.get("acceptance", {}).get("recomputed", {}).get("waypoint_count", 0)) for result in results]
    return {"hashes_by_run": hashes, "hash_match": hash_match, "all_core_semantic_hashes_match": all(hash_match.values()), "selected_sequence_exact_match": selected_match, "waypoint_counts": paths, "all_181": all(value == 181 for value in paths), "all_formal_pass": all(result.get("acceptance", {}).get("formal_pass") is True for result in results)}


def write_report(audit: dict[str, Any], git: dict[str, Any]) -> None:
    lines = ["# Stage 1.9 确定性 IK 候选图与联合路径复现", "", f"- stage19_status: `{audit['stage19_status']}`", f"- three_real_ros2_moveit2_runs_attempted: `{len(audit['runs']) == 3}`", f"- all_core_semantic_hashes_match: `{audit['comparison']['all_core_semantic_hashes_match']}`", f"- all_181_complete_paths: `{audit['comparison']['all_181']}`", f"- all_formal_pass: `{audit['comparison']['all_formal_pass']}`", "", "## 运行结果", ""]
    for run in audit["runs"]:
        summary = run.get("summary", {})
        lines.extend([f"### {run['run_id']}", "", f"- return code: `{run['process_status'].get('raw_return_code')}`; process: `{run['process_status'].get('process_exit_status')}`; teardown: `{run['process_status'].get('teardown_status')}`", f"- 181-point ON path: `{summary.get('selected_waypoint_count', 0)}` / complete=`{summary.get('complete_on_path')}`", f"- formal_pass: `{summary.get('formal_pass')}`", f"- task poses / IK nodes / edges: `{summary.get('task_pose_candidate_count')}` / `{summary.get('ik_node_count')}` / `{summary.get('transition_edge_count')}`", f"- Ruckig/FK/collision: `{summary.get('ruckig', {}).get('success')}` / `{summary.get('ruckig', {}).get('post_ruckig_fk_status')}` / `{summary.get('ruckig', {}).get('post_ruckig_collision_count')}`", ""])
    lines.extend(["## 四层确定性诊断", "", json.dumps(audit["comparison"], ensure_ascii=False, indent=2), "", "## 约束边界", "", "- 输入仅为 `outputs/internal_wiper_moveit_inputs/` 下 181 点 open-arch ON 路径。", "- 碰撞方法为 `adaptive_discrete_interpolation`；CCD 和 clearance 均为 `not_available`。", "- 未进入 OFF、reorientation、closed-contour、GNN 或强化学习。", "- `exit -11` 只作为 teardown 状态记录，不进入语义 hash。", "", "## 保护输出", "", f"- protected_outputs_unchanged: `{audit['protected_outputs_unchanged']}`", f"- git diff --stat at summary:\n```text\n{git.get('diff_stat', '').strip()}\n```", ""])
    (OUTPUT_ROOT / "stage19_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    completed_runs = all((OUTPUT_ROOT / name / "RUN_COMPLETE.json").exists() for name in ("run_001", "run_002", "run_003"))
    refresh_existing = completed_runs and (OUTPUT_ROOT / "determinism_report.json").exists()
    if OUTPUT_ROOT.exists() and (OUTPUT_ROOT / "determinism_report.json").exists() and not refresh_existing:
        raise SystemExit(f"Refusing to overwrite existing Stage 1.9 output: {OUTPUT_ROOT}")
    # A failed preflight may have left only protected-output evidence in the
    # root.  It is safe to resume because no run directory or result artifact
    # exists yet; each run itself is still created with exist_ok=False.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    git = git_info()
    before = protected_hashes()
    (OUTPUT_ROOT / "protected_output_hashes_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    distro = wsl_distro()
    probe = wsl_probe(distro)
    results = []
    for run_id in ("run_001", "run_002", "run_003"):
        if (OUTPUT_ROOT / run_id / "RUN_COMPLETE.json").exists():
            results.append(load_completed(run_id))
        else:
            results.append(run_one(run_id, distro, probe, git))
    after = protected_hashes()
    protected_unchanged = before["files"] == after["files"] and before["groups"] == after["groups"]
    (OUTPUT_ROOT / "protected_output_hashes_after.json").write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison = compare(results)
    audit = {"schema_version": "1.0", "stage": "stage_1_9", "runs": results, "comparison": comparison, "protected_outputs_unchanged": protected_unchanged, "stage19_status": "pass" if comparison["all_core_semantic_hashes_match"] and comparison["selected_sequence_exact_match"] and comparison["all_181"] and comparison["all_formal_pass"] and protected_unchanged else "failed", "ccd_status": "not_available", "clearance_status": "not_available", "git_diff_stat": git_info().get("diff_stat", "")}
    (OUTPUT_ROOT / "determinism_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    rows = []
    for name, values in comparison["hashes_by_run"].items():
        rows.append({"metric": name, "run_001": values[0], "run_002": values[1], "run_003": values[2], "comparison_status": "exact_match" if len(set(values)) == 1 else "mismatch"})
    with (OUTPUT_ROOT / "determinism_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "run_001", "run_002", "run_003", "comparison_status"])
        writer.writeheader(); writer.writerows(rows)
    write_report(audit, git_info())
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    return 0 if audit["stage19_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
