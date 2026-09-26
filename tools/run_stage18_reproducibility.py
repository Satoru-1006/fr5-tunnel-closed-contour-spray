#!/usr/bin/env python3
"""Run and summarize two isolated Stage 1.7 reproductions for Stage 1.8."""

from __future__ import annotations

import csv
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

try:
    from .audit_stage17_reproducibility import FLOAT_ABS_TOL, FLOAT_REL_TOL, audit_run, file_manifest, sha256_file
except ImportError:  # script execution from the tools directory
    from audit_stage17_reproducibility import FLOAT_ABS_TOL, FLOAT_REL_TOL, audit_run, file_manifest, sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "ik_graph_stage18"
STAGE17_CONFIG = ROOT / "config" / "stage17_pose_repair.yaml"
INPUT_TCP = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
INPUT_SEED = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv"
WSL_REPO = "/mnt/c/Users/86198/Desktop/robotfucker"
WSL_DISTRO = "Ubuntu-24.04-D"

PROTECTED_DIRS = [
    "outputs/ik_graph",
    "outputs/ik_graph_diagnostics",
    "outputs/ik_feasibility_ablation_stage1_6",
    "outputs/ik_graph_stage17",
]
MODEL_FILES = {
    "urdf": "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro",
    "srdf": "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
    "kinematics": "external/frcobot_ros2/fairino5_v6_moveit2_config/config/kinematics.yaml",
    "joint_limits": "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml",
}
STAGE17_CODE_FILES = [
    "tools/run_stage17_pose_repair.py",
    "src/task_pose_repair.py",
    "src/ik_candidate_graph.py",
    "ros2_moveit_bridge/plan_closed_contour_moveit.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_capture(command: list[str], *, cwd: Path = ROOT, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", shell=False, timeout=timeout)


def git_info() -> dict[str, Any]:
    status = run_capture(["git", "status", "--short"]).stdout.splitlines()
    diff_stat = run_capture(["git", "diff", "--stat"]).stdout
    branch = run_capture(["git", "branch", "--show-current"]).stdout.strip()
    commit = run_capture(["git", "rev-parse", "HEAD"]).stdout.strip()
    return {"commit": commit, "branch": branch, "status_porcelain": status, "diff_stat": diff_stat}


def hash_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        return {"path": relative, "exists": False, "sha256": None, "size_bytes": None}
    return {"path": relative, "exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def protected_hashes() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative in PROTECTED_DIRS:
        directory = ROOT / relative
        if not directory.exists():
            files[relative] = {"path": relative, "exists": False, "sha256": None, "size_bytes": None, "file_count": 0}
            continue
        members = [hash_record(path.relative_to(ROOT).as_posix()) for path in sorted(directory.rglob("*")) if path.is_file()]
        files[relative] = {"path": relative, "exists": True, "file_count": len(members), "files": members}
    flattened = {record["path"]: record for group in files.values() for record in (group["files"] if "files" in group else [group])}
    return {"schema_version": "1.0", "created_time_utc": utc_now(), "files": flattened, "groups": files}


def _decode_wsl_output(data: bytes) -> str:
    if b"\x00" in data:
        try:
            return data.decode("utf-16le", errors="replace")
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def detect_wsl_distribution() -> str:
    result = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, shell=False, timeout=20)
    names = [line.strip().lstrip("*").strip() for line in _decode_wsl_output(result.stdout).splitlines() if line.strip()]
    for name in names:
        if "Ubuntu" in name:
            return name
    return WSL_DISTRO


def wsl_query(distro: str) -> dict[str, str]:
    code = "import sys; import yaml, numpy, pyarrow; print('PYTHON=' + sys.version.split()[0]); print('PYARROW=' + pyarrow.__version__); print('NUMPY=' + numpy.__version__); print('YAML=' + yaml.__version__)"
    command = ["wsl.exe", "-d", distro, "--", "bash", "-lc", f"source /opt/ros/jazzy/setup.bash && source {WSL_REPO}/install/setup.bash && python3 -c \"{code}\" && ros2 pkg prefix moveit_py"]
    result = subprocess.run(command, capture_output=True, shell=False, timeout=30)
    text = _decode_wsl_output(result.stdout)
    values = {key: value for key, value in re.findall(r"(?m)^(PYTHON|PYARROW|NUMPY|YAML)=(.+)$", text)}
    values.update({"ros_distribution": "jazzy", "moveit_version": next((line.strip() for line in text.splitlines() if "/moveit_py" in line), "installed_moveit_py")})
    return values


def runtime_config(run_id: str, run_dir: Path) -> Path:
    text = STAGE17_CONFIG.read_text(encoding="utf-8")
    old = '  directory: "outputs/ik_graph_stage17"'
    new = f'  directory: "outputs/ik_graph_stage18/{run_id}"'
    if text.count(old) != 1:
        raise RuntimeError("Stage 1.7 output directory declaration was not uniquely found")
    path = run_dir / "stage17_config_runtime.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def manifest_for_run(run_id: str, run_dir: Path, distro: str, software: dict[str, str], git: dict[str, Any]) -> dict[str, Any]:
    runtime_path = runtime_config(run_id, run_dir)
    code_hashes = {relative: hash_record(relative) for relative in STAGE17_CODE_FILES}
    model_hashes = {key: hash_record(relative) for key, relative in MODEL_FILES.items()}
    return {
        "schema_version": "1.0",
        "stage": "stage_1_8_reproduction",
        "run_id": run_id,
        "git": git,
        "inputs": {
            "tcp_pose_path": "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv",
            "tcp_pose_sha256": sha256_file(INPUT_TCP),
            "seed_joint_path": "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv",
            "seed_joint_sha256": sha256_file(INPUT_SEED),
            "waypoint_count": 181,
        },
        "configuration": {
            "stage17_config_path": "config/stage17_pose_repair.yaml",
            "stage17_config_sha256": sha256_file(STAGE17_CONFIG),
            "runtime_config_path": (run_dir / "stage17_config_runtime.yaml").relative_to(ROOT).as_posix(),
            "runtime_config_sha256": sha256_file(runtime_path),
            "output_directory_override_only": True,
            "formal_constraints_locked_before_run": True,
            "stage17_code_sha256": code_hashes,
        },
        "robot_model": {
            "model_name": "FAIRINO_FR5_V6",
            "group_name": "fairino5_v6_group",
            "ee_link": "spray_tcp_link",
            "urdf_path": MODEL_FILES["urdf"],
            "urdf_sha256": model_hashes["urdf"]["sha256"],
            "srdf_path": MODEL_FILES["srdf"],
            "srdf_sha256": model_hashes["srdf"]["sha256"],
            "kinematics_path": MODEL_FILES["kinematics"],
            "kinematics_sha256": model_hashes["kinematics"]["sha256"],
            "joint_limits_path": MODEL_FILES["joint_limits"],
            "joint_limits_sha256": model_hashes["joint_limits"]["sha256"],
            "tool_tcp_source": "assumed_150mm_placeholder",
        },
        "software": {"python_version": software.get("PYTHON", "unknown"), "ros_distribution": software.get("ros_distribution", "jazzy"), "moveit_version": software.get("moveit_version", "unknown"), "ruckig_version": "MoveIt2 native Ruckig", "pyarrow_version": software.get("PYARROW", "unknown"), "numpy_version": software.get("NUMPY", "unknown")},
        "determinism": {"random_seed": 0, "python_hash_seed": "0", "seed_generation_order": "stage17 source order", "roll_candidate_order": "config order", "task_pose_sort_order": "stage17 source order", "ik_candidate_sort_order": "stage17 source order", "dp_tie_break": "strictly lower cost retains first encountered", "thread_environment": {key: os.environ.get(key, "1") for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "execution": {
            "command": [],
            "start_time_utc": utc_now(),
            "host": socket.gethostname(),
            "wsl_distribution": distro,
            "source_commands": ["source /opt/ros/jazzy/setup.bash", f"source {WSL_REPO}/install/setup.bash"],
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one(run_id: str, distro: str, software: dict[str, str], git: dict[str, Any]) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / run_id
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = manifest_for_run(run_id, run_dir, distro, software, git)
    runtime_wsl = f"{WSL_REPO}/outputs/ik_graph_stage18/{run_id}/stage17_config_runtime.yaml"
    run_wsl = f"{WSL_REPO}/outputs/ik_graph_stage18/{run_id}"
    launch_file = f"{WSL_REPO}/tools/stage18_moveit_launch.py"
    launch_command = f"set +e; source /opt/ros/jazzy/setup.bash; source {WSL_REPO}/install/setup.bash; export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ros2 launch {launch_file} stage17_config:={runtime_wsl} stage18_run_dir:={run_wsl} run_id:={run_id}; rc=$?; printf '%s\\n' \"${{rc:-1}}\" > {run_wsl}/launcher_returncode.txt; exit \"${{rc:-1}}\""
    command = ["wsl.exe", "-d", distro, "--", "bash", "-lc", launch_command]
    manifest["execution"]["command"] = command
    _write_json(run_dir / "run_manifest.json", manifest)
    start = time.time()
    start_utc = utc_now()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=stdout, stderr=stderr, shell=False, env=os.environ.copy())
        returncode = process.wait()
    end_utc = utc_now()
    launcher_file = run_dir / "launcher_returncode.txt"
    try:
        launcher_raw = int(launcher_file.read_text(encoding="utf-8").strip()) if launcher_file.exists() else int(returncode)
    except ValueError:
        launcher_raw = int(returncode)
    combined_log = (run_dir / "stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (run_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
    node_exit_match = re.search(r"process has died.*?exit code (-?\d+)", combined_log, flags=re.DOTALL)
    node_raw = int(node_exit_match.group(1)) if node_exit_match else None
    effective_raw = node_raw if node_raw is not None else launcher_raw
    lifecycle_path = run_dir / "lifecycle.log"
    lifecycle = []
    if lifecycle_path.exists():
        lifecycle = [json.loads(line) for line in lifecycle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    last_phase = lifecycle[-1].get("phase") if lifecycle else None
    try:
        from .audit_stage17_reproducibility import classify_process_returncode, teardown_warning_allowed
    except ImportError:
        from audit_stage17_reproducibility import classify_process_returncode, teardown_warning_allowed

    process_status = classify_process_returncode(effective_raw)
    run_complete_exists = (run_dir / "RUN_COMPLETE.json").exists()
    acceptance = json.loads((run_dir / "acceptance_recalculation.json").read_text(encoding="utf-8")) if (run_dir / "acceptance_recalculation.json").exists() else {}
    validation_pass = acceptance.get("overall_status") == "pass" and json.loads((run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8")).get("run_complete") is True if run_complete_exists else False
    if teardown_warning_allowed(process_status, run_complete_exists=run_complete_exists, validation_pass=validation_pass, last_lifecycle_phase=last_phase):
        process_status.update({"process_exit_status": "warning", "run_status": "validated_with_teardown_fault", "teardown_status": "moveitpy_teardown_segmentation_fault"})
    elif process_status.get("process_signal") == "SIGSEGV":
        process_status.update({"process_exit_status": "fail", "run_status": "failed_or_incomplete", "teardown_status": "segmentation_fault_stage_unconfirmed"})
    elif process_status.get("process_returncode_raw") == 0 and run_complete_exists and validation_pass:
        process_status.update({"process_exit_status": "pass", "run_status": "clean_pass", "teardown_status": "clean"})
    else:
        process_status.update({"process_exit_status": "fail", "run_status": "failed_or_incomplete", "teardown_status": "not_clean_pass"})
    process_status.update({"start_time_utc": start_utc, "end_time_utc": end_utc, "elapsed_seconds": time.time() - start, "launcher_returncode_raw": launcher_raw, "node_returncode_raw": node_raw, "process_returncode_raw": effective_raw, "observed_parent_returncode": returncode, "run_complete_exists": run_complete_exists, "last_lifecycle_phase": last_phase, "lifecycle_phases": [item.get("phase") for item in lifecycle]})
    _write_json(run_dir / "process_status.json", process_status)
    _write_json(run_dir / "file_sha256_manifest.json", {"schema_version": "1.0", "files": file_manifest(run_dir, exclude=("file_sha256_manifest.json",))})
    return {"run_id": run_id, "run_dir": run_dir, "process_status": process_status, "acceptance": acceptance, "manifest": manifest}


def compare_runs(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_specs = [
        ("waypoint_count", "recomputed", True), ("modified_waypoint_count", "recomputed", True), ("modified_waypoint_ids", "recomputed", True), ("max_position_offset_mm", "recomputed", False), ("mean_position_offset_mm", "recomputed", False), ("max_standoff_offset_mm", "recomputed", False), ("max_roll_offset_deg", "recomputed", False), ("max_normal_error_deg", "recomputed", False), ("max_joint_step_deg", "recomputed", False), ("total_joint_motion_rad", "recomputed", False), ("total_cost", "recomputed", False), ("sample_count", "ruckig_recomputed", True), ("max_velocity_ratio", "ruckig_recomputed", False), ("max_acceleration_ratio", "ruckig_recomputed", False), ("max_jerk_ratio", "ruckig_recomputed", False), ("post_ruckig_collision_count", "ruckig_recomputed", False), ("selected_task_pose_candidate_sequence", "recomputed", True), ("selected_ik_candidate_sequence", "recomputed", True), ("selected_joint_path", "recomputed", False),
    ]
    rows: list[dict[str, Any]] = []
    for name, section, exact in metric_specs:
        left = results[0]["acceptance"].get(section, {}).get(name)
        right = results[1]["acceptance"].get(section, {}).get(name)
        row = {"metric": name, "run_001": json.dumps(left, ensure_ascii=False) if isinstance(left, (list, dict)) else left, "run_002": json.dumps(right, ensure_ascii=False) if isinstance(right, (list, dict)) else right, "absolute_difference": "", "relative_difference": "", "comparison_status": "missing_source_data"}
        if left is not None and right is not None:
            if exact or isinstance(left, (list, dict)) or isinstance(right, (list, dict)):
                row["comparison_status"] = "exact_match" if left == right else "mismatch"
            else:
                difference = abs(float(left) - float(right))
                tolerance = FLOAT_ABS_TOL + FLOAT_REL_TOL * abs(float(right))
                row["absolute_difference"] = difference
                row["relative_difference"] = difference / abs(float(right)) if float(right) != 0 else (0.0 if difference == 0 else None)
                row["comparison_status"] = "exact_match" if difference == 0 else ("within_numeric_tolerance" if difference <= tolerance else "mismatch")
        rows.append(row)
    candidate_status = next(row["comparison_status"] for row in rows if row["metric"] == "selected_ik_candidate_sequence")
    cost_row = next(row for row in rows if row["metric"] == "total_cost")
    validation_equivalent = all(result["acceptance"].get("overall_status") == "pass" for result in results)
    if candidate_status == "mismatch" and validation_equivalent and cost_row["comparison_status"] in {"exact_match", "within_numeric_tolerance"}:
        candidate_status = "equivalent_optimum"
    summary = {"candidate_sequence_status": candidate_status, "trajectory_metrics_status": "within_numeric_tolerance" if all(row["comparison_status"] in {"exact_match", "within_numeric_tolerance"} for row in rows) else "mismatch", "both_runs_formal_pass": all(result["acceptance"].get("formal_constraints_pass") is True for result in results), "both_runs_post_ruckig_pass": all(result["acceptance"].get("post_ruckig_validation_status") == "pass" for result in results), "both_runs_collision_free_under_declared_method": all(result["acceptance"].get("collision_validation_pass") is True for result in results)}
    return rows, summary


def write_summary(results: list[dict[str, Any]], protected_before: dict[str, Any], protected_after: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    rows, comparison = compare_runs(results)
    with (OUTPUT_ROOT / "reproducibility_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "run_001", "run_002", "absolute_difference", "relative_difference", "comparison_status"])
        writer.writeheader()
        writer.writerows(rows)
    protected_unchanged = protected_before.get("files") == protected_after.get("files")
    all_validated = comparison["both_runs_formal_pass"] and comparison["both_runs_post_ruckig_pass"] and comparison["both_runs_collision_free_under_declared_method"] and protected_unchanged and comparison["candidate_sequence_status"] in {"exact_match", "equivalent_optimum"}
    statuses = [result["process_status"].get("process_exit_status") for result in results]
    if all_validated and all(status == "pass" for status in statuses):
        stage_status = "clean_pass"
    elif all_validated and all(status in {"pass", "warning"} for status in statuses) and any(status == "warning" for status in statuses):
        stage_status = "pass_with_teardown_warning"
    else:
        stage_status = "failed"
    audit = {"schema_version": "1.0", "stage": "stage_1_8", "runs": [{"run_id": result["run_id"], "trajectory_validation_status": result["acceptance"].get("trajectory_validation_status", "fail"), "post_ruckig_validation_status": result["acceptance"].get("post_ruckig_validation_status", "fail"), "artifact_integrity_status": result["acceptance"].get("artifact_integrity_status", "fail"), "process_exit_status": result["process_status"].get("process_exit_status"), "run_status": result["process_status"].get("run_status")} for result in results], "provenance_match": results[0]["manifest"]["inputs"]["tcp_pose_sha256"] == results[1]["manifest"]["inputs"]["tcp_pose_sha256"] and results[0]["manifest"]["configuration"]["stage17_code_sha256"] == results[1]["manifest"]["configuration"]["stage17_code_sha256"], "protected_outputs_unchanged": protected_unchanged, **comparison, "ccd_status": "not_available", "clearance_status": "not_available", "stage18_status": stage_status, "git_diff_stat_at_summary": git.get("diff_stat", "")}
    _write_json(OUTPUT_ROOT / "reproducibility_audit.json", audit)
    lines = ["# Stage 1.8 可重复性复现报告", "", f"- 最终状态: `{stage_status}`", f"- 两次候选序列: `{comparison['candidate_sequence_status']}`", f"- 轨迹数值指标: `{comparison['trajectory_metrics_status']}`", f"- 受保护输出未变化: `{protected_unchanged}`", "", "## 运行状态", ""]
    for result in results:
        acceptance = result["acceptance"]
        recomputed = acceptance.get("recomputed", {})
        ruckig = acceptance.get("ruckig_recomputed", {})
        lines.extend([f"### {result['run_id']}", "", f"- process returncode raw: `{result['process_status'].get('process_returncode_raw')}`; launcher returncode: `{result['process_status'].get('launcher_returncode_raw')}`", f"- process status: `{result['process_status'].get('process_exit_status')}`; run status: `{result['process_status'].get('run_status')}`", f"- RUN_COMPLETE before teardown: `{result['process_status'].get('run_complete_exists')}`", f"- last lifecycle phase: `{result['process_status'].get('last_lifecycle_phase')}`", f"- 181-point path: `{recomputed.get('waypoint_count')}`; formal constraints: `{acceptance.get('formal_constraints_pass')}`", f"- modified waypoint IDs: `{recomputed.get('modified_waypoint_ids')}`", f"- max joint step deg: `{recomputed.get('max_joint_step_deg')}`", f"- max/mean TCP offset mm: `{recomputed.get('max_position_offset_mm')}` / `{recomputed.get('mean_position_offset_mm')}`", f"- max roll / standoff offset: `{recomputed.get('max_roll_offset_deg')}` / `{recomputed.get('max_standoff_offset_mm')}`", f"- total joint motion / total cost: `{recomputed.get('total_joint_motion_rad')}` / `{recomputed.get('total_cost')}`", f"- Ruckig samples: `{ruckig.get('sample_count')}`; velocity/acceleration/jerk ratios: `{ruckig.get('max_velocity_ratio')}` / `{ruckig.get('max_acceleration_ratio')}` / `{ruckig.get('max_jerk_ratio')}`", f"- post-Ruckig collision count / FK: `{ruckig.get('post_ruckig_collision_count')}` / `{ruckig.get('fk_status')}`", f"- CCD / clearance: `not_available` / `not_available`", ""])
    lines.extend(["## 摘要与底层重算", "", "每个指标的逐运行审计状态见各运行目录的 `acceptance_recalculation.json`，两次比较见 `reproducibility_metrics.csv`。Stage 1.7 摘要仅用于对比，正式验收值来自底层文件独立重算。", "", "## 保护哈希", "", f"- before/after 文件清单一致: `{protected_unchanged}`", f"- git diff --stat: `{git.get('diff_stat', '').strip()}`", "", "## 未解决问题", "", "MoveItPy/MoveItCpp 清理阶段仍需以 `lifecycle.log` 和 `process_status.json` 为准；若出现 SIGSEGV，只有在 RUN_COMPLETE 已原子写入且最后阶段为 `moveitpy_destructor_pending` 时才允许标记为 teardown warning。", ""])
    (OUTPUT_ROOT / "stage18_report.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def main() -> int:
    if OUTPUT_ROOT.exists() and any((OUTPUT_ROOT / name).exists() for name in ("run_001", "run_002", "reproducibility_audit.json")):
        raise SystemExit(f"Refusing to overwrite existing Stage 1.8 output: {OUTPUT_ROOT}")
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    git = git_info()
    protected_before = protected_hashes()
    _write_json(OUTPUT_ROOT / "protected_output_hashes_before.json", protected_before)
    distro = detect_wsl_distribution()
    software = wsl_query(distro)
    results = [run_one(run_id, distro, software, git) for run_id in ("run_001", "run_002")]
    protected_after = protected_hashes()
    _write_json(OUTPUT_ROOT / "protected_output_hashes_after.json", protected_after)
    audit = write_summary(results, protected_before, protected_after, git_info())
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit.get("stage18_status") in {"clean_pass", "pass_with_teardown_warning"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
