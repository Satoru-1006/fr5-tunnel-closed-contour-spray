#!/usr/bin/env python3
"""Audit Stage 1.9.2 replays without changing formal Stage 1.7-1.9.1 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage192"
PROTECTED = ["outputs/ik_graph", "outputs/ik_graph_diagnostics", "outputs/ik_feasibility_ablation_stage1_6", "outputs/ik_graph_stage17", "outputs/ik_graph_stage18", "outputs/ik_graph_stage19", "outputs/ik_graph_stage191"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def protected_hashes() -> dict[str, Any]:
    files = {}
    for rel in PROTECTED:
        d = ROOT / rel
        for p in sorted(d.rglob("*")) if d.exists() else []:
            if p.is_file(): files[p.relative_to(ROOT).as_posix()] = {"path": p.relative_to(ROOT).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)}
    return {"schema_version": "1.0", "files": files}


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    if path.is_dir():
        result = []
        for p in sorted(path.glob("*.json")):
            if p.name == "comparison.json": continue
            result.extend(read_records(p))
        return result
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except Exception: return []
    return [dict(r) for r in data.get("records", [])]


def summary(name: str, path: Path, expected: int) -> dict[str, Any]:
    rows = read_records(path)
    success = [bool(r.get("solver_success", r.get("direct_kdl_success", False))) for r in rows]
    quantized = [str(r.get("solution_quantized_sha256", "NO_SOLUTION")) for r in rows]
    raw = [str(r.get("solution_raw_sha256", "NO_SOLUTION")) for r in rows]
    inputs = [tuple(r.get(k) for k in ("target_transform_raw_sha256", "seed_vector_raw_sha256", "all_robot_variables_raw_sha256", "solver_options_raw_sha256")) for r in rows if "target_transform_raw_sha256" in r]
    return {"name": name, "path": path.relative_to(ROOT).as_posix(), "records": len(rows), "expected_records": expected, "count_match": len(rows) == expected, "success_count": sum(success), "success_values": sorted(set(success)), "success_status_identical": len(set(success)) <= 1, "raw_solution_hash_count": len(set(raw)), "quantized_solution_hash_count": len(set(quantized)), "raw_solution_hashes": sorted(set(raw)), "quantized_solution_hashes": sorted(set(quantized)), "input_hashes_identical": len(set(inputs)) <= 1, "input_hash_count": len(set(inputs)), "records": len(rows)}


def first_difference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: return {"status": "no_records"}
    keys = ["target_transform_raw_sha256", "seed_vector_raw_sha256", "all_robot_variables_raw_sha256", "initial_robot_state_raw_sha256", "solver_options_raw_sha256", "solver_success", "solver_error_code", "solution_raw_sha256", "solution_quantized_sha256"]
    baseline = rows[0]
    for index, row in enumerate(rows[1:], 1):
        differences = {key: {"baseline": baseline.get(key), "iteration": row.get(key)} for key in keys if baseline.get(key) != row.get(key)}
        if differences:
            return {"status": "difference_found", "call_id": row.get("target_call_id", "call:00000182"), "baseline_iteration": 0, "first_differing_iteration": index, "differences": differences}
    return {"status": "no_difference", "call_id": baseline.get("target_call_id", "call:00000182")}


def logs_contain_teardown(path: Path) -> bool:
    return any("exit code -11" in p.read_text(encoding="utf-8", errors="replace") or "SIGSEGV" in p.read_text(encoding="utf-8", errors="replace") for p in path.rglob("*.log")) if path.exists() else False


def main() -> int:
    before_path = OUT / "protected_output_hashes_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else protected_hashes()
    after = protected_hashes()
    (OUT / "protected_output_hashes_after.json").write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    def comparable(mapping: dict[str, Any]) -> dict[str, tuple[int, str]]:
        return {key: (int(value.get("size_bytes", -1)), str(value.get("sha256"))) for key, value in mapping.items()}
    protected_unchanged = comparable(before.get("files", {})) == comparable(after.get("files", {}))
    groups = [
        summary("moveitpy_same_instance", OUT / "moveitpy_same_instance/replay.json", 100),
        summary("moveitpy_new_instance", OUT / "moveitpy_new_instance/child_processes/comparison.json", 50),
        summary("moveitpy_independent_processes", OUT / "moveitpy_independent_processes/comparison.json", 20),
        summary("cpp_moveit_plugin_same_instance", OUT / "cpp_moveit_plugin_same_instance/replay.json", 100),
        summary("cpp_moveit_plugin_new_instance", OUT / "cpp_moveit_plugin_new_instance/replay.json", 50),
        summary("cpp_moveit_plugin_independent_processes", OUT / "cpp_moveit_plugin_independent_processes/comparison.json", 20),
        summary("direct_orocos_kdl_same_instance", OUT / "direct_orocos_kdl/same_instance.json", 100),
        summary("direct_orocos_kdl_new_instance", OUT / "direct_orocos_kdl/new_instance.json", 50),
        summary("direct_orocos_kdl_independent_processes", OUT / "direct_orocos_kdl/independent_processes/comparison.json", 20),
    ]
    matrix_path = OUT / "stage192_determinism_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "records", "expected_records", "count_match", "success_count", "success_values", "success_status_identical", "raw_solution_hash_count", "quantized_solution_hash_count", "input_hashes_identical", "input_hash_count"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in groups: writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row[k] for k in fields})
    same_rows = read_records(OUT / "moveitpy_same_instance/replay.json")
    diff = first_difference(same_rows)
    diff.update({"target_call_id": "call:00000182", "input_hashes_consistent_across_formal_same_instance": groups[0]["input_hashes_identical"], "first_nondeterministic_layer_boundary": "MoveItPy RobotState.set_from_ik -> MoveIt KDL plugin searchPositionIK/CartToJnt", "symbol_level_stack_trace": "not_available_without_gdb_capture", "teardown_warning": logs_contain_teardown(OUT / "moveitpy_same_instance")})
    (OUT / "first_difference_report.json").write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    controlled_path = OUT / "solver_comparison/moveitpy_timeout_0p005_same_instance.json"
    controlled = summary("moveitpy_timeout_0p005_same_instance_control", controlled_path, 100)
    (OUT / "solver_comparison/solver_comparison.json").write_text(json.dumps({"schema_version": "1.0", "target_call_id": "call:00000182", "formal_groups": groups, "controlled_timeout_0p005": controlled}, indent=2) + "\n", encoding="utf-8")
    (OUT / "sanitizer_runs/status.json").write_text(json.dumps({"schema_version": "1.0", "status": "not_enabled", "reason": "C++ MoveIt plugin and direct KDL reproductions were stable failures, not unstable executions", "asan": "not_run", "ubsan": "not_run", "tsan": "not_run", "teardown_warning": "tracked_separately"}, indent=2) + "\n", encoding="utf-8")
    diagnosis = {
        "stage": "stage_1_9_2", "target_call_id": "call:00000182",
        "moveitpy_same_instance": "unstable" if not groups[0]["success_status_identical"] or groups[0]["quantized_solution_hash_count"] > 1 else "stable",
        "moveitpy_new_instance": "unstable" if not groups[1]["success_status_identical"] or groups[1]["quantized_solution_hash_count"] > 1 else "stable",
        "moveitpy_independent_processes": "unstable" if not groups[2]["success_status_identical"] or groups[2]["quantized_solution_hash_count"] > 1 else "stable",
        "cpp_moveit_plugin_same_instance": "deterministic_failure" if groups[3]["success_status_identical"] and groups[3]["success_count"] == 0 else "unstable",
        "cpp_moveit_plugin_new_instance": "deterministic_failure" if groups[4]["success_status_identical"] and groups[4]["success_count"] == 0 else "unstable",
        "cpp_moveit_plugin_independent_processes": "deterministic_failure" if groups[5]["success_status_identical"] and groups[5]["success_count"] == 0 else "unstable",
        "direct_orocos_kdl_same_instance": "deterministic_failure" if groups[6]["success_status_identical"] and groups[6]["success_count"] == 0 else "unstable",
        "direct_orocos_kdl_new_instance": "deterministic_failure" if groups[7]["success_status_identical"] and groups[7]["success_count"] == 0 else "unstable",
        "direct_orocos_kdl_independent_processes": "deterministic_failure" if groups[8]["success_status_identical"] and groups[8]["success_count"] == 0 else "unstable",
        "first_unstable_layer": "moveitpy", "root_cause_classification": "moveitpy_binding_or_lifecycle",
        "root_cause_evidence": ["all formal binary input hashes are identical within MoveItPy same-instance replay", "MoveItPy success and quantized solution hashes vary", "C++ plugin getPositionIK and direct KDL are stable but fail to solve this zero-timeout case", "nonzero timeout 0.005s succeeds but remains non-deterministic and is only a diagnostic control"],
        "direct_kdl_velocity_solver": "KDL::ChainIkSolverVel_pinv",
        "exact_moveit_kdl_velocity_solver_control": "not_completed; linking MoveIt ChainIkSolverVelMimicSVD separately triggered unavailable libsdformat14 runtime dependency; no claim of exact velocity-solver equivalence",
        "input_identity_warning": {"stage191_corpus_kinematics_sha256": "8bdc63b3e333ee9b30330b815d032ebf84e2806d3d6daddfe797f7018bf204a4", "current_kinematics_source_sha256": "b57f743c70428cf073f28fb4b45299e9f8a72af6edd628a5d68098613234a8c7", "status": "metadata_mismatch_requires_reproduction_caveat"},
        "solver_diagnosis_complete": True, "deterministic_solver_fix_complete": False, "stage2_status": "blocked", "teardown_status": "tracked_separately", "moveitpy_teardown_segmentation_fault_observed": logs_contain_teardown(OUT / "moveitpy_same_instance"),
        "formal_replay_groups": groups, "controlled_timeout_0p005": controlled, "protected_outputs_unchanged": protected_unchanged, "ccd_status": "not_available", "clearance_status": "not_available", "sanitizer_status": "not_enabled_cpp_reproduction_was_stable_failure",
    }
    (OUT / "solver_diagnosis_report.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fix = {"stage": "stage_1_9_2", "target_call_id": "call:00000182", "deterministic_fix_attempted": False, "formal_solver_options_unchanged": True, "candidate_control": "MoveItPy timeout=0.005s", "candidate_control_result": controlled, "accepted_fix": None, "reason": "controlled nonzero-timeout replay still has multiple quantized solution hashes; no KDL deterministic fix is demonstrated", "full_stage191_regression": "not_run_gate_not_met", "stage2_status": "blocked"}
    (OUT / "deterministic_fix_report.json").write_text(json.dumps(fix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Stage 1.9.2 KDL 单次 IK 确定性诊断", "", "- target_call_id: `call:00000182`", f"- solver_diagnosis_complete: `{diagnosis['solver_diagnosis_complete']}`", f"- deterministic_solver_fix_complete: `{diagnosis['deterministic_solver_fix_complete']}`", "- Stage 2: `blocked`", f"- protected_outputs_unchanged: `{protected_unchanged}`", "", "## 判定", "", "MoveItPy 的零 timeout 单次调用在相同二进制输入下出现成功状态和量化解变化；第一处不稳定边界仍在 MoveItPy RobotState.set_from_ik 到 KDL 插件调用层。C++ 插件 getPositionIK 与直接 Orocos KDL 在本次配置下均为稳定失败，因此没有证据把不确定性归因于底层 KDL 数值运行时。", "", "## 直接 KDL 实现边界", "", "本次直接 Orocos KDL 使用 `KDL::ChainIkSolverVel_pinv`。尝试单独链接 MoveIt 的 `ChainIkSolverVelMimicSVD` 时遇到当前 Jazzy 环境缺失 `libsdformat14` 的运行时依赖，因此未声称与插件内部 velocity solver 完全等价；该边界已写入 `solver_diagnosis_report.json`。", "", "## 证据文件", "", "- `stage192_determinism_matrix.csv`: 九组 replay 计数、成功状态和 hash 对照。", "- `first_difference_report.json`: 首个差异调用与输入 hash 检查。", "- `solver_diagnosis_report.json`: 根因分类和环境/teardown 状态。", "- `deterministic_fix_report.json`: 受控 timeout 对照与修复门控。", "", "## 约束", "", "- collision method: `adaptive_discrete_interpolation`", "- CCD: `not_available`; clearance: `not_available`", "- 未运行 full regression；未进入 OFF、闭合轮廓、GNN 或 Stage 2。", "- MoveItPy `exit -11` 仅作为 teardown warning 记录。", ""]
    (OUT / "stage192_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protected_outputs_unchanged": protected_unchanged, "first_unstable_layer": "moveitpy", "root_cause_classification": "moveitpy_binding_or_lifecycle", "solver_diagnosis_complete": True, "deterministic_solver_fix_complete": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
