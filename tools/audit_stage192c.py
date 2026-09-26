#!/usr/bin/env python3
"""Build the evidence-bound Stage 1.9.2c report from real replay outputs.

This tool never upgrades a replay result into a PlanningScene, FK, collision,
graph, dynamics, or Ruckig result. Those layers remain explicitly unavailable
when the direct KDL replay executable did not run them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ik_graph_stage192c"
B = ROOT / "outputs" / "ik_graph_stage192b"

POLICIES = [
    "P0_disabled",
    "P1_exact_replay",
    "P2_fixed_prng",
    "P3_axis_basis",
    "P4_multi_joint_templates",
    "P5_jacobian_svd",
    "P6_hybrid",
]
DIFFICULT = [67, 68, 87, 94]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = load_json(path)
    if isinstance(value.get("records"), list):
        return value["records"]
    return []


def successful(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in recs if r.get("solver_success") is True and r.get("solution")]


def result_summary(files: list[Path]) -> dict[str, Any]:
    recs = [r for path in files for r in records(path)]
    sols = successful(recs)
    hashes = sorted({r.get("solution_hash") for r in sols if r.get("solution_hash")})
    spread = None
    if len(sols) > 1:
        vectors = [[float(x) for x in r["solution"]] for r in sols]
        spread = max(
            math.dist(a, b) for i, a in enumerate(vectors) for b in vectors[i + 1 :]
        )
    return {
        "invocation_count": len(recs),
        "successful_invocations": len(sols),
        "success_rate": len(sols) / len(recs) if recs else None,
        "raw_candidates": len(sols),
        "candidates_within_joint_limits": "not_available_postcheck_not_run",
        "candidates_after_pose_check": len(sols),
        "candidates_after_collision_callback": "not_available_callback_not_run",
        "candidates_after_deduplication": "not_run_gate_not_met",
        "graph_candidates_inserted": "not_run_gate_not_met",
        "exact_solution_hash_count": len(hashes),
        "canonical_solution_set_hash_count": "not_run_gate_not_met",
        "max_joint_spread": spread,
        "median_iterations": None,
        "p95_iterations": None,
        "median_cpu_time_us": None,
        "p95_cpu_time_us": None,
        "solution_hashes": hashes,
    }


def parse_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        provenance = row.get("stage192c_provenance") or ""
        parsed: dict[str, Any] = {
            "call_id": row.get("call_id"),
            "policy_id": row.get("singularity_escape_policy"),
            "attempt_id": row.get("attempt_id"),
            "wiggle_event_index": row.get("wiggle_event_index"),
            "iteration": row.get("iteration_id"),
            "wiggle_condition_reached": row.get("wiggle_condition_reached"),
            "wiggle_triggered": row.get("wiggle_triggered"),
            "delta_twist_norm": row.get("delta_twist_norm"),
            "last_delta_twist_norm": row.get("last_delta_twist_norm"),
            "q_before_wiggle": row.get("q_before_wiggle"),
            "raw_direction": row.get("random_vector_raw"),
            "delta_q_after_scale": row.get("delta_q_after_scale"),
            "delta_q_after_joint_limit_clip": row.get("delta_q_after_joint_limit_clip"),
            "q_after_wiggle": row.get("q_after_wiggle"),
            "final_result": row.get("final_result"),
        }
        if provenance.startswith("stage192c_provenance="):
            provenance = provenance[len("stage192c_provenance=") :]
            for item in provenance.split(";"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    parsed[key] = value
        out.append(parsed)
    return out


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def frozen_replay_matrix() -> dict[str, dict[str, Any]]:
    independent = sorted((B / "replay_exact_independent_processes").glob("process_*.json"))
    return {
        "same_instance": result_summary([B / "replay_exact_same_instance_100_v2.json"]),
        "new_instance": result_summary([B / "replay_exact_new_instance_50.json"]),
        "independent_process": result_summary(independent),
    }


def current_call_matrix() -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in POLICIES:
        pdir = OUT / "matrix_call_00000182" / policy
        if policy == "P1_exact_replay":
            matrix[policy] = frozen_replay_matrix()
            continue
        if policy == "P0_disabled":
            files = [OUT / "policy_smoke" / "P0_disabled.json"]
            matrix[policy] = {"smoke_3": result_summary(files), "same_instance": {"status": "not_run_gate_not_met"}, "new_instance": {"status": "not_run_gate_not_met"}, "independent_process": {"status": "not_run_gate_not_met"}}
            continue
        matrix[policy] = {
            "same_instance": result_summary([pdir / "same_instance_100.json"]),
            "new_instance": result_summary([pdir / "new_instance_50.json"]),
            "independent_process": result_summary(sorted(pdir.glob("independent_*.json"))),
        }
    return matrix


def difficult_matrix() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for waypoint in DIFFICULT:
        result[str(waypoint)] = {}
        for policy in POLICIES:
            path = OUT / "matrix_waypoints" / f"waypoint_{waypoint}" / f"{policy}.json"
            result[str(waypoint)][policy] = result_summary([path])
    return result


def write_policy_manifest() -> None:
    text = """schema_version: '1.9.2c'
replay_exact_role:
  mechanism_validation: true
  regression_oracle: true
  generic_candidate_solver: false
  full_pipeline_solver: false
policies:
  P0_disabled: {wiggle: none}
  P1_exact_replay: {source: captured_internal_state, use: regression_only}
  P2_fixed_prng:
    generator: explicit_splitmix64
    initial_state: '0x6a09e667f3bcc909'
    mapping_to_double: 'top_53_bits_plus_half_times_2^-53; direction=2u-1; normalize'
    candidate_count: fixed_by_solver_event
  P3_axis_basis:
    directions: [+e1, -e1, +e2, -e2, +e3, -e3, +e4, -e4, +e5, -e5, +e6, -e6]
    amplitudes: 'max(epsilon,min(0.1,twist_norm))'
  P4_multi_joint_templates:
    directions: [signed_pairwise, orthogonal_sign_vectors, balanced_multi_joint_vectors]
    ordering: fixed
  P5_jacobian_svd:
    directions: [smallest_right_singular_vector, second_smallest_right_singular_vector]
    signs: [positive, negative]
    amplitudes: [0.25s, 0.5s, 1s, 2s]
  P6_hybrid:
    sequence: [jacobian_svd, axis_basis, multi_joint_templates, fixed_prng]
implementation:
  error_scaling_order_preserved: true
  joint_limit_clipping_order_preserved: true
  default_rng_used: false
  eigen_setRandom_used_by_new_policies: false
  collision_method: adaptive_discrete_interpolation
  ccd_status: not_available
  clearance_status: not_available
"""
    (OUT / "policy_manifest.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plugin = OUT / "overlay_install" / "deterministic_kdl_kinematics_plugin" / "lib" / "libdeterministic_kdl_kinematics_plugin.so"
    runtime = load_json(B / "runtime_kinematics_parameters.json")
    write_json("runtime_plugin_manifest.json", {
        "schema_version": "1.9.2c",
        "plugin_class_name": "deterministic_kdl_kinematics_plugin/DeterministicKDLKinematicsPlugin",
        "plugin_library_path": str(plugin),
        "plugin_library_sha256": sha256(plugin),
        "urdf_sha256": runtime["urdf_sha256"],
        "srdf_sha256": runtime["srdf_sha256"],
        "kinematics_config_sha256": runtime["kinematics_config_sha256"],
        "base_frame": runtime["base_frame"],
        "tip_frame": runtime["tip_frames"][0],
        "joint_order": ["j1", "j2", "j3", "j4", "j5", "j6"],
        "epsilon": runtime["epsilon"],
        "max_solver_iterations": runtime["max_solver_iterations"],
        "joint_weights": runtime["joint_weights"],
        "search_discretization": runtime["search_discretization"],
        "provenance": "new isolated Stage 1.9.2c overlay; Stage 1.9.2b output unchanged",
    })
    write_policy_manifest()

    call_matrix = current_call_matrix()
    difficult = difficult_matrix()
    write_json("coverage_call_00000182.json", {
        "scope": "single frozen call; direct KDL plugin replay only",
        "collision_method": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
        "policies": call_matrix,
        "full_pipeline_status": "not_run_gate_not_met",
    })
    write_json("coverage_waypoints_67_68_87_94.json", {
        "scope": "four difficult Stage 1 ON-state open-arch controls; 3 direct replay invocations per policy",
        "collision_method": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
        "waypoints": difficult,
        "full_pipeline_status": "not_run_gate_not_met",
    })

    event_paths = [
        *(OUT / "matrix_call_00000182" / p / "same_instance_100.csv" for p in POLICIES if p not in {"P0_disabled", "P1_exact_replay"}),
        *(OUT / "policy_smoke" / f"{p}.csv" for p in ["P0_disabled", "P1_exact_replay"]),
    ]
    events = [event for path in event_paths for event in parse_events(path)]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if events:
            table = pa.Table.from_pylist(events)
            pq.write_table(table, OUT / "wiggle_event_call_00000182.parquet")
    except Exception as exc:  # pragma: no cover - environment evidence is reported
        write_json("wiggle_event_call_00000182.parquet.status.json", {"status": "not_available", "reason": str(exc)})

    with (OUT / "wiggle_policy_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["policy", "target", "mode", "invocation_count", "successful_invocations", "success_rate", "exact_solution_hash_count", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for policy, modes in call_matrix.items():
            for mode, value in modes.items():
                if "invocation_count" in value:
                    writer.writerow({"policy": policy, "target": "call:00000182", "mode": mode, "status": "executed", **{k: value.get(k) for k in fields[3:7]}})
                else:
                    writer.writerow({"policy": policy, "target": "call:00000182", "mode": mode, "status": value.get("status", "not_available")})
        for waypoint, policies in difficult.items():
            for policy, value in policies.items():
                writer.writerow({"policy": policy, "target": f"waypoint {waypoint}", "mode": "same_instance_3", "status": "executed", **{k: value.get(k) for k in fields[3:7]}})

    with (OUT / "determinism_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["policy", "target", "mode", "invocation_count", "successful_invocations", "exact_solution_hash_count", "stable_observed", "canonical_hash_status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for policy, modes in call_matrix.items():
            for mode, value in modes.items():
                if "invocation_count" not in value:
                    writer.writerow({"policy": policy, "target": "call:00000182", "mode": mode, "canonical_hash_status": "not_run_gate_not_met"})
                    continue
                writer.writerow({"policy": policy, "target": "call:00000182", "mode": mode, "invocation_count": value["invocation_count"], "successful_invocations": value["successful_invocations"], "exact_solution_hash_count": value["exact_solution_hash_count"], "stable_observed": value["exact_solution_hash_count"] == 1, "canonical_hash_status": "not_run_candidate_set_unavailable"})

    write_json("ikfast_reproducibility_manifest.json", {
        "ikfast_status": "blocked",
        "reason": "No IKFast generator, generated solver C++, MoveIt IKFast wrapper, Collada generation model, or IKFast CMake target was found in the repository or active ROS overlay.",
        "system_dependencies_installed": False,
        "generated_solver_cpp_sha256": None,
        "moveit_wrapper_sha256": None,
        "collada_or_generation_model_sha256": None,
        "base_link": runtime["base_frame"],
        "tip_link": runtime["tip_frames"][0],
    })
    write_json("ikfast_candidate_comparison.json", {
        "ikfast_status": "blocked",
        "kdl_candidate_comparison": "not_available_ikfast_not_run",
        "ikfast_outputs": "not_available",
        "kdl_uncovered_reachable_solution_test": "not_available",
        "candidate_set_stability_comparison": "not_available",
    })

    frozen = frozen_replay_matrix()
    gate = {
        "C1_replay_exact_baseline": {"status": "passed", "evidence": "Stage 1.9.2b frozen overlay: 100/100, 50/50, 20/20; new isolated overlay smoke 3/3"},
        "C2_call_00000182_candidate_coverage": {"status": "failed", "evidence": "P2-P6 all 0/100, 0/50, 0/20; P1 is replay-only regression evidence and is excluded from generic solver recovery"},
        "C3_waypoint_67_68_87_94_candidate_coverage": {"status": "failed", "evidence": "P0-P6 each 0/3 on all four requested controls"},
        "C4_key_candidate_set_hash_three_runs": {"status": "not_run_gate_not_met", "evidence": "No accepted candidate set was available for the required comparison"},
        "C5_local_stage17_graph": {"status": "not_run_gate_not_met"},
        "C6_full_181_graph_hash": {"status": "not_run_gate_not_met"},
        "C7_full_path_vs_stage17": {"status": "not_run_gate_not_met"},
        "C8_ruckig_postcheck": {"status": "not_run_gate_not_met"},
        "C9_unblock_stage2": {"status": "blocked", "evidence": "C2 and C3 failed"},
        "full_pipeline_status": "not_run_gate_not_met",
        "Stage_1_9_2c": {"status": "failed_coverage"},
        "Stage_2": {"status": "blocked"},
    }
    write_json("gate_report.json", gate)
    status = """schema_version: '1.9.2c'
Stage_1_9_2c:
  status: failed_coverage
  C1_replay_exact_baseline: passed
  C2_call_00000182_candidate_coverage: failed
  C3_waypoint_67_68_87_94_candidate_coverage: failed
  C4_candidate_set_hash_three_runs: not_run_gate_not_met
  C5_local_stage17_graph: not_run_gate_not_met
  C6_full_181_graph_hash: not_run_gate_not_met
  C7_full_path: not_run_gate_not_met
  C8_ruckig: not_run_gate_not_met
Stage_2:
  status: blocked
ikfast_status: blocked
collision_method: adaptive_discrete_interpolation
ccd_status: not_available
clearance_status: not_available
full_pipeline_status: not_run_gate_not_met
"""
    (OUT / "stage192c_status.yaml").write_text(status, encoding="utf-8")

    report = """# Stage 1.9.2c structured implementation report

## Status

```yaml
Stage_1_9_2c: failed_coverage
Stage_2: blocked
ikfast_status: blocked
full_pipeline_status: not_run_gate_not_met
```

The new isolated overlay implements and independently executes P0 through P6. P1 exact replay remains regression-only. The direct replay executable calls the KDL plugin, but it does not run a PlanningScene collision callback, candidate-bank deduplication, graph insertion, dynamics, or Ruckig. Those layers are therefore not reported as passes.

## Answers to the required questions

1. `call:00000182`: P1_exact_replay restored the captured internal event and solved 3/3 in the new-overlay smoke; the generic deterministic policies P2-P6 solved 0/100, 0/50, and 0/20. P1 is not counted as generic coverage recovery.
2. Waypoints 67, 68, 87, and 94: P0-P6 each solved 0/3 in the direct single-point diagnostic. No policy restored accepted coverage.
3. The observed failure is at numerical KDL solution generation after wiggle; the direct replay never reached a valid candidate callback or graph layer. Joint-limit postcheck, collision callback, deduplication, and graph insertion were not run.
4. KDL-versus-IKFast raw solution counts and candidate sets cannot be compared because IKFast is unavailable.
5. IKFast discovery of KDL-only reachable solutions is not available.
6. P2-P6 deterministic outputs were stable as observed: failed calls produced no accepted solution and one empty-solution hash; P1 replay produced one accepted solution hash. Canonical candidate-set equivalence was not run because no candidate set was exported.
7. Stage 1.7 local graph recovery was not run because C2/C3 failed.
8. Three full 181-point graph hashes were not run.
9. Ruckig post-acceptance was not run.
10. Stage 2 cannot be unblocked; it remains blocked.

Collision labels remain `adaptive_discrete_interpolation`; CCD and clearance are `not_available` with JSON null where applicable. No OFF state, reorientation, GNN, reinforcement learning, closed-contour transition, or Stage 2 experiment was run.

The evidence supports only the bounded conclusion that the captured MoveIt KDL internal wiggle state is a causal facilitator for the observed frozen-call success. It does not establish that random wiggle is necessary for all difficult IK calls.
"""
    (OUT / "stage192c_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
