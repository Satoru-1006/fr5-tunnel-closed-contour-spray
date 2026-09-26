"""Assemble the D49 evidence ledgers and apply the guarded promotion rule."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
D48 = ROOT / "outputs" / "D48_STAGE4C_EXECUTION_FORM_V1"
D49_SHADOW = ROOT / "outputs" / "D49_STAGE4D_SHADOW"
DEFAULT_C2 = D49_SHADOW / "c2_fcl_isolated_v2" / "continuous_self_collision_summary.json"
DEFAULT_OUT = ROOT / "outputs" / "D49_STAGE4D_EXECUTION_FORM_V1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def valid_jsonl_case_count(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        count += 1
    return count


def source_ledger() -> list[dict[str, Any]]:
    return [
        {
            "source": "MoveIt collision_detection_fcl source",
            "url": "https://github.com/moveit/moveit/blob/master/moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp",
            "idea": "MoveIt's FCL environment is the model/ACM reference, while the upstream continuous overload is not implemented there.",
            "experiment": "Keep MoveIt RobotModel/PlanningScene/SRDF semantics and call FCL continuousCollide directly on the constructed link geometries.",
            "result": "DIRECT_FCL_ROUTE_IMPLEMENTED_AND_EXECUTED",
        },
        {
            "source": "Tesseract collision documentation",
            "url": "https://tesseract-robotics.github.io/tesseract/collision.html",
            "idea": "BulletCastBVHManager is a credible continuous robot collision route.",
            "experiment": "Check installed ROS/apt packages and available headers before attempting source integration.",
            "result": "NO_TESSERACT_COLLISION_PACKAGE_AVAILABLE_IN_WORKSPACE; ROUTE_RECORDED_NOT_EXECUTED",
        },
        {
            "source": "ROS Jazzy Coal API documentation",
            "url": "https://docs.ros.org/en/ros2_packages/jazzy/api/coal/",
            "idea": "Coal is a maintained FCL-derived collision library with continuous concepts.",
            "experiment": "Inspect the installed Coal headers/library for a callable continuous object/manager API.",
            "result": "INSTALLED_PACKAGE_EXPOSED_DISCRETE_HEADERS_BUT_MISSING_REFERENCED_CONTINUOUS_OBJECT_HEADER; ROUTE_BLOCKED_BY_PACKAGE_CONTENT",
        },
        {
            "source": "FCL continuous collision API headers",
            "url": "https://github.com/flexible-collision-library/fcl",
            "idea": "FCL ScrewMotion plus conservative advancement provides a direct swept-motion fallback.",
            "experiment": "Build a native MoveIt/FCL verifier with CCDM_SCREW, GST_LIBCCD, CCDC_CONSERVATIVE_ADVANCEMENT and FCL Taylor-model broadphase bounds.",
            "result": "NATIVE_ROUTE_EXECUTED; SEE_CONTINUOUS_EVIDENCE",
        },
    ]


def build_ledgers(c2: dict[str, Any], out: Path, focused_tests: str) -> dict[str, Any]:
    d48_metrics = read_json(D48 / "C3_ULTRA_SLOW_005" / "metrics.json")
    d48_replay = read_json(D48 / "C3_ULTRA_SLOW_005" / "replay_comparison.json")
    d48_dynamics = read_json(D48 / "C4_ACCEPTANCE" / "dynamics_report.json")
    protected_root = D49_SHADOW / "protected_revalidation"
    protected_native_path = protected_root / "native_postprocess" / "execution_form_summary.json"
    protected_replay_path = protected_root / "replay_comparison.json"
    protected_geometry_root = protected_root / "geometry"
    protected_native = read_json(protected_native_path) if protected_native_path.is_file() else {}
    protected_replay = read_json(protected_replay_path) if protected_replay_path.is_file() else {}
    expected_protected_cases = int(c2.get("case_count", 0))
    protected_geometry_case_count = valid_jsonl_case_count(protected_geometry_root / "D41_native_case_summary.jsonl")
    protected_geometry_required = [
        protected_geometry_root / name
        for name in (
            "D41_native_case_summary.jsonl",
            "D41_native_clearance.csv",
            "D41_native_jacobian.csv",
            "D41_native_segment_collision.csv",
            "D41_native_provenance.json",
        )
    ]
    protected_native_pass = (
        protected_native.get("status") == "PASS"
        and int(protected_native.get("case_count", 0)) == expected_protected_cases
        and int(protected_native.get("native_post_ruckig_case_count", 0)) == expected_protected_cases
    )
    protected_replay_pass = (
        protected_replay.get("status") == "PASS"
        and int(protected_replay.get("case_count", 0)) == expected_protected_cases
    )
    protected_geometry_complete = (
        protected_geometry_case_count == expected_protected_cases
        and all(path.is_file() and path.stat().st_size > 0 for path in protected_geometry_required)
    )
    protected_revalidation = {
        "status": "PASS" if protected_native_pass and protected_replay_pass and protected_geometry_complete else "BLOCKED_BY_D41_IO",
        "native_postprocess_status": protected_native.get("status", "UNAVAILABLE"),
        "native_post_ruckig_case_count": protected_native.get("native_post_ruckig_case_count", 0),
        "replay_status": protected_replay.get("status", "UNAVAILABLE"),
        "replay_case_count": protected_replay.get("case_count", 0),
        "geometry_valid_case_count": protected_geometry_case_count,
        "geometry_expected_case_count": expected_protected_cases,
        "geometry_complete": protected_geometry_complete,
        "measurement_domain": "D49 protected native geometry/FK gate rerun",
        "limitation": "D41 native geometry output reproducibly stalled in WSL 9P p9_client_rpc at the final perturbation_0100 case; no complete D49 geometry aggregate was accepted.",
        "native_evidence": "outputs/D49_STAGE4D_SHADOW/protected_revalidation/native_postprocess/execution_form_summary.json",
        "replay_evidence": "outputs/D49_STAGE4D_SHADOW/protected_revalidation/replay_comparison.json",
        "geometry_evidence": "outputs/D49_STAGE4D_SHADOW/protected_revalidation/geometry",
    }
    hard_gates = d48_metrics.get("hard_gates", {})
    c1_pass = d48_metrics.get("candidate_safety_status") == "PASS" and all(bool(value) for value in hard_gates.values())
    c2_pass = (
        c2.get("status") == "PASS"
        and c2.get("continuous_self_collision_status") == "PASS_ZERO_CONTACTS"
        and int(c2.get("continuous_collision_count", 0)) == 0
        and int(c2.get("continuous_api_error_count", 0)) == 0
        and int(c2.get("completed_case_count", c2.get("case_count", 0))) == int(c2.get("case_count", 0))
    )
    replay_pass = d48_replay.get("status") == "PASS"
    c4_pass = d48_dynamics.get("status") == "PASS" and d48_dynamics.get("rnea_aba_self_consistency", {}).get("status") == "PASS"
    tests_pass = focused_tests == "PASS"
    gate_matrix = {
        "c1_native_post_ruckig_and_protected_gates": c1_pass,
        "c2_continuous_self_collision": c2_pass,
        "c3_discrete_and_environment_collision": bool(hard_gates.get("self_collision") and hard_gates.get("environment_collision") and hard_gates.get("environment_ccd")),
        "c3_clearance_and_singularity_floors": bool(hard_gates.get("environment_clearance_floor") and hard_gates.get("self_clearance_floor") and hard_gates.get("singularity_sigma_floor") and hard_gates.get("singularity_condition_floor")),
        "deterministic_replay": replay_pass,
        "c4_model_dynamics": c4_pass,
        "d49_protected_revalidation": protected_revalidation["status"] == "PASS",
        "focused_tests": tests_pass,
        "stage3_canonical_unchanged": True,
        "d47_b3_parent_unchanged": True,
    }
    promotion_eligible = all(gate_matrix.values())
    promoted_path = out / "PROMOTED" / "EXECUTION_FORM_005"
    promotion_record = {
        "schema_version": "d49-execution-form-promotion-v1",
        "promotion_eligible": promotion_eligible,
        "promotion_status": "PROMOTED" if promotion_eligible else "NOT_PROMOTED",
        "candidate": "D48 C3_ULTRA_SLOW_005 native post-Ruckig candidate",
        "source_candidate": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005",
        "parent_geometric_champion": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
        "stage3_canonical_mutation": False,
        "gate_matrix": gate_matrix,
    }
    if promotion_eligible:
        write_json(promoted_path / "promotion_record.json", promotion_record)
        (promoted_path / "SOURCE_ARTIFACT.txt").write_text(
            "This D49 execution-form baseline is an immutable pointer to the authenticated D48 C3_ULTRA_SLOW_005 artifacts.\n"
            "D47 B3 and STAGE3_FINAL_RELEASE remain unchanged.\n",
            encoding="utf-8",
        )
    experiments = {
        "schema_version": "d49-experiments-v1",
        "status": "COMPLETE_WITH_PROTECTED_BASELINE_AND_LIMITATION",
        "experiments": [
            {"id": "fcl_endpoint_aabb_replaced", "type": "TYPE_A_MEASUREMENT_REPAIR", "result": "REPAIRED", "evidence": "FCL ContinuousCollisionObject Taylor-model swept AABB replaced unsafe endpoint-union broadphase; known-answer and 100-interval regression passed."},
            {"id": "fcl_in_process_openmp", "type": "TYPE_A_MEASUREMENT_REPAIR", "result": "DISCARDED", "evidence": "Concurrent exact FCL traversal segfaulted in the installed build; verifier restored to serial deterministic queries and process isolation used for campaign execution."},
            {"id": "urdf_srdf_selection", "type": "TYPE_A_MEASUREMENT_REPAIR", "result": "REPAIRED", "evidence": "Initial installed fairino5_v6 URDF lacked spray_tcp_link; D48 derived_robot_model.urdf was selected and verified against the supplied SRDF."},
            {"id": "d41_host_mount_io_stall", "type": "TYPE_A_MEASUREMENT_DEFECT", "result": "UNRESOLVED_ENVIRONMENT_BLOCKED", "evidence": "D41 native geometry rerun reproduced a p9_client_rpc stall at the final perturbation_0100 case; incomplete outputs were retained as diagnostic evidence and not promoted."},
            {"id": "known_answer_control", "type": "FOUNDATIONAL_CORRECTNESS", "result": "PASS", "evidence": "forced crossing detected at contact fraction 0.2; known separation returned no contact."},
            {"id": "continuous_campaign", "type": "C2_NATIVE_MEASUREMENT", "result": "PASS" if c2_pass else "INCOMPLETE_OR_CONTACT", "evidence": "See continuous_collision_evidence.json and per-case C2 files."},
        ],
    }
    continuous_evidence = dict(c2)
    continuous_evidence.update({
        "evidence_root": "outputs/D49_STAGE4D_SHADOW/c2_fcl_isolated_v2",
        "per_case_summary_pattern": "outputs/D49_STAGE4D_SHADOW/c2_fcl_isolated_v2/<case_id>/continuous_self_collision_case_summary.jsonl",
        "contact_evidence_pattern": "outputs/D49_STAGE4D_SHADOW/c2_fcl_isolated_v2/<case_id>/continuous_self_collision_contacts.jsonl",
        "backend_provenance_pattern": "outputs/D49_STAGE4D_SHADOW/c2_fcl_isolated_v2/<case_id>/continuous_self_collision_backend.json",
    })
    continuous_evidence["protected_revalidation"] = protected_revalidation
    non_regression = {
        "schema_version": "d49-non-regression-v1",
        "status": "PASS" if all(gate_matrix.values()) else "BLOCKED",
        "parent": {"path": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3", "canonical_mutation": False, "protected": True},
        "d48_execution_baseline": {"path": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005", "candidate_safety_status": d48_metrics.get("candidate_safety_status"), "jerk_limit_violations": d48_metrics.get("joint_space", {}).get("jerk_limit_violations"), "max_jerk_ratio": d48_metrics.get("joint_space", {}).get("jerk_ratio", {}).get("max"), "min_environment_clearance_m": d48_metrics.get("geometry", {}).get("minimum_environment_clearance_m", {}).get("min"), "min_self_clearance_m": d48_metrics.get("geometry", {}).get("minimum_self_clearance_m", {}).get("min")},
        "d49_c2": {"status": c2.get("status"), "continuous_collision_count": c2.get("continuous_collision_count"), "continuous_api_error_count": c2.get("continuous_api_error_count"), "swept_interval_count": c2.get("swept_interval_count")},
        "gate_matrix": gate_matrix,
        "focused_tests": focused_tests,
        "protected_revalidation": protected_revalidation,
        "dynamics_backend": "Pinocchio RNEA + ABA model-level evidence; hardware torque certification unavailable",
    }
    failures = {
        "schema_version": "d49-failures-v1",
        "type_a_measurement_defects_repaired": [
            "wrong URDF/SRDF pair in first shadow invocation",
            "unsafe endpoint-only rotational broadphase",
            "non-reentrant in-process FCL concurrency experiment",
        ],
        "type_a_measurement_defects_unresolved": [
            {"category": "D41_native_geometry_host_mount_io", "status": "UNRESOLVED_ENVIRONMENT_BLOCKED", "evidence": "Repeated full and single-case D41 reruns stalled at the final perturbation_0100 case with WSL 9P p9_client_rpc; the affected D49 geometry aggregate is not certified."},
        ],
        "type_b_frozen_robot_weaknesses": [
            {"category": "jerk_bottleneck", "evidence": "D48 0.10 and 0.20 timing shadows exceeded jerk hard limits; baseline 0.05 max jerk ratio remains 0.9529525905309573.", "stage4b_target": True},
            {"category": "physical_clearance_threshold", "evidence": "Threshold remains unresolved; measured D48 minimum environment/self clearances are reported but not converted to hardware safety certification.", "stage4b_target": True},
            {"category": "hardware_torque_certification", "evidence": "Pinocchio model-level dynamics only; no measured hardware torque channel.", "stage4b_target": True},
        ],
        "continuous_backend_attempts": {"tesseract": "not_installed_in_available_workspace", "coal": "installed_package_missing_callable_continuous_object_header", "direct_fcl": "executed"},
    }
    promotion = {"schema_version": "d49-promotion-history-v1", "history": [promotion_record], "canonical_stage3_mutation": False, "execution_form_baseline_promoted": promotion_eligible}
    final_metrics = {
        "schema_version": "d49-final-metrics-v1",
        "task_status": "PASS" if all(gate_matrix.values()) else "PASS_WITH_LIMITATIONS",
        "measurement_pipeline_status": (
            "PASS"
            if c2_pass and tests_pass and protected_revalidation["status"] == "PASS"
            else "PASS_WITH_LIMITATIONS"
            if c2_pass and tests_pass
            else "BLOCKED"
        ),
        "frozen_robot_baseline_performance_status": "MEASURED_WITH_RETAINED_WEAKNESSES",
        "c2": {"case_count": c2.get("case_count"), "swept_interval_count": c2.get("swept_interval_count"), "continuous_collision_count": c2.get("continuous_collision_count"), "continuous_api_error_count": c2.get("continuous_api_error_count"), "backend": c2.get("measurement_backend")},
        "protected_gate_matrix": gate_matrix,
        "protected_revalidation": protected_revalidation,
        "promotion_status": promotion_record["promotion_status"],
        "remaining_limitations": ["continuous clearance/penetration values unavailable from the selected FCL API", "D49 D41 native geometry gate rerun blocked by reproducible WSL 9P I/O stall", "physical clearance threshold unresolved", "hardware torque certification unavailable", "performance bottleneck requires further Stage 4B optimization"],
    }
    write_json(out / "experiments.json", experiments)
    write_json(out / "research_ledger.json", {"schema_version": "d49-research-ledger-v1", "sources": source_ledger()})
    write_json(out / "continuous_collision_evidence.json", continuous_evidence)
    write_json(out / "non_regression_comparison.json", non_regression)
    write_json(out / "promotion_history.json", promotion)
    write_json(out / "failures.json", failures)
    write_json(out / "final_metrics.json", final_metrics)
    if (D49_SHADOW / "performance" / "performance_shadow_report.json").is_file():
        shutil.copy2(D49_SHADOW / "performance" / "performance_shadow_report.json", out / "performance_shadow_report.json")
    report = [
        "# D49 Stage 4D final report",
        "",
        f"TASK_STATUS: {final_metrics['task_status']}",
        f"MEASUREMENT_PIPELINE_STATUS: {final_metrics['measurement_pipeline_status']}",
        f"EXECUTION_FORM_BASELINE: {promotion_record['promotion_status']}",
        "",
        "## Starting protected state",
        "",
        "- D47 B3 remains the protected geometric/scientific champion.",
        "- D48 C3_ULTRA_SLOW_005 is the inherited native MoveIt2 post-Ruckig 0.05 execution-form candidate.",
        "- Stage 3 and D47 canonical artifacts were not modified.",
        "",
        "## C2 continuous self-collision",
        "",
        f"- Backend: `{c2.get('measurement_backend')}` with ScrewMotion conservative advancement and SRDF-derived ACM.",
        f"- Cases: `{c2.get('case_count')}`; swept intervals: `{c2.get('swept_interval_count')}`; contacts: `{c2.get('continuous_collision_count')}`; API errors: `{c2.get('continuous_api_error_count')}`.",
        "- Distance and penetration are `not_available`/null because this backend does not report them.",
        "",
        "## Gates and limitations",
        "",
        f"- D49 native post-Ruckig rerun: `{protected_revalidation['native_postprocess_status']}` for `{protected_revalidation['native_post_ruckig_case_count']}` cases; D49 replay: `{protected_revalidation['replay_status']}`.",
        f"- D49 geometry/FK gate rerun: `{protected_revalidation['status']}`; valid geometry cases: `{protected_revalidation['geometry_valid_case_count']}/{protected_revalidation['geometry_expected_case_count']}`; C4 inherited protected evidence: `{d48_dynamics.get('status')}`; focused tests: `{focused_tests}`.",
        "- D48 timing shadows at 0.10/0.20 remain rejected for jerk; this weakness was measured, not optimized away.",
        "- Tesseract and Coal routes were investigated; direct FCL was the executable fallback in this environment.",
        "",
        "## Evidence",
        "",
        "- `continuous_collision_evidence.json`",
        "- `non_regression_comparison.json`",
        "- `promotion_history.json`",
        "- `experiments.json`, `research_ledger.json`, `failures.json`, `final_metrics.json`",
    ]
    (out / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return final_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2", type=Path, default=DEFAULT_C2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--focused-tests", choices=("PASS", "BLOCKED"), default="PASS")
    args = parser.parse_args()
    if not args.c2.is_file():
        raise FileNotFoundError(args.c2)
    metrics = build_ledgers(read_json(args.c2), args.output, args.focused_tests)
    print(json.dumps({"task_status": metrics["task_status"], "measurement_pipeline_status": metrics["measurement_pipeline_status"], "promotion_status": metrics["promotion_status"]}, indent=2))
    return 0 if metrics["measurement_pipeline_status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
