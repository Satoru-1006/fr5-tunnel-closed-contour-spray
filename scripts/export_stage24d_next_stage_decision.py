"""Export next-stage decision information from existing Stage 2.4D evidence.

This is a read-only evidence post-processor.  It does not invoke the Stage
2.4D runner, regenerate IK, validate collisions, or start any later stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE24D = ROOT / "outputs/ik_graph_stage24d_global_branch_cycle_audit/fr5_scaled_horseshoe_demo_v45"
OUT = STAGE24D / "next_stage_decision"
GOAL_OBJECTIVE = Path(r"C:\Users\86198\.codex\attachments\dfaa2bfc-a965-43d5-9322-d155d4488a3a\goal-objective.md")
WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(), "sha256": sha256(path) if path.exists() else None}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def angle_pair(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    q0 = [float(value) for value in source["joint_values_rad"]]
    q1 = [float(value) for value in target["joint_values_rad"]]
    signed = [math.degrees(b - a) for a, b in zip(q0, q1)]
    absolute = [abs(value) for value in signed]
    index = max(range(6), key=lambda i: (absolute[i], -i))
    return {"source_joint_values_rad": q0, "target_joint_values_rad": q1, "source_joint_values_deg": [math.degrees(x) for x in q0], "target_joint_values_deg": [math.degrees(x) for x in q1], "signed_delta_deg": signed, "absolute_delta_deg": absolute, "max_joint_delta_deg": absolute[index], "blocking_joint": f"j{index + 1}", "rejection_reason": "joint_step_exceeds_20_deg" if absolute[index] > MAX_STEP_DEG else None}


def endpoint_residual(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "fk_tcp_position_residual_m": node.get("fk_position_error"),
        "fk_tcp_orientation_residual_deg": node.get("fk_orientation_error"),
        "spray_distance_residual_m": node.get("spray_distance_error"),
        "normal_residual_deg": node.get("normal_error"),
        "roll_residual_deg": node.get("roll_error"),
        "residual_evidence_status": "partial_existing_stage24d_node_fields; spray_distance_normal_roll_not_recorded_in_stage24d_nodes" if any(node.get(key) is None for key in ("spray_distance_error", "normal_error", "roll_error")) else "complete",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = read_json(STAGE24D / "stage24d_gate_report.json")
    certificate = read_json(STAGE24D / "stage24d_infeasibility_certificate.json")
    topology = read_json(STAGE24D / "stage24d_joint_topology_audit.json")
    nodes = read_jsonl(STAGE24D / "stage24d_nodes.jsonl")
    edges = read_jsonl(STAGE24D / "stage24d_edges.jsonl")
    minima = list(csv.DictReader((STAGE24D / "stage24d_transition_minima.csv").open(newline="", encoding="utf-8")))
    by_id = {row["candidate_id"]: row for row in nodes}
    edge_by_key = {(int(row["from_waypoint"]), int(row["to_waypoint"]), row["source_node"], row["target_node"]): row for row in edges}
    edge_by_transition: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        edge_by_transition.setdefault(edge["transition"], []).append(edge)

    top20 = []
    seam_source = sorted((row for row in nodes if row["waypoint_index"] == 719), key=lambda row: row["candidate_id"])
    seam_target = sorted((row for row in nodes if row["waypoint_index"] == 0), key=lambda row: row["candidate_id"])
    for source in seam_source:
        for target in seam_target:
            delta = angle_pair(source, target)
            key = (delta["max_joint_delta_deg"], delta["absolute_delta_deg"][4], sum(delta["absolute_delta_deg"]), source["candidate_id"], target["candidate_id"])
            edge = edge_by_key.get((719, 0, source["candidate_id"], target["candidate_id"]))
            top20.append((key, {
                "rank": 0,
                "transition": "719->0",
                "source_node_id": source["candidate_id"],
                "target_node_id": target["candidate_id"],
                "source_set": source["source_stage"].replace("Stage_2_4", ""),
                "target_set": target["source_stage"].replace("Stage_2_4", ""),
                "source_waypoint": source["waypoint_index"],
                "target_waypoint": target["waypoint_index"],
                **delta,
                "source_residual": endpoint_residual(source),
                "target_residual": endpoint_residual(target),
                "source_fcl_node_valid": source["fcl_node_valid"],
                "source_bullet_node_valid": source["bullet_node_valid"],
                "target_fcl_node_valid": target["fcl_node_valid"],
                "target_bullet_node_valid": target["bullet_node_valid"],
                "edge_fcl_valid": edge["fcl_valid"] if edge else "not_validated_in_existing_evidence",
                "edge_bullet_valid": edge["bullet_valid"] if edge else "not_validated_in_existing_evidence",
                "edge_accepted_dual_backend": edge["accepted"] if edge else False,
                "edge_rejection_reason": edge["rejection_reason"] if edge else "joint_step_exceeds_20_deg; exact pair not present in persisted native edge evidence",
                "collision_method": edge["collision_method"] if edge else "adaptive_discrete_interpolation_not_run_for_this_pair",
                "clearance_m": "not_available",
                "clearance_status": "not_available",
            }))
    top20.sort(key=lambda item: item[0])
    top20_rows = [row for _, row in top20[:20]]
    for rank, row in enumerate(top20_rows, 1):
        row["rank"] = rank

    all_transitions = []
    for row in minima:
        transition = row["transition"]
        evidence_edges = edge_by_transition.get(transition, [])
        accepted = [edge for edge in evidence_edges if edge["accepted"] is True]
        all_transitions.append({
            "transition": transition,
            "source_waypoint": int(row["source_waypoint"]),
            "target_waypoint": int(row["target_waypoint"]),
            "minimum_possible_max_joint_delta_deg": float(row["minimum_possible_max_joint_step_deg"]),
            "blocking_joint": row["blocking_joint"],
            "has_le_20deg_dual_backend_valid_edge": bool(accepted),
            "dual_backend_valid_edge_count": len(accepted),
            "unique_source_node_count": len({edge["source_node"] for edge in accepted}),
            "unique_target_node_count": len({edge["target_node"] for edge in accepted}),
            "single_fragile_branch": len(accepted) == 1,
            "best_source_node": row["best_source_node"],
            "best_target_node": row["best_target_node"],
            "best_pair_fcl_valid": row["fcl_valid"] == "True",
            "best_pair_bullet_valid": row["bullet_valid"] == "True",
            "best_pair_rejection_reason": "joint_step_exceeds_20_deg" if float(row["minimum_possible_max_joint_step_deg"]) > MAX_STEP_DEG else None,
        })
    over_limit = [row for row in all_transitions if row["minimum_possible_max_joint_delta_deg"] > MAX_STEP_DEG]
    no_edge = [row for row in all_transitions if not row["has_le_20deg_dual_backend_valid_edge"]]
    fragile = [row for row in all_transitions if row["single_fragile_branch"]]

    source_refs = [
        source_ref(GOAL_OBJECTIVE, "Stage 2.4D objective and pass/blocked gate definition"),
        source_ref(ROOT / "config/scaled_demo/fr5_scaled_horseshoe_demo_v1.yaml", "closed-loop benchmark configuration"),
        source_ref(ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/scaled_demo_parameters.yaml", "frozen v44 process tolerance configuration"),
        source_ref(ROOT / "scripts/run_stage24d_global_branch_cycle_audit.py", "existing Stage 2.4D implementation"),
        source_ref(ROOT / "scripts/run_stage24c_seam_window_repair.py", "existing Stage 2.4C tolerance candidate generation"),
        source_ref(ROOT / "tests/test_stage24d_outputs.py", "existing Stage 2.4D tests"),
        source_ref(STAGE24D / "stage24d_gate_report.json", "Stage 2.4D gate evidence"),
        source_ref(STAGE24D / "stage24d_infeasibility_certificate.json", "Stage 2.4D infeasibility certificate"),
        source_ref(STAGE24D / "stage24d_nodes.jsonl", "Stage 2.4D frozen global nodes"),
        source_ref(STAGE24D / "stage24d_edges.jsonl", "Stage 2.4D persisted edge evidence"),
        source_ref(STAGE24D / "stage24d_transition_minima.csv", "Stage 2.4D all-transition minima"),
        source_ref(ROOT / "outputs/segmented_process_summary.json", "existing segmented spray-off execution evidence"),
        source_ref(ROOT / "outputs/production_readiness_check.json", "existing project production-readiness scope"),
        source_ref(ROOT / "outputs/final_acceptance_summary.json", "existing overall acceptance summary"),
        source_ref(ROOT / "README.md", "project-level execution description"),
    ]

    decision = {
        "schema_version": "stage24d-next-stage-decision-v1",
        "execution_policy": {"stage24d_rerun": False, "stage25_run": False, "TOTG": "not_run", "Ruckig": "not_run", "CCD": "not_run", "GNN": "not_run", "formal_inputs_modified": False},
        "closed_cycle_requirement": {
            "stage24d_gate_requirement": True,
            "stage24d_gate_source": "goal-objective.md Stage 2.4D pass gate requires 720 selected nodes, 720 selected edges, closure_edge_included, all waypoints covered, and maximum joint step <=20 degrees.",
            "benchmark_config_closed_loop": True,
            "benchmark_config_source": "config/scaled_demo/fr5_scaled_horseshoe_demo_v1.yaml process.closed_loop=true and waypoint_sampling.closed_loop=true.",
            "production_infinite_ON_requirement_explicitly_proven": False,
            "single_pass_with_spray_off_reorientation_supported_by_existing_project_evidence": True,
            "single_pass_evidence": "outputs/segmented_process_summary.json and outputs/production_readiness_check.json report segmented_process_with_smooth_reorientation_stops, 13 process segments, 12 spray-off reorientation transitions, and pass statuses.",
            "formal_reconciliation_status": "not_resolved_for_the_v45_Stage_2_4D_benchmark",
            "answer": "719->0 is a hard requirement for passing Stage 2.4D as currently defined, but the repository evidence does not establish that infinite pure-ON repetition is the production process requirement. Existing production-scope evidence supports a single-pass segmented process with spray-off reorientation; this must be treated as a separate process contract from the Stage 2.4D benchmark gate.",
        },
        "tolerance_coverage": {
            "global_nodes": len(nodes),
            "all_720_waypoints_have_at_least_one_node": len({row["waypoint_index"] for row in nodes}) == WAYPOINT_COUNT,
            "complete_formal_tolerance_space_covered": False,
            "position_offset": {"configured_position_tolerance_m": 0.006, "searched_in_stage24c": False, "evidence": "Stage 2.4C pose_target changes only normal-direction standoff and orientation; no lateral/longitudinal position sweep is generated."},
            "spray_distance": {"configured_standoff_tolerance_m": 0.005, "searched_offsets_m": [-0.005, 0.005], "searched_in_stage24c": True, "coverage": "finite endpoint variants only; not a continuous or per-waypoint exhaustive sweep"},
            "normal": {"configured_normal_tolerance_deg": 10.0, "searched_offsets_deg": [-10.0, 10.0], "searched_axes": ["x", "y"], "searched_in_stage24c": True, "coverage": "finite x/y perturbations in seam windows only; no full normal-cone or combined sweep"},
            "roll_orientation": {"configured_roll_freedom_deg": 15.0, "searched_offsets_deg": [-15.0, 15.0], "searched_in_stage24c": True, "coverage": "finite roll variants in seam windows only; no continuous or combined roll/normal/position search"},
            "joint_search_per_waypoint": {"all_720_waypoints": False, "stage24c_window_waypoints": "0..64 and 655..719 for the unioned seam windows", "stage24a_and_stage24b": "formal baseline plus seam candidates, not a complete tolerance Cartesian product"},
            "unsearched_degrees_of_freedom": ["lateral TCP position offset", "longitudinal/tangential TCP position offset", "continuous standoff interval between -5 mm and +5 mm", "continuous roll interval between -15 and +15 degrees", "full normal-cone orientation combinations", "jointly combined position+standoff+normal+roll perturbations at every waypoint", "new IK branch generation outside the frozen A/B/C candidate seeds"],
            "do_not_claim": "The current certificate cannot be generalized to complete in-tolerance process-space infeasibility.",
        },
        "closure_top20": {"transition": "719->0", "ranking": "minimum model-aware maximum joint delta over existing 6722 frozen nodes; no new IK or collision validation", "rows": top20_rows, "clearance": "not_available"},
        "all_720_transitions": {"rows": all_transitions, "transitions_over_20deg_minimum": over_limit, "transitions_without_dual_backend_valid_edge": no_edge, "single_fragile_branch_transitions": fragile, "answer": {"only_719_to_0_over_20deg": len(over_limit) == 1 and over_limit[0]["transition"] == "719->0", "only_719_to_0_without_dual_edge": len(no_edge) == 1 and no_edge[0]["transition"] == "719->0", "other_single_fragile_branches_exist": bool(fragile)}},
        "next_stage_classification": {
            "recommended": "requires_stage_2_4E_tolerance_complete_search",
            "conditional_alternative": "closed_joint_cycle_not_required_for_single_pass",
            "ready_for_stage_2_5_process_off_reorientation_bridge": False,
            "reason": "The current Stage 2.4D gate is blocked and its 6722-node union does not cover the complete formal tolerance space. Existing segmented-process evidence supports the single-pass alternative, but the v45 benchmark requirement and production process contract are not formally reconciled.",
        },
        "source_files": source_refs,
        "stage24d_snapshot": {"Stage_2_4D": gate["Stage_2_4D"], "closed_cycle_found": gate["closed_cycle_found"], "minimum_achievable_max_step_deg": certificate["minimum_achievable_max_step_deg"], "blocking_joint": certificate["blocking_joint"], "blocking_transition": certificate["blocking_transition"], "deterministic_reproduction": gate["deterministic_reproduction"], "tests": gate["tests"]},
    }

    write_json(OUT / "stage24d_next_stage_decision.json", decision)
    with (OUT / "stage24d_transition_719_0_top20.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["rank", "transition", "source_node_id", "target_node_id", "source_set", "target_set", "source_joint_values_deg", "target_joint_values_deg", "signed_delta_deg", "absolute_delta_deg", "max_joint_delta_deg", "blocking_joint", "source_fk_tcp_position_residual_m", "source_fk_tcp_orientation_residual_deg", "target_fk_tcp_position_residual_m", "target_fk_tcp_orientation_residual_deg", "source_spray_distance_residual_m", "target_spray_distance_residual_m", "source_normal_residual_deg", "target_normal_residual_deg", "source_roll_residual_deg", "target_roll_residual_deg", "source_fcl_node_valid", "source_bullet_node_valid", "target_fcl_node_valid", "target_bullet_node_valid", "edge_fcl_valid", "edge_bullet_valid", "edge_accepted_dual_backend", "edge_rejection_reason", "clearance_m", "clearance_status"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in top20_rows:
            flat = {"rank": row["rank"], "transition": row["transition"], "source_node_id": row["source_node_id"], "target_node_id": row["target_node_id"], "source_set": row["source_set"], "target_set": row["target_set"], "source_joint_values_deg": json.dumps(row["source_joint_values_deg"], separators=(",", ":")), "target_joint_values_deg": json.dumps(row["target_joint_values_deg"], separators=(",", ":")), "signed_delta_deg": json.dumps(row["signed_delta_deg"], separators=(",", ":")), "absolute_delta_deg": json.dumps(row["absolute_delta_deg"], separators=(",", ":")), "max_joint_delta_deg": row["max_joint_delta_deg"], "blocking_joint": row["blocking_joint"], "source_fk_tcp_position_residual_m": row["source_residual"]["fk_tcp_position_residual_m"], "source_fk_tcp_orientation_residual_deg": row["source_residual"]["fk_tcp_orientation_residual_deg"], "target_fk_tcp_position_residual_m": row["target_residual"]["fk_tcp_position_residual_m"], "target_fk_tcp_orientation_residual_deg": row["target_residual"]["fk_tcp_orientation_residual_deg"], "source_spray_distance_residual_m": row["source_residual"]["spray_distance_residual_m"], "target_spray_distance_residual_m": row["target_residual"]["spray_distance_residual_m"], "source_normal_residual_deg": row["source_residual"]["normal_residual_deg"], "target_normal_residual_deg": row["target_residual"]["normal_residual_deg"], "source_roll_residual_deg": row["source_residual"]["roll_residual_deg"], "target_roll_residual_deg": row["target_residual"]["roll_residual_deg"], "source_fcl_node_valid": row["source_fcl_node_valid"], "source_bullet_node_valid": row["source_bullet_node_valid"], "target_fcl_node_valid": row["target_fcl_node_valid"], "target_bullet_node_valid": row["target_bullet_node_valid"], "edge_fcl_valid": row["edge_fcl_valid"], "edge_bullet_valid": row["edge_bullet_valid"], "edge_accepted_dual_backend": row["edge_accepted_dual_backend"], "edge_rejection_reason": row["edge_rejection_reason"], "clearance_m": row["clearance_m"], "clearance_status": row["clearance_status"]}
            writer.writerow(flat)
    with (OUT / "stage24d_all_720_transition_decision.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_transitions[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_transitions)

    md = ["# Stage 2.4D next-stage decision information", "", "This export is post-processing only. Stage 2.4D, Stage 2.5, TOTG, Ruckig, CCD and GNN were not executed by this export.", "", "## Decision summary", "", "```yaml", "Stage_2_4D: " + gate["Stage_2_4D"], "closed_cycle_requirement_for_Stage_2_4D_gate: true", "production_infinite_pure_ON_requirement_proven: false", "single_pass_segmented_spray_off_evidence_present: true", "complete_tolerance_space_covered_by_6722_nodes: false", "recommended_next_step: requires_stage_2_4E_tolerance_complete_search", "conditional_single_pass_classification: closed_joint_cycle_not_required_for_single_pass", "ready_for_stage_2_5: false", "```", "", "## Hard-gate versus production-contract interpretation", "", "The Stage 2.4D benchmark gate is explicitly cyclic and requires the 719→0 edge. The repository's separate production-scope evidence describes a single-pass segmented execution with 13 process segments and 12 spray-off reorientation transitions. No existing source formally reconciles those two scopes; therefore Stage 2.5 is not marked ready.", "", "## Tolerance coverage", "", "The 6722 nodes cover all 720 waypoint indices, but they are not a complete tolerance-space Cartesian or joint search. Stage 2.4C generated finite standoff/roll/normal variants only in seam windows. Lateral/longitudinal position offsets, continuous intervals, combined perturbations, and all-waypoint tolerance search remain unsearched.", "", f"## Closure 719→0", "", f"The top 20 existing-node pairs are in `stage24d_transition_719_0_top20.csv`. The best model-only pair is `{certificate['minimum_achievable_max_step_source_node']} -> {certificate['minimum_achievable_max_step_target_node']}`, with `{certificate['minimum_achievable_max_step_deg']:.12f}°` at `{certificate['blocking_joint']}`. The exact pair has no persisted dual-backend-valid edge; clearance is `not_available`.", "", "## All 720 transitions", "", f"The complete table is `stage24d_all_720_transition_decision.csv`. Transitions with minimum model-aware step >20°: `{', '.join(row['transition'] for row in over_limit)}`. Transitions without a persisted dual-backend-valid edge: `{', '.join(row['transition'] for row in no_edge)}`. Single-edge transitions: `{', '.join(row['transition'] for row in fragile) if fragile else 'none'}`.", "", "## Source evidence", ""]
    md.extend([f"- `{item['path']}` — {item['role']}" for item in source_refs])
    (OUT / "stage24d_next_stage_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    yaml_lines = ["Stage_2_4D: " + gate["Stage_2_4D"], "closed_cycle_requirement_for_Stage_2_4D_gate: true", "production_infinite_pure_ON_requirement_proven: false", "single_pass_segmented_spray_off_evidence_present: true", "complete_tolerance_space_covered_by_6722_nodes: false", "minimum_719_to_0_max_joint_delta_deg: " + str(certificate["minimum_achievable_max_step_deg"]), "minimum_719_to_0_blocking_joint: " + certificate["blocking_joint"], "transitions_over_20deg: [" + ", ".join(row["transition"] for row in over_limit) + "]", "transitions_without_dual_backend_valid_edge: [" + ", ".join(row["transition"] for row in no_edge) + "]", "single_fragile_branch_transitions: [" + ", ".join(row["transition"] for row in fragile) + "]", "recommended_next_step: requires_stage_2_4E_tolerance_complete_search", "conditional_single_pass_classification: closed_joint_cycle_not_required_for_single_pass", "ready_for_stage_2_5_process_off_reorientation_bridge: false", "stage24d_rerun: false", "stage25_run: false", "TOTG: not_run", "Ruckig: not_run", "CCD: not_run", "GNN: not_run"]
    (OUT / "stage24d_next_stage_decision.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT.resolve()), "top20_rows": len(top20_rows), "all_transition_rows": len(all_transitions), "transitions_over_20deg": [row["transition"] for row in over_limit], "transitions_without_dual_backend_valid_edge": [row["transition"] for row in no_edge]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
