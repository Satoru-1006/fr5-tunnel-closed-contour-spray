"""Materialize the D62 final shadow status and concise report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D62_CONTINUOUS_CERTIFICATE_SHADOW"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    cases = {}
    for case, filename in (("auto0", "D62_CERTIFICATE_combined_full.json"), ("auto1", "D62_CERTIFICATE_combined_full.json")):
        cases[case] = load(OUT / case / filename)
    summary = {}
    for case, value in cases.items():
        measurement = value.get("measurement", {})
        requested = int(value.get("checked_time_interval", {}).get("requested_interval_count", 0))
        certified = int(measurement.get("combined_certified_regions", 0))
        unresolved = int(measurement.get("combined_unresolved_regions", 0))
        summary[case] = {
            "status": value.get("status"),
            "requested_intervals": requested,
            "checked_intervals": int(value.get("checked_time_interval", {}).get("checked_interval_count", 0)),
            "certified_intervals": certified,
            "unresolved_intervals": unresolved,
            "collision_intervals": int(measurement.get("combined_collision_regions", 0)),
            "certified_fraction": certified / requested if requested else None,
            "resource_limit_reached": bool(value.get("coverage", {}).get("resource_limit_reached", False)),
            "coarse_wall_time_s": value.get("execution", {}).get("coarse_wall_time_s"),
            "refinement_wall_time_s": value.get("execution", {}).get("refinement_wall_time_s"),
            "total_wall_time_s": value.get("execution", {}).get("wall_time_s"),
        }
    status = {
        "schema_version": "d62-final-shadow-status-v1",
        "D62_TASK_STATUS": "PARTIALLY_EXECUTED_MEASURED_NO_PROMOTION",
        "claim_scope": "current measured shadow only; full interval evaluation with unresolved conservative enclosures is not a continuous-certificate PASS",
        "supersession": {
            "historical_predecessor": "D59/D60 zero-collision/zero-unresolved observations remain scoped to their conservative models",
            "current_boundary": "D62 unresolved intervals block theorem-level full coverage and exact external articulated FK(q(t)) self-CCD",
            "promotion": "NO_PROMOTION",
        },
        "coverage_status": "FULL_INTERVALS_EVALUATED_BUT_CONTINUOUS_CERTIFICATION_REMAINS_UNRESOLVED",
        "route_status": {
            "route_a_bvh_obb": "EXECUTED_SOUND_TRIANGLE_PRESERVING",
            "route_b_adaptive_interval_subdivision": "EXECUTED_BOUNDED_ON_500_INTERVAL_PROBE; FULL_UNRESOLVED_REFINEMENT_INTERRUPTED_BEFORE_VALID_OUTPUT",
            "route_c_combined": "EXECUTED_COARSE_THEN_INDEXED_REFINEMENT_ARCHITECTURE",
            "convex_decomposition": "NOT_AVAILABLE_NOT_SILENTLY_SUBSTITUTED",
            "external_articulated_ccd": "NOT_AVAILABLE",
        },
        "cases": summary,
        "known_answer_and_regression": {
            "d62_tests": "12 passed",
            "triangle_partition_soundness_repair": True,
            "direct_script_import_repair": True,
            "projection_axis_degeneracy_repair": True,
            "protected_baseline_mutated": False,
        },
        "type_a_measurement_repairs": [
            "direct-script import path repair",
            "complete-triangle rather than individual-vertex BVH partitioning",
            "sound PCA OBB pose/extrema construction with degenerate fallback",
            "invalid projection-axis filtering",
            "node-shape, interval-transform, and projection caching",
            "explicit interval-indexed refinement and fail-closed merge",
        ],
        "remaining_p0": [
            "interval FK enclosure inflation leaves thousands of forearm_link|wrist2_link regions unresolved",
            "bounded adaptive refinement is too expensive in the Python reference implementation for full unresolved sets",
            "exact external nonlinear articulated CCD backend is unavailable",
            "physical clearance and hardware safety thresholds remain unavailable",
        ],
        "evidence": {
            "problem_map": str((OUT / "D62_PROBLEM_MAP.md").resolve()),
            "route_matrix": str((OUT / "D62_ROUTE_MATRIX.md").resolve()),
            "auto0_combined": str((OUT / "auto0" / "D62_CERTIFICATE_combined_full.json").resolve()),
            "auto1_combined": str((OUT / "auto1" / "D62_CERTIFICATE_combined_full.json").resolve()),
        },
        "promotion": "NO_PROMOTION",
        "canonical": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
        "protected_floor": "c4_clearance_adversarial0101_amp005_consistent_0025",
    }
    (OUT / "D62_FINAL_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    superseded = {
        "status": "SUPERSEDED_UNSOUND_PRE_TRIANGLE_PARTITION_FIX",
        "reason": "An earlier disposable D62 probe partitioned individual STL vertices and is not valid evidence. Use D62_CERTIFICATE_coarse_full.json and D62_CERTIFICATE_combined_full.json instead.",
        "authoritative_replacement": "D62_CERTIFICATE_combined_full.json",
    }
    for case in ("auto0", "auto1"):
        (OUT / case / "D62_CERTIFICATE_full.json").write_text(json.dumps(superseded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# D62 Continuous Certification Coverage Upgrade — Final Shadow Report",
        "",
        "Status: `PARTIALLY_EXECUTED_MEASURED_NO_PROMOTION`",
        "",
        "D62 completed the coarse sound triangle-preserving BVH/OBB pass over every native trajectory interval for both D59 finalists. It did not obtain full continuous certification: unresolved remains unresolved, and the bounded full-set refinement was stopped before producing a valid result because the Python reference cost became hour-scale.",
        "",
        "Claim fence: D59/D60 zero-collision/zero-unresolved observations remain historical, scoped conservative-model results. D62's full interval accounting supersedes those PASS-like labels for current coverage: the nonzero unresolved sets below are not a continuous-certificate PASS and do not establish exact external articulated `FK(q(t))` self-CCD.",
        "",
        "## Coverage",
        "",
        "| Case | Full intervals evaluated | Certified | Unresolved | Collision | Resource hit | Certified fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in ("auto0", "auto1"):
        item = summary[case]
        lines.append(f"| {case} | {item['checked_intervals']}/{item['requested_intervals']} | {item['certified_intervals']} | {item['unresolved_intervals']} | {item['collision_intervals']} | {item['resource_limit_reached']} | {item['certified_fraction']:.4f} |")
    lines.extend([
        "",
        "No collision was found by this conservative shadow predicate, but that is not equivalent to a full PASS when unresolved regions remain. `minimum_clearance` and `minimum_certified_clearance` remain unavailable; projection gaps are not physical clearance.",
        "",
        "## Route outcome",
        "",
        "- Route A: executed a sound BVH whose leaves contain complete STL triangles; PCA OBBs tighten enclosures. No unverified convex-decomposition backend was substituted.",
        "- Route B: adaptive time subdivision passed a 445/500 auto0 hard-case probe at depth 4; full unresolved-set depth 2 was interrupted before a valid output. This is a compute boundary, not a PASS.",
        "- Route C: coarse BVH followed by interval-indexed refinement and fail-closed merge is implemented. The merge cannot erase collision, resource, or missing coverage.",
        "",
        "## D63/D64 boundary",
        "",
        "D59 motion, native execution, Profile.j, model dynamics, deterministic replay, FCL rigid endpoint cross-check, and MoveIt robot-world checks remain valid measured software evidence. D62 leaves the exact articulated continuous-certification P0 unresolved, so D64 can freeze the measured software evidence and limitations, but cannot claim physical safety closure or theorem-level full trajectory certification.",
        "",
        "Promotion: `NO_PROMOTION`; canonical and protected floor unchanged.",
    ])
    (OUT / "D62_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["D62_TASK_STATUS"], "output": str((OUT / "D62_FINAL_STATUS.json").resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
