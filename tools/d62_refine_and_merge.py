"""Run bounded D62 refinement on coarse unresolved intervals and merge evidence.

The refinement pass is deliberately interval-indexed.  It cannot erase a
coarse collision or turn a missing index into a pass; it can only replace a
specific coarse UNRESOLVED interval with a certified result when the same
fail-closed D62 predicate proves it at the requested adaptive depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.d62_bvh_interval_certificate import certify
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.d62_bvh_interval_certificate import certify


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def refine_from_coarse(
    coarse_path: Path,
    urdf_path: Path,
    srdf_path: Path | None,
    trajectory_path: Path,
    output_path: Path,
    trajectory_id: str,
    jerk_bound: float,
    threshold: float,
    max_leaf_points: int,
    max_node_pairs: int,
    max_time_depth: int,
) -> dict[str, Any]:
    coarse = _load(coarse_path)
    indices = coarse.get("unresolved_interval_indices")
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError("coarse certificate has no integer unresolved_interval_indices")
    result = certify(
        urdf_path.resolve(),
        srdf_path.resolve() if srdf_path else None,
        trajectory_path.resolve(),
        output_path.resolve(),
        trajectory_id,
        jerk_bound,
        threshold,
        max_leaf_points,
        max_node_pairs,
        max_time_depth,
        None,
        indices,
    )
    result["refinement_source"] = str(coarse_path.resolve())
    result["refinement_contract"] = {
        "only_coarse_unresolved_indices": True,
        "coarse_unresolved_count": len(indices),
        "refinement_indices": indices,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def merge_certificates(coarse_path: Path, refinement_path: Path, output_path: Path) -> dict[str, Any]:
    coarse = _load(coarse_path)
    refinement = _load(refinement_path)
    coarse_checked = set(coarse.get("checked_interval_indices", coarse.get("coverage", {}).get("selected_interval_indices", [])))
    refinement_checked = set(refinement.get("checked_interval_indices", refinement.get("coverage", {}).get("selected_interval_indices", [])))
    coarse_certified = set(coarse.get("certified_interval_indices", []))
    refinement_certified = set(refinement.get("certified_interval_indices", []))
    coarse_unresolved = set(coarse.get("unresolved_interval_indices", []))
    refinement_unresolved = set(refinement.get("unresolved_interval_indices", []))
    coarse_collision = set(coarse.get("collision_interval_indices", []))
    refinement_collision = set(refinement.get("collision_interval_indices", []))
    requested = int(coarse.get("checked_time_interval", {}).get("requested_interval_count", 0))
    certified = sorted(coarse_certified | refinement_certified)
    unresolved = sorted((coarse_unresolved - refinement_checked) | refinement_unresolved)
    collisions = sorted(coarse_collision | refinement_collision)
    checked = sorted(coarse_checked | refinement_checked)
    resource_limit = bool(coarse.get("coverage", {}).get("resource_limit_reached", False) or refinement.get("coverage", {}).get("resource_limit_reached", False))
    if collisions:
        status = "COLLISION_FOUND"
    elif unresolved or resource_limit or checked != list(range(requested)):
        status = "UNRESOLVED"
    else:
        status = "PASS"
    certificate = {
        "schema_version": "d62-bvh-combined-coarse-refinement-certificate-v1",
        "trajectory_id": coarse.get("trajectory_id"),
        "verification_domain": coarse.get("verification_domain"),
        "verification_result": status,
        "status": status,
        "certificate_type": "CONSERVATIVE_FK_INTERVAL_BVH_COARSE_PLUS_ADAPTIVE_REFINEMENT",
        "collision_method": "coarse_sound_mesh_BVH_OBB_enclosure_then_bounded_adaptive_interval_subdivision",
        "sampling_semantics": "trajectory rows define endpoint states; no sampled-only PASS is emitted",
        "checked_time_interval": {
            "requested_interval_count": requested,
            "checked_interval_count": len(checked),
            "coverage_complete": bool(checked == list(range(requested)) and not resource_limit),
        },
        "certified_interval_indices": certified,
        "unresolved_interval_indices": unresolved,
        "collision_interval_indices": collisions,
        "measurement": {
            "coarse": coarse.get("measurement", {}),
            "refinement": refinement.get("measurement", {}),
            "combined_certified_regions": len(certified),
            "combined_unresolved_regions": len(unresolved),
            "combined_collision_regions": len(collisions),
            "resource_limit_reached": resource_limit,
        },
        "coverage": {
            "coarse_certificate": str(coarse_path.resolve()),
            "refinement_certificate": str(refinement_path.resolve()),
            "checked_interval_count": len(checked),
            "requested_interval_count": requested,
            "coverage_complete": bool(checked == list(range(requested)) and not resource_limit),
            "resource_limit_reached": resource_limit,
            "unresolved_after_refinement": len(unresolved),
        },
        "minimum_clearance": None,
        "minimum_certified_clearance": None,
        "minimum_certified_projection_gap_m": min(
            [value for value in (coarse.get("minimum_certified_projection_gap_m"), refinement.get("minimum_certified_projection_gap_m")) if isinstance(value, (int, float))],
            default=None,
        ),
        "failure_witness": (refinement.get("failure_witness") or coarse.get("failure_witness")),
        "failure_witness_count": int(coarse.get("failure_witness_count", 0)) + int(refinement.get("failure_witness_count", 0)),
        "input_contract": coarse.get("input_contract", {}),
        "backend_status": coarse.get("backend_status", {}),
        "execution": {
            "coarse_wall_time_s": coarse.get("execution", {}).get("wall_time_s"),
            "refinement_wall_time_s": refinement.get("execution", {}).get("wall_time_s"),
            "wall_time_s": sum(value for value in (coarse.get("execution", {}).get("wall_time_s"), refinement.get("execution", {}).get("wall_time_s")) if isinstance(value, (int, float))),
            "implementation": "Python reference shadow; coarse plus interval-indexed bounded refinement",
        },
        "route_evidence": {
            "route_a_bvh": "EXECUTED_SOUND_TRIANGLE_PRESERVING_OBB_HIERARCHY",
            "route_b_adaptive_interval_subdivision": "EXECUTED_ONLY_ON_EXPLICIT_COARSE_UNRESOLVED_INDICES",
            "route_c_combination": "COARSE_BVH_THEN_BOUNDED_ADAPTIVE_REFINEMENT",
            "route_d_external_backend": "NOT_USED_AS_LOCAL_AUTHORITY",
        },
        "conservativeness_bound": coarse.get("conservativeness_bound", {}),
        "merge_contract": {
            "coarse_collision_cannot_be_erased": True,
            "uncovered_interval_cannot_be_pass": True,
            "unresolved_is_not_pass": True,
            "refinement_only_replaces_selected_coarse_unresolved": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate


def write_interrupted_refinement(coarse_path: Path, output_path: Path, reason: str) -> dict[str, Any]:
    """Record an interrupted refinement without presenting it as measurement."""

    coarse = _load(coarse_path)
    requested = int(coarse.get("coverage", {}).get("requested_interval_count", 0))
    result: dict[str, Any] = {
        "schema_version": "d62-bounded-refinement-interrupted-v1",
        "trajectory_id": coarse.get("trajectory_id"),
        "status": "UNRESOLVED",
        "verification_result": "UNRESOLVED",
        "interrupted": True,
        "interruption_reason": reason,
        "checked_interval_indices": [],
        "certified_interval_indices": [],
        "unresolved_interval_indices": [],
        "collision_interval_indices": [],
        "measurement": {"executed": False, "refined_regions": 0},
        "coverage": {
            "requested_interval_count": requested,
            "selected_interval_indices": [],
            "selected_interval_count": 0,
            "coverage_complete": False,
            "resource_limit_reached": False,
        },
        "execution": {"wall_time_s": 0.0, "implementation": "no refinement result; interrupted before evidence production"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    refine = subparsers.add_parser("refine")
    refine.add_argument("--coarse", type=Path, required=True)
    refine.add_argument("--urdf", type=Path, required=True)
    refine.add_argument("--srdf", type=Path, default=None)
    refine.add_argument("--trajectory", type=Path, required=True)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--trajectory-id", required=True)
    refine.add_argument("--jerk-bound", type=float, default=8.0)
    refine.add_argument("--threshold", type=float, default=0.0)
    refine.add_argument("--max-leaf-points", type=int, default=256)
    refine.add_argument("--max-node-pairs", type=int, default=5000000)
    refine.add_argument("--max-time-depth", type=int, default=4)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--coarse", type=Path, required=True)
    merge.add_argument("--refinement", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    aborted = subparsers.add_parser("aborted")
    aborted.add_argument("--coarse", type=Path, required=True)
    aborted.add_argument("--output", type=Path, required=True)
    aborted.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        if args.command == "refine":
            result = refine_from_coarse(args.coarse, args.urdf, args.srdf, args.trajectory, args.output, args.trajectory_id, args.jerk_bound, args.threshold, args.max_leaf_points, args.max_node_pairs, args.max_time_depth)
        elif args.command == "merge":
            result = merge_certificates(args.coarse, args.refinement, args.output)
        else:
            result = write_interrupted_refinement(args.coarse, args.output, args.reason)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": result.get("status"), "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
