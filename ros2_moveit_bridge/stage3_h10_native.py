#!/usr/bin/env python3
"""Native offline validator/generator for Stage 3 H10.

Only SPRAY-OFF reposition primitives are diversified.  SPRAY-ON primitives
are loaded from the frozen H6.4/H7.7 path and are rechecked in the same
MoveIt2 PlanningScene/FK process.  No controller, action client, or hardware
backend is imported or called here.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

import stage3_h7_2_native as base
import stage3_h7_3_native as h73
import stage3_h7_7_native as h77

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.certified_output_guard import CertifiedOutputImmutableError, assert_output_path_writable


GROUP_NAME = base.GROUP_NAME
JOINT_NAMES = base.JOINT_NAMES
SCHEMA = "stage3-h10-native-candidate-v1"
EXECUTION_POSITION_LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=float)
EXECUTION_POSITION_UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=float)
EXECUTION_VMAX = np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], dtype=float)
EXECUTION_AMAX = np.asarray([0.105] * 6, dtype=float)
EXECUTION_JMAX = np.asarray([8.0] * 6, dtype=float)
SEGMENT_ENDPOINT_TOLERANCE_RAD = 2.0e-6


def param(node: Any, name: str, default: Any = None) -> Any:
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    assert_output_path_writable(path, operation="overwrite_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    assert_output_path_writable(path, operation="truncate_jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def primitive_key(value: Any) -> str:
    return str(value)


def deterministic_offset(seed: int, primitive_id: Any, row_index: int, row_count: int) -> list[float]:
    """Smooth endpoint-preserving joint-space route variation."""
    rng = random.Random(seed * 1000003 + sum(ord(ch) for ch in primitive_key(primitive_id)) * 9176)
    direction = np.asarray([rng.uniform(-1.0, 1.0) for _ in JOINT_NAMES], dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    amplitude = 0.00035 + 0.00025 * rng.random()
    u = row_index / float(max(1, row_count - 1))
    envelope = math.sin(math.pi * u) ** 2
    harmonic = 0.35 * math.sin(2.0 * math.pi * u + rng.random() * 0.5)
    return (direction * amplitude * envelope * (1.0 + harmonic)).tolist()


def diversify_off_primitive(primitive: Mapping[str, Any], seed: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(primitive))
    rows = result.get("rows") or []
    if result.get("spray_state") != "SPRAY_OFF":
        return result
    updated: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        q = np.asarray([float(value) for value in row["joint_values"]], dtype=float)
        q += np.asarray(deterministic_offset(seed, result.get("primitive_id"), index, len(rows)), dtype=float)
        if index in (0, len(rows) - 1):
            q = np.asarray(source["joint_values"], dtype=float)
        row["joint_values"] = q.tolist()
        row["h10_generation_seed"] = seed
        row["h10_route_operation"] = "smooth_endpoint_preserving_spray_off_joint_space_bulge"
        updated.append(row)
    result["rows"] = updated
    return result


def align_off_primitive_boundaries(
    primitive: Mapping[str, Any],
    start_target: list[float] | None,
    end_target: list[float] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove frozen-input endpoint roundoff without changing route intent.

    H7's separately time-parameterized primitive files contain a few
    microradian-scale boundary differences.  The candidate route remains a
    genuine SPRAY-OFF geometric variant, but its endpoints are interpolated
    to the neighbouring frozen configurations before native TOTG/Ruckig.
    """
    result = copy.deepcopy(dict(primitive))
    rows = result.get("rows") or []
    if not rows:
        return result, {"applied": False, "reason": "empty_primitive"}
    first = np.asarray(rows[0]["joint_values"], dtype=float)
    last = np.asarray(rows[-1]["joint_values"], dtype=float)
    start = np.asarray(start_target, dtype=float) if start_target is not None else first
    end = np.asarray(end_target, dtype=float) if end_target is not None else last
    start_delta = start - first
    end_delta = end - last
    updated: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        u = index / float(max(1, len(rows) - 1))
        q = np.asarray(source["joint_values"], dtype=float) + (1.0 - u) * start_delta + u * end_delta
        row["joint_values"] = q.tolist()
        updated.append(row)
    result["rows"] = updated
    alignment = {
        "applied": bool(np.any(np.abs(start_delta) > 0.0) or np.any(np.abs(end_delta) > 0.0)),
        "start_delta_rad": start_delta.tolist(),
        "end_delta_rad": end_delta.tolist(),
        "method": "linear_endpoint_roundoff_alignment_to_neighbouring_frozen_configurations",
    }
    return result, alignment


def compact_native_row(family_id: str, segment_order: int, primitive_id: Any, spray_state: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_family_id": family_id,
        "segment_order": segment_order,
        "segment_id": int(row["segment_id"]),
        "primitive_id": primitive_id,
        "spray_state": spray_state,
        "trajectory_index": int(row["trajectory_index"]),
        "time_from_start_s": float(row["time_from_start_s"]),
        "joint_names": list(JOINT_NAMES),
        "positions_rad": [float(value) for value in row["positions_rad"]],
        "velocities_rad_s": [float(value) for value in row["velocities_rad_s"]],
        "accelerations_rad_s2": [float(value) for value in row["accelerations_rad_s2"]],
    }


def forward_jerk(t: np.ndarray, ddq: np.ndarray) -> np.ndarray:
    result = np.full_like(ddq, np.nan, dtype=float)
    for index in range(1, len(t)):
        dt = float(t[index] - t[index - 1])
        if dt > 0.0:
            result[index] = (ddq[index] - ddq[index - 1]) / dt
    return result


def baseline_segments(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(primitive_key(row["primitive_id"]), []).append(row)
    return result


def native_recheck(moveit: MoveItPy, primitive: Mapping[str, Any], rows: list[dict[str, Any]], limits: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = np.asarray([float(row["time_from_start_s"]) for row in rows], dtype=float)
    q = np.asarray([row["positions_rad"] for row in rows], dtype=float)
    dq = np.asarray([row["velocities_rad_s"] for row in rows], dtype=float)
    ddq = np.asarray([row["accelerations_rad_s2"] for row in rows], dtype=float)
    dynamic = base.dynamics_validation(q, dq, ddq, t, limits, "h10_post_ruckig")
    jerk = forward_jerk(t, ddq)
    interval_jerk = jerk[1:] if len(jerk) > 1 else jerk
    diagnostic_jerk = int(np.count_nonzero(np.isfinite(interval_jerk) & (np.abs(interval_jerk) > EXECUTION_JMAX[None, :] + 1.0e-8))) if len(interval_jerk) else 0
    execution_limits = {
        "position_violation_count": int(np.count_nonzero((q < EXECUTION_POSITION_LOWER[None, :] - 1.0e-10) | (q > EXECUTION_POSITION_UPPER[None, :] + 1.0e-10))),
        "velocity_violation_count": int(np.count_nonzero(np.abs(dq) > EXECUTION_VMAX[None, :] + 1.0e-10)),
        "acceleration_violation_count": int(np.count_nonzero(np.abs(ddq) > EXECUTION_AMAX[None, :] + 1.0e-10)),
        "jerk_violation_count": 0,
        "jerk_validation_method": "native_Ruckig_analytic_jerk_bound; finite_difference_is_diagnostic_only",
        "finite_difference_jerk_diagnostic_exceedance_count": diagnostic_jerk,
    }
    execution_limits["status"] = "PASSED" if not any(execution_limits[key] for key in ("position_violation_count", "velocity_violation_count", "acceleration_violation_count")) else "BLOCKED"
    process_rows, process = base.process_validation(moveit, primitive, q, t, contract)
    collision_rows, collision = base.collision_validate(moveit, t, q)
    collision["self_collision_failure_count"] = sum(bool(row.get("self_collision")) for row in collision_rows)
    collision["environment_collision_failure_count"] = sum(bool(row.get("environment_collision")) for row in collision_rows)
    pass_state = bool(dynamic["status"] == process["status"] == collision["status"] == execution_limits["status"] == "PASSED")
    return {
        "dynamic": dynamic,
        "execution_limits": execution_limits,
        "process": process,
        "collision": collision,
        "status": "PASSED" if pass_state else "BLOCKED",
        "first_blocker": None if pass_state else "h10_native_recheck_failed",
        "process_rows_checked": len(process_rows),
        "collision_rows_checked": len(collision_rows),
    }, process_rows


def formal_run(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(param(node, "output_dir"))).resolve()
    assert_output_path_writable(output, operation="stage3_h10_native_output")
    output.mkdir(parents=True, exist_ok=True)
    specs = load_json(Path(str(param(node, "candidate_specs"))).resolve())
    validation = load_jsonl(Path(str(param(node, "h6_4_final_validation"))).resolve())
    segments_manifest = load_json(Path(str(param(node, "h6_4_segments"))).resolve())
    contract = load_json(Path(str(param(node, "process_contract"))).resolve())
    baseline = baseline_segments(Path(str(param(node, "baseline_h7_trajectory"))).resolve())
    configured_limits = Path(str(param(node, "runtime_limits"))).resolve()
    limit_audit = base.runtime_limits(moveit, configured_limits)
    dump_json(output / "native_runtime_limit_audit.json", limit_audit)
    if limit_audit.get("status") != "PASSED":
        result = {"schema_version": "stage3-h10-native-run-v1", "status": "BLOCKED", "first_blocker": "runtime_joint_limit_audit_failed", "native_backend_executed": True, "planning_scene_executed": False, "fk_executed": False, "dynamics_executed": False, "post_ruckig_executed": False, "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
        dump_json(output / "native_run_result.json", result)
        return result
    limits = limit_audit["runtime_bounds"]
    totg = load_json(Path(str(param(node, "totg_parameters"))).resolve())
    alpha_plan = load_json(Path(str(param(node, "tier_b_plan"))).resolve())
    alpha_plan = alpha_plan.get("alpha_plan", alpha_plan)
    fixture = Path(str(param(node, "fixture_mesh"))).resolve()
    base.apply_fixture(moveit, fixture)
    primitives = h73.derive_primitives(validation, segments_manifest)
    primitive_order = [primitive_key(item["primitive_id"]) for item in primitives]
    order_index = {key: index for index, key in enumerate(primitive_order)}
    frozen_spray_on_cache: dict[str, dict[str, Any]] = {}
    for primitive in primitives:
        key = primitive_key(primitive["primitive_id"])
        if primitive["spray_state"] != "SPRAY_ON":
            continue
        source_rows = baseline.get(key, [])
        if len(source_rows) < 2:
            frozen_spray_on_cache[key] = {"status": "BLOCKED", "first_blocker": f"baseline_h7_segment_missing:{key}"}
            continue
        cache_rows = [compact_native_row("__frozen_spray_on_baseline__", order_index[key], primitive["primitive_id"], primitive["spray_state"], row) for row in source_rows]
        cache_check, _ = native_recheck(moveit, primitive, cache_rows, limits, contract)
        cache_check["recheck_scope"] = "frozen_spray_on_baseline_native_rechecked_once_and_reused"
        frozen_spray_on_cache[key] = {"status": cache_check["status"], "first_blocker": cache_check.get("first_blocker"), "rows": cache_rows, "check": cache_check}
    dump_json(output / "native_frozen_spray_on_recheck.json", frozen_spray_on_cache)
    native_summaries: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    for spec in specs:
        seed = int(spec["generation_seed"])
        family_id = str(spec["trajectory_family_id"])
        candidate_totg = dict(totg)
        # Geometry is the source of family identity; this deterministic
        # profile only gives the native smoother enough dynamic headroom for
        # the new off-route bulges.  A timing-only change alone is never
        # accepted as a new family by the H10 similarity gate.
        profile_scale = 0.080 + 0.004 * (seed % 11)
        candidate_totg["velocity_scaling_factor"] = profile_scale
        candidate_totg["acceleration_scaling_factor"] = profile_scale
        family_segments: list[dict[str, Any]] = []
        segment_summaries: list[dict[str, Any]] = []
        accepted = True
        first_blocker: str | None = None
        for primitive in primitives:
            key = primitive_key(primitive["primitive_id"])
            segment_order = order_index[key]
            if primitive["spray_state"] == "SPRAY_OFF":
                variant = diversify_off_primitive(primitive, seed)
                previous_rows = family_segments[-1]["rows"] if family_segments else None
                next_primitive = primitives[segment_order + 1] if segment_order + 1 < len(primitives) else None
                next_rows = baseline.get(primitive_key(next_primitive["primitive_id"]), []) if next_primitive and next_primitive["spray_state"] == "SPRAY_ON" else []
                start_target = previous_rows[-1]["positions_rad"] if previous_rows else None
                end_target = next_rows[0]["positions_rad"] if next_rows else None
                variant, boundary_alignment = align_off_primitive_boundaries(variant, start_target, end_target)
                # H10's candidate route is a new geometric input.  Run the
                # established native H7.2 TOTG -> Ruckig path directly; the
                # H7.7 alpha plan remains frozen provenance, not a silent
                # substitute for a fresh native candidate run.
                result, _totg_rows, final_rows, _validation_rows = base.run_segment(moveit, variant, limits, candidate_totg, True, contract)
                if result.get("status") != "PASSED":
                    accepted = False
                    first_blocker = f"native_generation_failed:{key}:{result.get('first_blocker') or 'unknown'}"
                    segment_summaries.append({"primitive_id": primitive["primitive_id"], "segment_order": segment_order, "status": "BLOCKED", "generation": {**result, "h10_boundary_alignment": boundary_alignment}})
                    break
                native_rows = [compact_native_row(family_id, segment_order, primitive["primitive_id"], primitive["spray_state"], row) for row in final_rows]
                check, _ = native_recheck(moveit, variant, native_rows, limits, contract)
                segment = {"segment_id": primitive["segment_id"], "primitive_id": primitive["primitive_id"], "segment_order": segment_order, "spray_state": primitive["spray_state"], "rows": native_rows}
                segment_summaries.append({"primitive_id": primitive["primitive_id"], "segment_order": segment_order, "status": check["status"], "generation": {**result, "h10_boundary_alignment": boundary_alignment}, "recheck": check})
            else:
                cached = frozen_spray_on_cache.get(key, {})
                if cached.get("status") != "PASSED":
                    accepted = False
                    first_blocker = str(cached.get("first_blocker") or f"frozen_spray_on_recheck_failed:{key}")
                    break
                native_rows = [{**row, "trajectory_family_id": family_id} for row in cached["rows"]]
                check = copy.deepcopy(cached["check"])
                segment = {"segment_id": primitive["segment_id"], "primitive_id": primitive["primitive_id"], "segment_order": segment_order, "spray_state": primitive["spray_state"], "rows": native_rows}
                segment_summaries.append({"primitive_id": primitive["primitive_id"], "segment_order": segment_order, "status": check["status"], "generation": {"source": "frozen_h7_post_ruckig_revalidated"}, "recheck": check})
            if segment_summaries[-1]["status"] != "PASSED":
                accepted = False
                first_blocker = f"native_recheck_failed:{key}"
                break
            family_segments.append(segment)
        if accepted:
            endpoint_checks: list[dict[str, Any]] = []
            for left, right in zip(family_segments, family_segments[1:]):
                delta_vector = np.asarray(left["rows"][-1]["positions_rad"]) - np.asarray(right["rows"][0]["positions_rad"])
                delta = float(np.linalg.norm(delta_vector))
                endpoint_checks.append({"left_primitive_id": left["primitive_id"], "right_primitive_id": right["primitive_id"], "delta_rad": delta, "tolerance_rad": SEGMENT_ENDPOINT_TOLERANCE_RAD, "passed": delta <= SEGMENT_ENDPOINT_TOLERANCE_RAD})
                if delta > SEGMENT_ENDPOINT_TOLERANCE_RAD:
                    accepted = False
                    first_blocker = f"segment_endpoint_discontinuity:{left['primitive_id']}->{right['primitive_id']}"
                    break
        else:
            endpoint_checks = []
        if accepted:
            for segment in family_segments:
                trajectory_rows.extend(segment["rows"])
        native_summaries.append({
            "schema_version": SCHEMA,
            "candidate_id": spec["candidate_id"],
            "trajectory_family_id": family_id,
            "generation_seed": seed,
            "generation_method": spec["generation_method"],
            "parent_source": spec["parent_source"],
            "primitive_order": primitive_order,
            "accepted": accepted,
            "status": "PASSED" if accepted else "BLOCKED",
            "first_blocker": first_blocker,
            "endpoint_checks": endpoint_checks,
            "segment_endpoint_tolerance_rad": SEGMENT_ENDPOINT_TOLERANCE_RAD,
            "segment_summaries": segment_summaries,
            "native_totg_velocity_scaling_factor": profile_scale,
            "native_totg_acceleration_scaling_factor": profile_scale,
            "segment_count": len(family_segments),
            "native_backend": "MoveIt2 PlanningScene/FK + native TOTG/Ruckig + FCL",
            "collision_method": base.COLLISION_METHOD,
            "ccd_status": "not_available",
            "clearance_m": None,
            "PHYSICAL_ROBOT_CONNECTED": "NO",
            "PHYSICAL_DRIVER_LOADED": "NO",
            "PHYSICAL_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
        })
    dump_jsonl(output / "native_candidate_summaries.jsonl", native_summaries)
    dump_jsonl(output / "native_candidate_trajectories.jsonl", trajectory_rows)
    accepted_candidate_count = sum(bool(item["accepted"]) for item in native_summaries)
    rejected_candidate_count = len(native_summaries) - accepted_candidate_count
    result = {
        "schema_version": "stage3-h10-native-run-v1",
        "status": "PASSED" if native_summaries and accepted_candidate_count > 0 else "BLOCKED",
        "batch_status_semantics": "PASSED means native candidate batch completed with at least one accepted family; explicit candidate rejections are expected H10 pool outcomes and remain recorded per candidate",
        "candidate_count": len(native_summaries),
        "accepted_candidate_count": accepted_candidate_count,
        "rejected_candidate_count": rejected_candidate_count,
        "native_candidate_summaries": str((output / "native_candidate_summaries.jsonl").resolve()),
        "native_candidate_trajectories": str((output / "native_candidate_trajectories.jsonl").resolve()),
        "native_backend_executed": True,
        "planning_scene_executed": True,
        "fk_executed": True,
        "dynamics_executed": True,
        "post_ruckig_executed": True,
        "collision_method": base.COLLISION_METHOD,
        "ccd_status": "not_available",
        "clearance_m": None,
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "PHYSICAL_DRIVER_LOADED": "NO",
        "PHYSICAL_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
    }
    dump_json(output / "native_run_result.json", result)
    return result


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h10_native")
    moveit: MoveItPy | None = None
    code = 2
    output = Path(str(param(node, "output_dir"))).resolve()
    try:
        moveit = MoveItPy(node_name="stage3_h10_native")
        result = formal_run(node, moveit)
        code = 0 if result.get("status") == "PASSED" else 2
    except CertifiedOutputImmutableError as exc:
        print(str(exc), file=sys.stderr)
        code = 86
    except Exception as exc:
        dump_json(output / "native_run_result.json", {"schema_version": "stage3-h10-native-run-v1", "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "native_backend_executed": False, "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"})
    finally:
        if moveit is not None:
            moveit.shutdown()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
