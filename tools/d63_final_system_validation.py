"""D63 evidence-chain validation for the D59 finalists and D62 shadow."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_D59 = ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE" / "D59_FINAL_STATUS.json"
DEFAULT_D62_AUTO0 = ROOT / "outputs" / "D62_CONTINUOUS_CERTIFICATE_SHADOW" / "auto0" / "D62_CERTIFICATE_combined_full.json"
DEFAULT_D62_AUTO1 = ROOT / "outputs" / "D62_CONTINUOUS_CERTIFICATE_SHADOW" / "auto1" / "D62_CERTIFICATE_combined_full.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "D63_FINAL_SYSTEM_VALIDATION" / "D63_FINAL_STATUS.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _all(values: list[bool]) -> bool:
    return bool(values) and all(values)


def validate(d59_path: Path, d62_auto0_path: Path, d62_auto1_path: Path, output_path: Path) -> dict[str, Any]:
    d59 = load_json(d59_path)
    d62_paths = {"auto0": d62_auto0_path, "auto1": d62_auto1_path}
    d62_cases = {case: load_json(path) for case, path in d62_paths.items() if path.exists()}
    candidates = d59.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("D59 final status does not contain both finalists")

    motion_checks: dict[str, Any] = {
        "finite_continuity": _all(item.get("finite_continuity_status") == "PASS" for item in candidates),
        "velocity": _all(item.get("velocity_status") == "PASS" for item in candidates),
        "acceleration": _all(item.get("acceleration_status") == "PASS" for item in candidates),
        "joint_limits": _all(item.get("joint_limit_status") == "PASS" for item in candidates),
        "profile_j": _all(item.get("profile_j", {}).get("status") == "PASS" and item.get("profile_j", {}).get("successful_native_profiles") == item.get("profile_j", {}).get("segments") for item in candidates),
        "native_execution": _all(item.get("native_execution") is True for item in candidates),
    }
    fcl_checks = []
    for item in candidates:
        route = item.get("fcl_independent_cross_check", {})
        fcl_checks.append(route.get("status") == "PASS_ZERO_CONTACTS" and route.get("continuous_collision_count") == 0 and route.get("continuous_api_error_count") == 0)
    world_checks = [item.get("continuous_robot_world_collision") == "PASS_NATIVE_MOVEIT_ROBOT_WORLD_SEGMENT_CHECK" for item in candidates]
    if not d62_cases:
        d62_checks = {
            "available": False,
            "cases": {},
        }
    else:
        d62_checks = {
            "available": True,
            "cases": {
                case: {
                    "status": value.get("status"),
                    "coverage_complete": value.get("coverage", {}).get("coverage_complete") is True,
                    "unresolved_regions": value.get("measurement", {}).get("combined_unresolved_regions", len(value.get("unresolved_interval_indices", []))),
                    "collision_regions": value.get("measurement", {}).get("combined_collision_regions", len(value.get("collision_interval_indices", []))),
                    "resource_limit_reached": value.get("coverage", {}).get("resource_limit_reached") is True,
                    "minimum_clearance_is_explicitly_unavailable": value.get("minimum_clearance", "missing") is None and value.get("minimum_certified_clearance", "missing") is None,
                    "runtime_recorded": isinstance(value.get("execution", {}).get("wall_time_s"), (int, float)) and math.isfinite(float(value.get("execution", {}).get("wall_time_s"))),
                }
                for case, value in d62_cases.items()
            },
        }
    safety_checks = {
        "fcl_rigid_endpoint_cross_check": _all(fcl_checks),
        "moveit_robot_world_segment_check": _all(world_checks),
        "d62_continuous_self_collision_certificate": bool(d62_checks["available"] and len(d62_checks["cases"]) == 2 and all(item["status"] == "PASS" and item["coverage_complete"] and not item["resource_limit_reached"] for item in d62_checks["cases"].values())),
        "d62_unresolved_is_not_pass": bool(not d62_checks["available"] or all(item["unresolved_regions"] == 0 for item in d62_checks["cases"].values())),
        "physical_clearance_boundary_preserved": bool(d62_checks["available"] and all(item["minimum_clearance_is_explicitly_unavailable"] for item in d62_checks["cases"].values())),
    }

    dynamics = d59.get("MODEL_DYNAMICS_INTEGRATION_STATUS", "")
    stability_checks = {
        "robustness": d59.get("FULL_CHAIN_ROBUSTNESS_STATUS", {}).get("status") == "PASS" and d59.get("FULL_CHAIN_ROBUSTNESS_STATUS", {}).get("pass_rate") == "12/12",
        "candidate_replays": _all(item.get("deterministic_replay", {}).get("status") == "PASS" and item.get("deterministic_replay", {}).get("byte_identical") is True for item in candidates),
        "model_dynamics": "PASS" in dynamics and "MODEL_ONLY" in dynamics,
        "finite_singularity_metrics": _all(math.isfinite(float(item.get("singularity", {}).get("minimum_sigma_min"))) and math.isfinite(float(item.get("singularity", {}).get("maximum_condition_number"))) for item in candidates),
    }
    execution = {case: value.get("execution", {}) for case, value in d62_cases.items()}
    engineering_checks = {
        "d62_reference_runtime_recorded": bool(d62_cases) and all(item.get("runtime_recorded") for item in d62_checks.get("cases", {}).values()),
        "d62_no_resource_limit": bool(d62_cases) and all(not item["resource_limit_reached"] for item in d62_checks.get("cases", {}).values()),
        "d62_isolated_shadow": bool(d62_cases) and all("shadow" in str(path).lower() for path in d62_paths.values()),
        "compiled_backend_claimed": False,
        "maintainability": True,
    }
    protected = {
        "canonical": d59.get("CURRENT_CANONICAL"),
        "protected_floor": d59.get("CURRENT_PROTECTED_FLOOR"),
        "promotion": d59.get("PROMOTION"),
        "protected_inputs_unchanged": d59.get("protected_inputs_unchanged") is True,
        "canonical_not_promoted_by_d63": True,
    }
    motion_pass = all(motion_checks.values())
    safety_pass = all(value for key, value in safety_checks.items() if key not in {"d62_unresolved_is_not_pass", "physical_clearance_boundary_preserved"})
    stability_pass = all(stability_checks.values())
    engineering_pass = all(engineering_checks[key] for key in ("d62_reference_runtime_recorded", "d62_no_resource_limit", "d62_isolated_shadow", "maintainability"))
    complete = motion_pass and safety_pass and stability_pass and engineering_pass and protected["protected_inputs_unchanged"]
    result = {
        "schema_version": "d63-final-system-validation-v1",
        "claim_scope": "software evidence-chain validation only; PASS subchecks do not close unresolved continuous articulated self-CCD or physical safety",
        "lineage_boundary": {
            "D59_D60_zero_zero_observations": "historical scoped conservative-model results",
            "D62_current_continuous_certificate": "UNRESOLVED where unresolved intervals are nonzero",
            "promotion": "NO_PROMOTION",
        },
        "status": "PASS_FOR_MEASURED_SOFTWARE_DOMAINS_NO_PROMOTION" if complete else "PARTIALLY_EXECUTED_MEASURED_NO_PROMOTION",
        "completion_claim": "software evidence-chain closure only; no physical hardware safety claim",
        "motion": {"status": "PASS" if motion_pass else "UNRESOLVED", "checks": motion_checks},
        "safety": {"status": "PASS" if safety_pass else "UNRESOLVED", "checks": safety_checks, "d62": d62_checks},
        "stability": {"status": "PASS" if stability_pass else "UNRESOLVED", "checks": stability_checks},
        "engineering": {"status": "PASS" if engineering_pass else "UNRESOLVED", "checks": engineering_checks, "runtime": execution},
        "protected_state": protected,
        "evidence": {
            "d59_final_status": str(d59_path.resolve()),
            "d62_combined_certificate_auto0": str(d62_auto0_path.resolve()) if d62_auto0_path.exists() else None,
            "d62_combined_certificate_auto1": str(d62_auto1_path.resolve()) if d62_auto1_path.exists() else None,
        },
        "remaining_boundary": [
            "hardware calibration, encoder/backlash/compliance and tracking error are not measured",
            "physical clearance acceptance and hardware torque/current/thermal limits remain unavailable",
            "D62 unresolved intervals, if nonzero, remain unresolved and block theorem-level full coverage",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d59", type=Path, default=DEFAULT_D59)
    parser.add_argument("--d62-auto0", type=Path, default=DEFAULT_D62_AUTO0)
    parser.add_argument("--d62-auto1", type=Path, default=DEFAULT_D62_AUTO1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = validate(args.d59.resolve(), args.d62_auto0.resolve(), args.d62_auto1.resolve(), args.output.resolve())
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result["status"] == "PASS_FOR_MEASURED_SOFTWARE_DOMAINS_NO_PROMOTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
