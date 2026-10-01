"""C4 gated native replay: identity equivalence first, then 12 small smoke cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p2b3_c3_native_smoke import run as run_shared_native_evaluator  # noqa: E402
from p2b3_c4_scene_contract import (  # noqa: E402
    compare_c3_to_c1_parameters,
    extract_c1_scene_contract,
    extract_legacy_c3_scene_contract,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = run_shared_native_evaluator(args)
    c1_result = json.loads(args.c1_result.read_text(encoding="utf-8"))
    c3_result = json.loads((ROOT / "outputs" / "p2b3_c3_result.json").read_text(encoding="utf-8"))
    c1_contract = extract_c1_scene_contract(
        c1_result,
        (ROOT / "scripts" / "run_p2b3_c1_r0.py").read_text(encoding="utf-8"),
        (ROOT / "ros2_moveit_bridge" / "plan_closed_contour_moveit.py").read_text(encoding="utf-8"),
    )
    legacy_c3 = extract_legacy_c3_scene_contract(ROOT, c3_result)
    drift = compare_c3_to_c1_parameters(legacy_c3, c1_contract)
    old_smoke = c3_result.get("native_smoke", {})
    old_cases = old_smoke.get("cases", []) if isinstance(old_smoke, dict) else []
    old_zero = old_cases[0] if old_cases else {}
    result.update({
        "P2B3_C4_STATUS": "PASS" if result.get("status") == "PASS" else "BLOCKED_OR_INCOMPLETE",
        "scene_parameter_drift_found": drift["status"] == "DRIFT_FOUND",
        "scene_parameter_drift": drift,
        "scene_drift_root_cause": {
            "classification": "C3_NOMINAL_SCENE_SOURCE_MISMATCH" if drift["status"] == "DRIFT_FOUND" else "NO_PARAMETER_DRIFT_FOUND",
            "c3_historical_builder_parameters": legacy_c3,
            "c1_authoritative_parameters": dict(c1_contract.parameters),
            "c3_old_zero_collision_samples": old_zero.get("full_scene_collision_sample_count"),
            "c3_old_zero_sample_count": old_zero.get("sample_count"),
            "c3_old_scene_object_count": old_smoke.get("scene_geometry", {}).get("collision_object_count"),
            "c4_zero_collision_samples": result.get("zero_baseline_collision_count"),
            "causal_scope": "A zero-collision C4 replay with the authoritative C1 contract strongly attributes the C3 saturation conflict to scene mismatch; individual C3 parameters were not isolated in a one-factor experiment.",
        },
        "legacy_c3_distance_reinterpretation": {
            "raw_value_m": old_zero.get("minimum_reported_fcl_distance_m"),
            "old_label": old_zero.get("fcl_distance_status"),
            "corrected_status": "NOT_AVAILABLE_SENTINEL",
            "sentinel_rejected": old_zero.get("minimum_reported_fcl_distance_m") == -1.0,
            "reason": "MoveIt CollisionResult.distance=-1.0 is a sentinel in the recorded colliding C3 run, not a physical distance or penetration depth.",
        },
        "registration_margin_readiness": "READY" if result.get("zero_transform_equivalence", {}).get("status") == "PASS" and result.get("nonzero_engineering_smoke") == "COMPLETE_12_CASES" else "BLOCKED",
        "physical_registration_margin": "NOT_RUN",
        "physical_normalized_margin": "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        "physical_registration_bound": "UNAVAILABLE_NO_MEASURED_FR5_BOUND",
        "strict_continuous_ccd": "NOT_AVAILABLE",
        "hardware_validated": "NOT_RUN",
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--c1-fk-trace", type=Path, required=True)
    parser.add_argument("--c1-result", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--joint-limits", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--identity-gate-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--phase", choices=("identity", "smoke"), required=True)
    args = parser.parse_args()
    args.stage = "P2-B3-C4"
    try:
        result = run(args)
        identity_phase_pass = args.phase == "identity" and result.get("status") == "IDENTITY_GATE_PASS_PHASE_D_PENDING"
        exit_code = 0 if result.get("status") == "PASS" or identity_phase_pass else 2
    except Exception as exc:
        result = {
            "schema_version": "p2b3-c4-native-smoke-v1",
            "project": "FAIRINO_FR5",
            "stage": "P2-B3-C4",
            "status": "INCOMPLETE_NATIVE_VALIDATION",
            "native_scene_propagation_tested": "NO",
            "reason": f"native_exception:{type(exc).__name__}:{exc}",
            "strict_continuous_ccd": "NOT_AVAILABLE",
            "hardware_validation": "NOT_RUN",
            "nonzero_engineering_smoke": "NOT_RUN_IDENTITY_GATE_UNRESOLVED",
            "registration_margin_campaign": "NOT_RUN",
        }
        exit_code = 2
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in (
        "status", "zero_transform_equivalence", "zero_baseline_collision_count",
        "c1_expected_collision_count", "nonzero_engineering_smoke", "scene_manifest_sha256", "reason",
    )}, sort_keys=True), flush=True)
    # MoveItPy teardown is known to fault in this ROS environment; close artifacts before exiting.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
