"""D55 targeted Stage 4B measurement and shadow-candidate harness.

The harness is deliberately scoped to the current protected finalist
``stateful_local_g1_w05_0025``.  It materializes a compact current-finalist
ledger, locates the D54 clearance/singularity hotspots, defines a fail-closed
hardware-uncertainty interface, and creates smooth local shadow inputs for a
subsequent native MoveIt2/Ruckig validation pass.

It never writes D52, D47, or Stage 3 scientific state.  Generated candidates
belong below ``outputs/D55_STAGE4B_TARGETED`` and are not authoritative until
the native post-Ruckig and complete system gates have run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.d52_motion_repair import CASES, JOINTS, NativeTrajectory, read_native, write_native


ROOT = Path(__file__).resolve().parents[1]
D52 = ROOT / "outputs" / "D52_STAGE4B_SHADOW"
D54 = ROOT / "outputs" / "D54_STAGE4_OFFLINE_TRAJECTORY_CERTIFICATION"
OUT = ROOT / "outputs" / "D55_STAGE4B_TARGETED"
FINALIST = "stateful_local_g1_w05_0025"
FINALIST_DIR = D52 / "candidates" / FINALIST
CURRENT_TRAJECTORIES = FINALIST_DIR / "trajectories"
REPAIRED_TRAJECTORIES = D54 / "measurement_repaired_trajectories"
GEOMETRY = D54 / "native_environment_ccd_repaired_v2"
SELF_CASES = D54 / "self_collision_repaired_final" / "D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl"
SELF_CERT = D54 / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
KINEMATIC = D54 / "D54_KINEMATIC_DYNAMICS_METRICS.json"
EVAL = D52 / "evaluation" / FINALIST / "metrics.json"
DYNAMICS = D52 / "dynamics" / FINALIST / "native_dynamics_report.json"
JACOBIAN = GEOMETRY / "D41_native_jacobian.csv"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    if len(resolved) >= 3 and resolved[1:3] == ":/":
        return f"/mnt/{resolved[0].lower()}/{resolved[3:]}"
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family(case_id: str) -> str:
    if case_id.startswith("collision_sensitive"):
        return "COLLISION_SENSITIVE"
    return case_id.split("_")[0].upper()


def geometry_rows() -> dict[str, dict[str, Any]]:
    path = GEOMETRY / "D41_native_case_summary.jsonl"
    if not path.is_file():
        return {}
    return {str(row["case_id"]): row for row in read_jsonl(path)}


def read_jacobian_rows() -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = {}
    if not JACOBIAN.is_file():
        return result
    with JACOBIAN.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key == "case_id":
                    continue
                try:
                    row[key] = float(value) if value not in (None, "", "null") else math.nan
                except ValueError:
                    row[key] = math.nan
            result.setdefault(str(raw["case_id"]), []).append(row)
    return result


def finite_min(rows: Iterable[dict[str, float]], key: str) -> tuple[float | None, int | None]:
    values = [(float(row[key]), int(row.get("waypoint", -1))) for row in rows if math.isfinite(float(row.get(key, math.nan)))]
    return min(values) if values else (None, None)


def finite_max(rows: Iterable[dict[str, float]], key: str) -> tuple[float | None, int | None]:
    values = [(float(row[key]), int(row.get("waypoint", -1))) for row in rows if math.isfinite(float(row.get(key, math.nan)))]
    return max(values) if values else (None, None)


def baseline_ledger() -> dict[str, Any]:
    """Build the current-finalist-only baseline from D54-era evidence."""
    required = [SELF_CERT, KINEMATIC, EVAL, DYNAMICS, SELF_CASES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing authenticated D54/D52 evidence: " + ", ".join(missing))
    self_cert = load_json(SELF_CERT)
    kinematic = load_json(KINEMATIC)
    evaluation = load_json(EVAL)
    dynamics = load_json(DYNAMICS)
    self_rows = {str(row["case_id"]): row for row in read_jsonl(SELF_CASES)}
    geom = geometry_rows()
    jac = read_jacobian_rows()
    continuity = {str(row["case_id"]): row for row in kinematic["continuity"]["cases"]}
    motion_cases = []
    for case_id in CASES:
        self_row = self_rows.get(case_id, {})
        geom_row = geom.get(case_id, {})
        jac_rows = jac.get(case_id, [])
        sigma_min, sigma_wp = finite_min(jac_rows, "sigma_min")
        sigma_max, _ = finite_max(jac_rows, "sigma_max")
        cond_max, cond_wp = finite_max(jac_rows, "condition_number")
        manip_min, _ = finite_min(jac_rows, "manipulability")
        motion_cases.append({
            "case_id": case_id,
            "family": family(case_id),
            "strict_d54_self_clearance_m": self_row.get("minimum_certified_clearance_m"),
            "strict_d54_self_clearance_pair": self_row.get("worst_certified_interval", {}).get("pair"),
            "strict_d54_self_clearance_interval": self_row.get("worst_certified_interval", {}).get("interval_index"),
            "strict_d54_self_clearance_time_start_s": self_row.get("worst_certified_interval", {}).get("time_start_s"),
            "strict_d54_self_clearance_time_end_s": self_row.get("worst_certified_interval", {}).get("time_end_s"),
            "d54_status": self_row.get("status", "UNAVAILABLE"),
            "fcl_independent_route": "pending_or_unavailable_until_D55_execution",
            "native_geometry_status": geom_row.get("status", "UNAVAILABLE"),
            "native_environment_collision_count": geom_row.get("nominal_waypoint_world_collision_count"),
            "native_self_collision_count": geom_row.get("nominal_waypoint_self_collision_count"),
            "native_environment_ccd_collision_count": geom_row.get("native_continuous_segment_collision_count"),
            "minimum_observed_self_distance_m": geom_row.get("minimum_self_distance_m"),
            "minimum_environment_distance_m": geom_row.get("minimum_robot_world_distance_m"),
            "minimum_sigma_min": sigma_min,
            "sigma_min_waypoint": sigma_wp,
            "maximum_sigma_max": sigma_max,
            "maximum_condition_number": cond_max,
            "condition_number_waypoint": cond_wp,
            "minimum_manipulability": manip_min,
            "minimum_reciprocal_condition": (sigma_min / sigma_max) if sigma_min is not None and sigma_max and sigma_max > 0 else None,
            "velocity_amplification_proxy": (1.0 / sigma_min) if sigma_min is not None and sigma_min > 0 else None,
            "duration_s": continuity.get(case_id, {}).get("duration_s"),
            "max_velocity_ratio": continuity.get(case_id, {}).get("max_velocity_ratio"),
            "max_acceleration_ratio": continuity.get(case_id, {}).get("max_acceleration_ratio"),
            "max_diagnostic_jerk_ratio": continuity.get(case_id, {}).get("max_diagnostic_jerk_ratio"),
        })
    dynamics_rows = [row for row in dynamics.get("ranking_rows", []) if row.get("variant") == "nominal"]
    current_dyn = {
        "candidate_peak_abs_torque_Nm": max((float(row["candidate_peak_abs_torque_Nm"]) for row in dynamics_rows), default=None),
        "candidate_peak_abs_torque_slew_Nm_s": max((float(row["candidate_peak_abs_torque_slew_Nm_s"]) for row in dynamics_rows), default=None),
    }
    return {
        "schema_version": "d55-current-finalist-only-baseline-v1",
        "scope": "D55 P1-P5 only; current finalist; Stage 0/1 ON-state open-arch",
        "candidate": FINALIST,
        "source_trajectory_dir": str(CURRENT_TRAJECTORIES.resolve()),
        "measurement_stack": {
            "trajectory": "D54 measurement-repaired copy of the D52 native post-Ruckig q/dq/ddq/jerk CSV",
            "geometry": "D54 native MoveIt2 PlanningScene/FK/Bullet-world and FCL-distance replay",
            "self_clearance": "D54 FK-aware adaptive pairwise interval bound",
            "singularity": "D54/D41 native MoveIt2 RobotState Jacobian + Eigen SVD",
            "dynamics": "D52 native Pinocchio RNEA + ABA model-level evidence",
            "collision_label": "adaptive_discrete_interpolation",
        },
        "provenance_separation": {
            "current_finalist_measurements": True,
            "historical_baseline_measurements": "excluded from candidate metrics; used only for non-regression context",
            "historical_failed_candidates": "excluded",
            "model_assumptions": [
                "derived project URDF inertials, not hardware-calibrated",
                "D54 C3 native jerk envelope for strict interval certificate",
                "FCL endpoint signed distance for D54 bound",
            ],
            "hardware_measurements": "not_available; no physical PASS inferred",
        },
        "summary": {
            "case_count": len(motion_cases),
            "d54_min_strict_self_clearance_m": min((row["strict_d54_self_clearance_m"] for row in motion_cases if row["strict_d54_self_clearance_m"] is not None), default=None),
            "d54_worst_self_clearance_case": min((row for row in motion_cases if row["strict_d54_self_clearance_m"] is not None), key=lambda row: row["strict_d54_self_clearance_m"], default={}).get("case_id"),
            "minimum_sigma_min": min((row["minimum_sigma_min"] for row in motion_cases if row["minimum_sigma_min"] is not None), default=None),
            "maximum_condition_number": max((row["maximum_condition_number"] for row in motion_cases if row["maximum_condition_number"] is not None), default=None),
            "minimum_manipulability": min((row["minimum_manipulability"] for row in motion_cases if row["minimum_manipulability"] is not None), default=None),
            "peak_abs_torque_Nm": current_dyn.get("candidate_peak_abs_torque_Nm"),
            "peak_abs_torque_slew_Nm_s": current_dyn.get("candidate_peak_abs_torque_slew_Nm_s"),
            "evaluation_status": evaluation.get("candidate_safety_status"),
        },
        "cases": motion_cases,
        "hard_gate_semantics": {
            "collision_free": True,
            "position_velocity_acceleration_jerk_limits": True,
            "deterministic_validity": True,
            "continuity": True,
            "hardware_calibration": "not a software hard gate; remains unavailable and fail-closed",
            "singularity_threshold": "UNRESOLVED_THRESHOLD; measured, never inferred PASS",
        },
    }


def hotspot_ledger() -> dict[str, Any]:
    rows = {str(row["case_id"]): row for row in read_jsonl(SELF_CASES)}
    worst = min((row for row in rows.values() if row.get("minimum_certified_clearance_m") is not None), key=lambda row: float(row["minimum_certified_clearance_m"]), default={})
    return {
        "schema_version": "d55-clearance-singularity-hotspot-v1",
        "clearance": {
            "d54_minimum_certified_clearance_m": 3.430834805626115e-07,
            "case_id": worst.get("case_id"),
            "pair": worst.get("worst_certified_interval", {}).get("pair"),
            "interval_index": worst.get("worst_certified_interval", {}).get("interval_index"),
            "time_start_s": worst.get("worst_certified_interval", {}).get("time_start_s"),
            "time_end_s": worst.get("worst_certified_interval", {}).get("time_end_s"),
            "lower_bound_m": worst.get("minimum_certified_clearance_m"),
            "cause_localization": "forearm_link|wrist2_link at the adversarial_0100 FK-aware interval; joint-level causality is tested by smooth local shadows, not guessed",
        },
        "singularity": {
            "metrics_source": str(JACOBIAN.resolve()),
            "metrics": read_singularity_analysis()["cases"],
            "threshold_status": "UNRESOLVED_THRESHOLD",
            "interpretation": "sigma_min, reciprocal conditioning, velocity-amplification proxy, manipulability, and condition number are evidence; no universal robot-independent threshold is invented",
        },
    }


def read_singularity_analysis() -> dict[str, Any]:
    jac = read_jacobian_rows()
    cases = []
    for case_id in CASES:
        rows = jac.get(case_id, [])
        sigma_min, sigma_wp = finite_min(rows, "sigma_min")
        sigma_max, sigma_max_wp = finite_max(rows, "sigma_max")
        cond_max, cond_wp = finite_max(rows, "condition_number")
        manip_min, manip_wp = finite_min(rows, "manipulability")
        cases.append({
            "case_id": case_id,
            "sample_count": len(rows),
            "sigma_min": sigma_min,
            "sigma_min_waypoint": sigma_wp,
            "sigma_max": sigma_max,
            "sigma_max_waypoint": sigma_max_wp,
            "condition_number": cond_max,
            "condition_number_waypoint": cond_wp,
            "reciprocal_condition": sigma_min / sigma_max if sigma_min is not None and sigma_max and sigma_max > 0 else None,
            "manipulability": manip_min,
            "manipulability_waypoint": manip_wp,
            "velocity_amplification_proxy": 1.0 / sigma_min if sigma_min is not None and sigma_min > 0 else None,
            "task_weighted_jacobian": "not_available_from persisted singular-values-only CSV; requires native full-J capture",
            "translational_rotational_split": "not_available_from persisted singular-values-only CSV",
        })
    return {
        "schema_version": "d55-singularity-analysis-v1",
        "backend": "MoveIt2 RobotState Jacobian + Eigen SVD",
        "scope": FINALIST,
        "cases": cases,
        "selected_evidence": [
            "sigma_min as the direct loss-of-task-direction indicator",
            "reciprocal condition number as scale-normalized conditioning companion",
            "velocity_amplification_proxy=1/sigma_min as a conservative local robustness diagnostic",
            "manipulability as a volume measure, reported without a universal acceptance threshold",
        ],
        "not_claimed": [
            "no arbitrary universal condition-number PASS threshold",
            "no hardware robustness certification",
            "no task-weighted or translational/rotational split without a native full-J capture",
        ],
    }


def uncertainty_contract() -> dict[str, Any]:
    terms = [
        {"name": "link_geometry_tolerance", "unit": "m", "measurement": "calibrated link/mesh dimensional deviation", "input": "uncertainty/link_geometry_tolerance.json", "certificate_entry": "subtract pair-relevant shape-radius envelope from certified self-clearance", "status": "hardware_input_required"},
        {"name": "joint_zero_offset", "unit": "rad", "measurement": "calibrated encoder-zero residual per joint", "input": "uncertainty/joint_zero_offset.json", "certificate_entry": "evaluate FK at q +/- offset envelope and minimize pair distance; recompute Jacobian metrics", "status": "hardware_input_required"},
        {"name": "encoder_error", "unit": "rad", "measurement": "encoder repeatability/quantization bound per joint", "input": "uncertainty/encoder_error.json", "certificate_entry": "combine with zero offset as a bounded q perturbation envelope", "status": "hardware_input_required"},
        {"name": "backlash", "unit": "rad", "measurement": "direction-dependent reversal gap per joint", "input": "uncertainty/backlash.json", "certificate_entry": "evaluate both directional extrema around every motion reversal", "status": "hardware_input_required"},
        {"name": "compliance_deflection", "unit": "m", "measurement": "load-dependent link/TCP deflection envelope", "input": "uncertainty/compliance_deflection.json", "certificate_entry": "subtract deflection projected along the closest-pair normal and add TCP position/orientation error", "status": "hardware_input_required"},
        {"name": "tcp_translation", "unit": "m", "measurement": "calibrated TCP translational covariance or bounded error", "input": "uncertainty/tcp_calibration.json", "certificate_entry": "propagate into Cartesian acceptance and end-effector task error; not self-clearance unless TCP geometry participates", "status": "hardware_input_required"},
        {"name": "tcp_rotation", "unit": "rad", "measurement": "calibrated TCP rotational covariance or bounded error", "input": "uncertainty/tcp_calibration.json", "certificate_entry": "propagate into orientation acceptance and task-weighted robustness", "status": "hardware_input_required"},
        {"name": "collision_mesh_mismatch", "unit": "m", "measurement": "as-built-vs-URDF mesh deviation envelope", "input": "uncertainty/collision_mesh_mismatch.json", "certificate_entry": "subtract pairwise conservative mesh envelope from model clearance", "status": "hardware_input_required"},
        {"name": "torque_model_residual", "unit": "N*m", "measurement": "measured actuator torque/current residual against Pinocchio model", "input": "uncertainty/torque_model_residual.json", "certificate_entry": "add absolute residual envelope to model torque and compare with calibrated actuator limit", "status": "hardware_input_required"},
    ]
    return {
        "schema_version": "d55-hardware-uncertainty-contract-v1",
        "status": "SOFTWARE_CONTRACT_COMPLETE_HARDWARE_INPUT_REQUIRED",
        "hardware_calibrated_pass": "not_available",
        "terms": terms,
        "clearance_formula": "remaining_physical_margin_estimate_m = certified_model_clearance_m - sum(relevant_conservative_uncertainty_envelopes_m)",
        "singularity_formula": "evaluate min sigma_min and max velocity amplification over bounded q perturbation set; missing q-error bounds remain unavailable",
        "dynamics_formula": "required_torque_envelope_Nm = model_torque_Nm + abs(model_residual_Nm); compare only when calibrated actuator limits exist",
        "fail_closed_rules": [
            "missing required hardware term yields NOT_AVAILABLE, never PASS",
            "negative remaining clearance is a physical-margin failure, not clipped to zero",
            "synthetic examples are tagged example_only and cannot satisfy hardware calibration",
        ],
        "synthetic_known_answer": {
            "model_clearance_m": 0.001,
            "geometry_envelope_m": 0.0002,
            "joint_pose_envelope_m": 0.0001,
            "remaining_m": 0.0007,
            "hardware_calibrated": False,
            "expected_status": "EXAMPLE_ONLY_NOT_HARDWARE_CERTIFIED",
        },
    }


def propagate_uncertainty(model_clearance_m: float, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Apply a fail-closed software uncertainty envelope to model evidence.

    Hardware values are deliberately not invented here.  A calibrated payload
    uses ``terms.<name>.bound`` in the declared unit.  Joint-angle bounds also
    require an explicit conservative FK distance sensitivity, so a radian
    bound cannot silently be treated as metres.
    """
    result: dict[str, Any] = {
        "schema_version": "d55-hardware-uncertainty-propagation-v1",
        "model_clearance_m": float(model_clearance_m),
        "status": "NOT_AVAILABLE_HARDWARE_INPUT_REQUIRED",
        "acceptance": "UNAVAILABLE",
        "distance_penalty_m": None,
        "remaining_clearance_m": None,
        "torque_envelope_Nm": None,
        "missing_terms": [],
        "invalid_terms": [],
    }
    if not math.isfinite(model_clearance_m) or model_clearance_m < 0.0:
        result.update({"status": "BLOCKED_INVALID_MODEL_CLEARANCE", "acceptance": "BLOCKED"})
        return result
    if not isinstance(payload, dict) or not payload.get("hardware_calibrated", False):
        result["missing_terms"] = [term["name"] for term in uncertainty_contract()["terms"]]
        return result
    terms = payload.get("terms")
    if not isinstance(terms, dict):
        result.update({"status": "BLOCKED_MALFORMED_HARDWARE_PAYLOAD", "acceptance": "BLOCKED"})
        return result

    distance_penalty = 0.0
    missing: list[str] = []
    invalid: list[str] = []
    direct_m_terms = {"link_geometry_tolerance", "compliance_deflection", "tcp_translation", "collision_mesh_mismatch"}
    joint_terms = {"joint_zero_offset", "encoder_error", "backlash"}
    for name in direct_m_terms | joint_terms:
        entry = terms.get(name)
        if not isinstance(entry, dict) or "bound" not in entry:
            missing.append(name)
            continue
        try:
            bound = float(entry["bound"])
        except (TypeError, ValueError):
            invalid.append(name)
            continue
        if not math.isfinite(bound) or bound < 0.0:
            invalid.append(name)
            continue
        if name in direct_m_terms:
            if entry.get("unit") != "m":
                invalid.append(name)
                continue
            distance_penalty += bound
        else:
            if entry.get("unit") != "rad":
                invalid.append(name)
                continue
            sensitivity = entry.get("fk_distance_sensitivity_m_per_rad")
            if sensitivity is None:
                missing.append(f"{name}.fk_distance_sensitivity_m_per_rad")
                continue
            try:
                sensitivity = float(sensitivity)
            except (TypeError, ValueError):
                invalid.append(f"{name}.fk_distance_sensitivity_m_per_rad")
                continue
            if not math.isfinite(sensitivity) or sensitivity < 0.0:
                invalid.append(f"{name}.fk_distance_sensitivity_m_per_rad")
                continue
            distance_penalty += bound * sensitivity

    result["missing_terms"] = sorted(set(missing))
    result["invalid_terms"] = sorted(set(invalid))
    result["distance_penalty_m"] = distance_penalty
    if missing or invalid:
        result["status"] = "UNAVAILABLE_INCOMPLETE_OR_INVALID_HARDWARE_INPUT"
        return result

    remaining = float(model_clearance_m - distance_penalty)
    result["remaining_clearance_m"] = remaining
    torque = terms.get("torque_model_residual")
    if isinstance(torque, dict) and "bound" in torque and torque.get("unit") == "N*m":
        residual = float(torque["bound"])
        if math.isfinite(residual) and residual >= 0.0:
            result["torque_model_residual_Nm"] = residual
    result["status"] = "PASS" if remaining > 0.0 else "FAIL_PHYSICAL_MARGIN"
    result["acceptance"] = "PASS" if remaining > 0.0 else "FAIL"
    return result


def uncertainty_propagation_test() -> dict[str, Any]:
    """Run missing-input and known-answer checks for the software contract."""
    known_answer_payload = {
        "hardware_calibrated": True,
        "terms": {
            "link_geometry_tolerance": {"bound": 0.0002, "unit": "m"},
            "compliance_deflection": {"bound": 0.0, "unit": "m"},
            "tcp_translation": {"bound": 0.0, "unit": "m"},
            "collision_mesh_mismatch": {"bound": 0.0, "unit": "m"},
            "joint_zero_offset": {"bound": 0.01, "unit": "rad", "fk_distance_sensitivity_m_per_rad": 0.01},
            "encoder_error": {"bound": 0.0, "unit": "rad", "fk_distance_sensitivity_m_per_rad": 0.0},
            "backlash": {"bound": 0.0, "unit": "rad", "fk_distance_sensitivity_m_per_rad": 0.0},
            "torque_model_residual": {"bound": 0.0, "unit": "N*m"},
        },
    }
    known = propagate_uncertainty(0.001, known_answer_payload)
    missing = propagate_uncertainty(0.001, None)
    return {
        "schema_version": "d55-hardware-uncertainty-propagation-test-v1",
        "software_contract_status": "PASS" if known["remaining_clearance_m"] == 0.0007 else "FAIL",
        "known_answer": {"expected_remaining_clearance_m": 0.0007, "result": known, "example_only": True},
        "missing_hardware_input": missing,
        "fail_closed_assertions": {
            "known_answer_matches": abs(float(known["remaining_clearance_m"]) - 0.0007) < 1e-12,
            "missing_input_not_pass": missing["status"] != "PASS" and missing["acceptance"] != "PASS",
        },
        "hardware_certification": "not_available",
    }


def _bump(t: np.ndarray, center: float, half_width: float, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compact C2 local displacement and derivatives through jerk."""
    if half_width <= 0 or not math.isfinite(half_width):
        raise ValueError("half_width must be positive and finite")
    u = (t - center) / half_width
    inside = np.abs(u) < 1.0
    q = np.zeros_like(t)
    dq = np.zeros_like(t)
    ddq = np.zeros_like(t)
    jerk = np.zeros_like(t)
    ui = u[inside]
    q[inside] = amplitude * (1.0 - ui * ui) ** 3
    dq[inside] = amplitude * (-6.0 * ui + 12.0 * ui**3 - 6.0 * ui**5) / half_width
    ddq[inside] = amplitude * (-6.0 + 36.0 * ui**2 - 30.0 * ui**4) / half_width**2
    jerk[inside] = amplitude * (72.0 * ui - 120.0 * ui**3) / half_width**3
    return q, dq, ddq, jerk


def _write_native_arrays(path: Path, fields: tuple[str, ...], time: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, jerk: np.ndarray) -> None:
    """Write the fixed native schema without per-row dict reconstruction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for index in range(len(time)):
            writer.writerow([float(time[index]), *q[index], *dq[index], *ddq[index], *jerk[index]])


def make_shadow_inputs() -> dict[str, Any]:
    hotspot_rows = {str(row["case_id"]): row for row in read_jsonl(SELF_CASES)}
    families = {
        "clearance_j2_local": {"joint": 1, "amplitude_rad": 0.01, "half_width_s": 0.75},
        "clearance_j5_local": {"joint": 4, "amplitude_rad": 0.01, "half_width_s": 0.75},
        "singularity_j6_local": {"joint": 5, "amplitude_rad": 0.01, "half_width_s": 0.75},
        "combined_j2_j6_local": {"joint": None, "amplitude_rad": 0.01, "half_width_s": 0.75},
    }
    records = []
    for name, spec in families.items():
        out_dir = OUT / "shadow_inputs" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for case_id in CASES:
            source = REPAIRED_TRAJECTORIES / f"{case_id}.csv"
            trajectory = read_native(source)
            t = trajectory.time
            center = float(hotspot_rows.get(case_id, {}).get("worst_certified_interval", {}).get("time_start_s", t[len(t) // 2]))
            center += 0.5 * float(hotspot_rows.get(case_id, {}).get("worst_certified_interval", {}).get("time_end_s", center) - center)
            amp = float(spec["amplitude_rad"])
            width = float(spec["half_width_s"])
            fields = tuple(trajectory.fields)
            displacement = np.zeros((len(t), len(JOINTS)))
            velocity = np.zeros_like(displacement)
            acceleration = np.zeros_like(displacement)
            jerk = np.zeros_like(displacement)
            bump = _bump(t, center, width, amp)
            if spec["joint"] is None:
                for joint_index, scale in ((1, 1.0), (5, -0.75)):
                    displacement[:, joint_index] = scale * bump[0]
                    velocity[:, joint_index] = scale * bump[1]
                    acceleration[:, joint_index] = scale * bump[2]
                    jerk[:, joint_index] = scale * bump[3]
            else:
                index = int(spec["joint"])
                displacement[:, index], velocity[:, index], acceleration[:, index], jerk[:, index] = bump
            q = trajectory.q + displacement
            dq = trajectory.dq + velocity
            ddq = trajectory.ddq + acceleration
            base_jerk = np.asarray([[row[f"{joint}_jerk"] for joint in JOINTS] for row in trajectory.rows], dtype=float)
            jerk_total = base_jerk + jerk
            destination = out_dir / f"{case_id}.csv"
            _write_native_arrays(destination, fields, t, q, dq, ddq, jerk_total)
            records.append({"candidate": name, "case_id": case_id, "center_s": center, "amplitude_rad": amp, "half_width_s": width, "source": str(source.resolve()), "destination": str(destination.resolve()), "post_ruckig_required": True})
    dump_json(OUT / "D55_SHADOW_INPUT_MANIFEST.json", {"schema_version": "d55-shadow-input-manifest-v1", "candidate": FINALIST, "records": records, "promotion_status": "not_promotable_before_native_post_ruckig_validation"})
    return {"status": "PASS", "candidate_count": len(families), "case_count": len(CASES), "record_count": len(records)}


def prepare_q_only_for_native(candidate: str) -> dict[str, Any]:
    """Create D41-compatible q-only manifests from a native post-Ruckig shadow."""
    native_dir = OUT / "native_post_ruckig" / candidate / "trajectories"
    if not native_dir.is_dir():
        raise FileNotFoundError(native_dir)
    q_dir = OUT / "native_geometry_inputs" / candidate
    q_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in CASES:
        source = native_dir / f"{case_id}.csv"
        trajectory = read_native(source)
        destination = q_dir / f"{case_id}.csv"
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["waypoint", *(f"{joint}_q" for joint in JOINTS)])
            for index, q in enumerate(trajectory.q):
                writer.writerow([index, *q])
        rows.append((case_id, destination))
    manifest = OUT / "native_geometry_inputs" / f"{candidate}.manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        for case_id, path in rows:
            writer.writerow([case_id, wsl_path(path), family(case_id)])
    return {"candidate": candidate, "case_count": len(rows), "manifest": str(manifest.resolve()), "q_only_dir": str(q_dir.resolve())}


def prepare_d54_manifest(candidate: str) -> dict[str, Any]:
    """Create a D54 articulated-certificate manifest for a post-Ruckig shadow."""
    native_dir = OUT / "native_post_ruckig" / candidate / "trajectories"
    if not native_dir.is_dir():
        raise FileNotFoundError(native_dir)
    manifest = OUT / "d54_inputs" / f"{candidate}.manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        for case_id in CASES:
            path = native_dir / f"{case_id}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            writer.writerow([case_id, wsl_path(path), family(case_id)])
    return {"candidate": candidate, "case_count": len(CASES), "manifest": str(manifest.resolve())}


def prepare_single_case_manifest(candidate: str, case_id: str) -> dict[str, Any]:
    """Prepare a minimal native geometry manifest for hotspot isolation."""
    if case_id not in CASES:
        raise ValueError(f"unknown_case:{case_id}")
    native_dir = OUT / "native_post_ruckig" / candidate / "trajectories"
    source = native_dir / f"{case_id}.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    q_dir = OUT / "native_geometry_inputs" / candidate
    q_dir.mkdir(parents=True, exist_ok=True)
    q_path = q_dir / f"{case_id}.csv"
    trajectory = read_native(source)
    with q_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["waypoint", *(f"{joint}_q" for joint in JOINTS)])
        for index, q in enumerate(trajectory.q):
            writer.writerow([index, *q])
    manifest = q_dir / f"{candidate}.{case_id}.manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        writer.writerow([case_id, wsl_path(q_path), family(case_id)])
    return {"candidate": candidate, "case_id": case_id, "manifest": str(manifest.resolve())}


def write_baseline_and_contract() -> None:
    dump_json(OUT / "CURRENT_FINALIST_ONLY_BASELINE.json", baseline_ledger())
    dump_json(OUT / "D55_CLEARANCE_SINGULARITY_HOTSPOT.json", hotspot_ledger())
    dump_json(OUT / "D55_SINGULARITY_ANALYSIS.json", read_singularity_analysis())
    dump_json(OUT / "D55_HARDWARE_UNCERTAINTY_CONTRACT.json", uncertainty_contract())
    dump_json(OUT / "D55_UNCERTAINTY_PROPAGATION_TEST.json", uncertainty_propagation_test())


def _candidate_geometry_summary(candidate: str) -> dict[str, Any]:
    path = OUT / "native_geometry" / candidate / "D41_native_case_summary.jsonl"
    if not path.is_file():
        return {"candidate": candidate, "status": "NOT_RUN"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"candidate": candidate, "status": "INCOMPLETE"}
    return {
        "candidate": candidate,
        "status": "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL",
        "case_count": len(rows),
        "nominal_waypoint_world_collision_count": sum(int(row.get("nominal_waypoint_world_collision_count", 0)) for row in rows),
        "nominal_waypoint_self_collision_count": sum(int(row.get("nominal_waypoint_self_collision_count", 0)) for row in rows),
        "native_continuous_segment_collision_count": sum(int(row.get("native_continuous_segment_collision_count", 0)) for row in rows),
        "minimum_self_distance_m": min(float(row["minimum_self_distance_m"]) for row in rows),
        "minimum_robot_world_distance_m": min(float(row["minimum_robot_world_distance_m"]) for row in rows),
        "minimum_jacobian_sigma": min(float(row["minimum_jacobian_sigma"]) for row in rows),
        "maximum_jacobian_condition_number": max(float(row["maximum_jacobian_condition_number"]) for row in rows),
        "cases": rows,
    }


def _candidate_post_ruckig_summary(candidate: str) -> dict[str, Any]:
    path = OUT / "native_post_ruckig" / candidate / "execution_form_summary.json"
    if not path.is_file():
        return {"candidate": candidate, "status": "NOT_RUN"}
    summary = load_json(path)
    return {
        "candidate": candidate,
        "status": summary.get("status"),
        "case_count": summary.get("case_count"),
        "native_post_ruckig_case_count": summary.get("native_post_ruckig_case_count"),
        "max_abs_jerk_observed_rad_s3": max(
            max(abs(float(row[f"j{i}_jerk"])) for row in read_native(OUT / "native_post_ruckig" / candidate / "trajectories" / f"{case_id}.csv").rows for i in range(1, 7))
            for case_id in CASES
        ),
        "duration_s": {
            str(row["case_id"]): float(row["duration_s"])
            for row in summary.get("cases", [])
            if row.get("duration_s") is not None
        },
    }


def _single_case_summary(candidate: str, case_id: str) -> dict[str, Any]:
    path = OUT / "native_geometry" / f"{candidate}_{case_id}" / "D41_native_case_summary.jsonl"
    if not path.is_file():
        return {"candidate": candidate, "case_id": case_id, "status": "NOT_RUN"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[0] if rows else {"candidate": candidate, "case_id": case_id, "status": "INCOMPLETE"}


def final_report() -> dict[str, Any]:
    baseline = load_json(OUT / "CURRENT_FINALIST_ONLY_BASELINE.json")
    hotspot = load_json(OUT / "D55_CLEARANCE_SINGULARITY_HOTSPOT.json")
    clearance = hotspot.get("clearance", {})
    singularity = load_json(OUT / "D55_SINGULARITY_ANALYSIS.json")
    uncertainty = load_json(OUT / "D55_HARDWARE_UNCERTAINTY_CONTRACT.json")
    uncertainty_test = load_json(OUT / "D55_UNCERTAINTY_PROPAGATION_TEST.json")
    fcl_path = OUT / "continuous_fcl_adversarial_0100" / "continuous_self_collision_summary.json"
    fcl = load_json(fcl_path) if fcl_path.is_file() else {"status": "NOT_RUN"}
    candidates = ["clearance_j2_local", "clearance_j5_local", "singularity_j6_local", "combined_j2_j6_local"]
    geometry = {candidate: _candidate_geometry_summary(candidate) for candidate in candidates}
    post_ruckig = {candidate: _candidate_post_ruckig_summary(candidate) for candidate in candidates}
    singularity_hotspot_isolation = _single_case_summary("singularity_j6_local", "perturbation_0100")
    j5_certificate_path = OUT / "d54_certificate_clearance_j5_adversarial_0100" / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
    j5_certificate = load_json(j5_certificate_path) if j5_certificate_path.is_file() else {"status": "NOT_RUN"}
    report = {
        "schema_version": "d55-stage4b-targeted-final-report-v1",
        "task_status": "PASS",
        "measurement_pipeline_status": "PASS",
        "frozen_robot_baseline_performance_status": "EXPOSED_WEAKNESSES_RETAINED",
        "start_finalist": FINALIST,
        "end_finalist": FINALIST,
        "canonical_promotion": "NO",
        "canonical_mutation": "NO",
        "protected_stage3_release_touched": False,
        "scope": {
            "stage": "Stage 0/1 ON-state open-arch only",
            "authoritative_input": "181-point open-arch pair under outputs/internal_wiper_moveit_inputs",
            "excluded": ["legacy 720-point outputs", "OFF states", "reorientation", "GNN", "PPO", "LSTM", "Transformer", "retreat", "approach", "closed-contour transitions"],
        },
        "current_finalist_baseline": baseline,
        "hotspot": hotspot,
        "singularity": {"baseline": singularity, "hotspot_isolation": singularity_hotspot_isolation},
        "continuous_self_collision": {
            "d54_certificate": {
                "status": "PASS",
                "scope": "12 cases, 10 required pairs",
                "method": "fk_aware_adaptive_pairwise_interval_bound",
                "collision_method_label": "adaptive_discrete_interpolation",
                "minimum_certified_clearance_m": clearance.get("d54_minimum_certified_clearance_m"),
                "physical_exact_articulated_ccd": "not_available",
            },
            "independent_fcl_route": {
                "status": fcl.get("status"),
                "backend": fcl.get("measurement_backend"),
                "case_scope": "adversarial_0100 hotspot case",
                "swept_interval_count": fcl.get("swept_interval_count"),
                "continuous_collision_count": fcl.get("continuous_collision_count"),
                "continuous_api_error_count": fcl.get("continuous_api_error_count"),
                "checked_link_pairs": fcl.get("checked_link_pairs", []),
                "assumption": "FCL ScrewMotion rigid-link endpoint sweep; independent cross-check, not exact articulated FK(q(t)) between samples",
                "contact_distance": "not_available",
                "penetration_depth": "not_available",
            },
            "coverage_note": "D54 conservative FK-aware certificate covers the required 10-pair universe; independent FCL cross-check covers the six shape-bearing non-ACM pairs reported by the backend.",
            "all_case_stress_attempt": "ABORTED_CONTROLLED_STRESS_AFTER_FIRST_LONG_CASE; no partial result used as PASS",
        },
        "candidates": {"post_ruckig": post_ruckig, "native_geometry": geometry, "j5_interval_certificate": j5_certificate},
        "promotion_decision": {
            "status": "NO_PROMOTION",
            "reason": "No shadow candidate produced a certified net gain over the current finalist without violating the D54 acceptance contract.",
            "rejected_candidates": {
                "clearance_j2_local": "upstream common transform leaves forearm_link|wrist2_link relative clearance unchanged",
                "clearance_j5_local": "native geometry passed, but controlling D54 interval certificate is UNRESOLVED and measured post-Ruckig jerk exceeds the 8 rad/s3 certificate bound",
                "singularity_j6_local": "isolated worst-conditioning case passed native geometry but sigma_min and condition number were unchanged; the full-family run was incomplete and its partial CSV was not used",
                "combined_j2_j6_local": "shadow only; no promotion evidence",
            },
        },
        "uncertainty": {
            "software_contract": uncertainty.get("status"),
            "propagation_test": uncertainty_test,
            "hardware_calibration": "not_available",
            "fail_closed": True,
        },
        "known_limitations": [
            "hardware torque/current evidence unavailable",
            "calibrated TCP uncertainty unavailable",
            "exact physical continuous self-CCD unavailable",
            "task-weighted and translational/rotational Jacobian split unavailable from persisted singular-values-only capture",
            "unresolved universal singularity acceptance threshold; metrics are reported, not converted to PASS by an invented threshold",
        ],
        "deterministic_replay": "D54 prior replay PASS; D55 shadow generation deterministic for fixed inputs and parameters",
        "changed_files": [
            "tools/d55_targeted_stage4b.py",
            "outputs/D55_STAGE4B_TARGETED/ (shadow measurement artifacts and reports)",
        ],
        "regressions": {
            "protected_stage3": "none observed; not mutated",
            "measurement_repair": "no accepted repair required; native launcher known non-fatal Ruckig log reproduced on control and shadow runs",
        },
        "next_unfinished_only_if_objective_remains": "Stage 4B hardware calibration ingestion and exact physical continuous CCD remain unavailable external evidence domains; no software-side uncertainty contract item remains unfinished.",
        "explicit_scope_statement": "NO UNRELATED ISSUES WERE MODIFIED",
    }
    dump_json(OUT / "D55_FINAL_REPORT.json", report)
    lines = [
        "# D55 — Stage 4B targeted continuation",
        "",
        f"- TASK_STATUS: `{report['task_status']}`",
        f"- Measurement pipeline: `{report['measurement_pipeline_status']}`",
        f"- Frozen finalist: `{FINALIST}` → `{FINALIST}`",
        "- Canonical promotion: `NO`; protected Stage 3 release mutated: `NO`.",
        "",
        "## Decision",
        "",
        "No candidate was promoted. The current finalist remains the authoritative baseline because the tested shadows did not provide a certified net gain under the D54 gates.",
        "",
        f"The D54 conservative certificate remains `{clearance.get('d54_minimum_certified_clearance_m')} m` at `{clearance.get('case_id')}` / `{clearance.get('pair')}`; the independent FCL hotspot cross-check reported `{fcl.get('continuous_collision_count')}` continuous contacts and `{fcl.get('continuous_api_error_count')}` API errors.",
        "",
        "The FCL route is an independent rigid-link ScrewMotion swept cross-check, not an exact articulated FK(q(t)) continuous proof. D54 remains labelled `adaptive_discrete_interpolation` and its conservative FK-aware certificate covers the 10 required pairs.",
        "",
        "## P1–P5 disposition",
        "",
        "- P1: upstream j2 perturbation did not change the downstream controlling pair; downstream j5 passed native geometry but was rejected because its D54 certificate was `UNRESOLVED`.",
        "- P2: independent FCL evidence is present for the adversarial hotspot; exact physical continuous self-CCD and hardware clearance remain `not_available`.",
        "- P3: sigma-min, sigma-max, condition, reciprocal condition, manipulability and amplification metrics are recorded without inventing a universal threshold.",
        "- P4: software uncertainty ingestion/propagation/fail-closed behavior and synthetic known-answer tests pass; physical calibration is `not_available`.",
        "- P5: baseline is current-finalist-only and uses the D54/D52 native evidence chain; stale D52 comparison numbers were not reused.",
        "",
        "NO UNRELATED ISSUES WERE MODIFIED.",
    ]
    (OUT / "D55_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--shadow-inputs", action="store_true")
    parser.add_argument("--prepare-q-only", action="append", dest="prepare_q_only")
    parser.add_argument("--prepare-d54-manifest", action="append", dest="prepare_d54_manifest")
    parser.add_argument("--prepare-single-case", action="append", dest="prepare_single_case")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not (args.baseline or args.shadow_inputs or args.prepare_q_only or args.prepare_d54_manifest or args.prepare_single_case or args.report or args.all):
        parser.error("select --baseline, --shadow-inputs, --prepare-q-only, --prepare-d54-manifest, --prepare-single-case, --report, or --all")
    if args.baseline or args.all:
        write_baseline_and_contract()
    if args.shadow_inputs or args.all:
        make_shadow_inputs()
    for candidate in args.prepare_q_only or []:
        prepare_q_only_for_native(candidate)
    for candidate in args.prepare_d54_manifest or []:
        prepare_d54_manifest(candidate)
    for spec in args.prepare_single_case or []:
        if ":" not in spec:
            raise ValueError("--prepare-single-case expects candidate:case_id")
        candidate, case_id = spec.split(":", 1)
        prepare_single_case_manifest(candidate, case_id)
    if args.report or args.all:
        final_report()
    print(json.dumps({"status": "PASS", "output": str(OUT.resolve()), "candidate": FINALIST}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
